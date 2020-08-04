# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

import os
import timeit

import click
import torch

from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import SquadV1Processor, squad_convert_examples_to_features
from loguru import logger
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult

from scripts.evaluate_intervention import print_examples
from scripts.utils import get_output_predictions_file_name, get_baseline_intervention_control_from_baseline
from scripts.utils_transformers import to_list, _is_gpu_available, get_tokenizer, get_model, load_examples, Args, \
    debug_features_examples_dataset, convert_to_features, load_or_convert
from stresstest.eval_utils import align, evaluate_intervention
from stresstest.util import load_json

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


@click.command()
@click.argument("in-files", nargs=-1)
@click.option('--out-folder', type=str)
@click.option("model_paths", '--model-path', type=str, multiple=True)
@click.option("model_types", '--model-type', type=str, multiple=True)
@click.option('--no-cuda', type=bool, default=None)
@click.option('--per-gpu-eval-batch-size', type=int, default=8)
@click.option('--do-not-lower-case', is_flag=True, default=False)
@click.option('--lang-id', type=int, default=0)
@click.option('--v2', is_flag=True, default=False)
@click.option('--n-best-size', type=int, default=5)
@click.option('--max-answer-length', type=int, default=10)
@click.option('--verbose-logging', is_flag=True, default=False)
@click.option('--null-score-diff-threshold', type=float, default=0.0)
@click.option('--max-seq-length', type=int, default=384)
@click.option('--doc-stride', type=int, default=128)
@click.option('--max-query-length', type=int, default=64)
@click.option('--num-workers', type=int, default=1)
@click.option('--debug-features', type=bool, is_flag=True, default=False)
def predictions(in_files, out_folder, model_paths, model_types, no_cuda, per_gpu_eval_batch_size, do_not_lower_case,
                lang_id, v2, n_best_size, max_answer_length, verbose_logging, null_score_diff_threshold, **kwargs):
    assert len(model_paths) == len(model_types)
    for model_path, model_type in zip(model_paths, model_types):
        model = get_model(model_path)
        args = Args(model_path=model_path, model_type=model_type, predictions_folder=out_folder,
                    no_cuda=no_cuda, do_not_lower_case=do_not_lower_case,
                    per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                    lang_id=lang_id, v2=v2, n_best_size=n_best_size, max_answer_length=max_answer_length,
                    verbose_logging=verbose_logging, null_score_diff_threshold=null_score_diff_threshold, **kwargs)
        tokenizer = get_tokenizer(model_path, args.do_lower_case)
        for in_file in in_files:
            args.eval_file = in_file
            logger.debug(args)
            dataset, examples, features = load_or_convert(args, tokenizer, evaluate=True)
            evaluate(args, model, tokenizer, dataset, examples, features,
                     suffix=os.path.basename(os.path.normpath(model_path)))


def evaluate(args: Args, model, tokenizer, dataset, examples, features, suffix="", return_raw=False):
    if args.no_cuda is None:
        args.no_cuda = not _is_gpu_available()
    if args.predictions_folder:
        assert args.eval_file, "Need name of the eval file to save predictions!"
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    eval_batch_size = args.per_gpu_eval_batch_size * max(1, n_gpu)

    # Note that DistributedSampler samples randomly

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)
    model.to(device)
    # multi-gpu evaluate
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    click.echo(f"Generating predictions for model {click.style(args.model_path, fg='blue')}, "
               f"running on {click.style(str(device), fg='green')}")
    click.echo("  Num examples = %d" % len(dataset))
    click.echo("  Batch size = %d" % eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(device)}
                    )
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    eval_time = timeit.default_timer() - start_time
    logger.info(f"Evaluation done in total {eval_time} secs ({eval_time / len(dataset)} sec per example)")
    eval_file = args.eval_file
    predictions_folder = args.predictions_folder
    v2 = args.v2
    if predictions_folder:
        out_file = get_output_predictions_file_name(eval_file, predictions_folder, suffix)
        logger.info(f"Saving predictions in {out_file}")

        # Compute predictions
        file_name = os.path.basename(out_file)
        output_prediction_file = os.path.join(predictions_folder, file_name)
        output_nbest_file = os.path.join(predictions_folder, f"nbest-{file_name}")

        if v2:
            output_null_log_odds_file = os.path.join(predictions_folder, f"null-odds-{file_name}")
        else:
            output_null_log_odds_file = None
    else:
        logger.info("Not saving predictions...")
        output_prediction_file = None
        output_nbest_file = None
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.v2,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.v2,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    # results = squad_evaluate(examples, predictions)
    # return results
    if return_raw:
        return predictions
    else:
        return squad_evaluate(examples, predictions)


@click.command()
@click.option("--model-path", type=str)
@click.option("--model-type", type=str)
@click.option("--no-cuda", is_flag=True, type=bool, default=None)
@click.option("--baseline-gold-file", type=str)
@click.option("--do-not-lower-case", is_flag=True, type=bool, default=False)
@click.option("--per-gpu-eval-batch-size", type=int, default=8)
@click.option("--max-answer-length", type=int, default=30)
@click.option("--verbose-logging", is_flag=True, type=bool, default=False)
@click.option('--stfu', type=bool, is_flag=True, default=False)
@click.option('--max-seq-length', type=int, default=384)
@click.option('--doc-stride', type=int, default=128)
@click.option('--max-query-length', type=int, default=64)
@click.option('--num-workers', type=int, default=4)
@click.option('--predictions-folder', type=str, default='')
def debug_eval(model_path, model_type, baseline_gold_file, no_cuda, do_not_lower_case, per_gpu_eval_batch_size,
               verbose_logging, max_answer_length, max_seq_length, doc_stride, max_query_length, num_workers, stfu,
               predictions_folder):
    eval_files = get_baseline_intervention_control_from_baseline(baseline_gold_file)
    model = get_model(model_path)
    do_lower_case = not do_not_lower_case
    tokenizer = get_tokenizer(model_path, do_lower_case)
    processor = SquadV1Processor()
    defs = []

    for eval_file in eval_files:
        data_dir = os.path.dirname(eval_file)
        file_name = os.path.basename(eval_file)
        examples = processor.get_dev_examples(data_dir, filename=file_name)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
            return_dataset="pt",
            threads=num_workers,
        )
        defs.append((dataset, examples, features))
    args = Args(model_path, model_type, per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                max_answer_length=max_answer_length, predictions_folder=predictions_folder)

    if args.local_rank == -1 or no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        args.n_gpu = 0 if no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)

    model.to(device=args.device)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    baseline_dataset, intervention_dataset, control_dataset = defs
    if not stfu:
        debug_features_examples_dataset(*baseline_dataset, tokenizer)
    args.eval_file = f'debug-{model_path}-baseline'
    baseline_predictions = evaluate(args, model, tokenizer, *baseline_dataset, return_raw=True)
    args.eval_file = f'debug-{model_path}-intervention'
    intervention_predictions = evaluate(args, model, tokenizer, *intervention_dataset,
                                        return_raw=True)

    args.eval_file = f'debug-{model_path}-control'
    control_predictions = evaluate(args, model, tokenizer, *control_dataset, return_raw=True)
    golds = tuple(
        load_json(g) for g in eval_files
    )
    aligneds = align(*golds)
    # obtain predictions on all three of them
    (overall_results, results_baseline, results_intervention, results_control,
     correct_before_intervention, correct_change_correct, correct_keep_wrong, correct_change_wrong,
     wrong_change_right, wrong_keep_right, correct_baseline_control, correct_baseline_control_intervention
     ) = evaluate_intervention(*aligneds,
                               baseline_predictions, intervention_predictions, control_predictions)
    print_examples(correct_baseline_control, correct_baseline_control_intervention, correct_change_correct,
                   correct_keep_wrong, correct_change_wrong, wrong_change_right,
                   wrong_keep_right)
    click.echo(f"Got {sum(results_baseline)} correct for baseline.")
    click.echo(f"Got {sum(results_intervention)} correct for intervention.")
    click.echo(f"Out of {sum(results_baseline)} correct baseline results, got {len(correct_change_correct)} "
               f"correct after intervention.")
    click.echo(f"Out of {len(correct_baseline_control)} correct for both baseline and control "
               f"got {len(correct_baseline_control_intervention)} correct after intervention.")
    click.echo(f"Interventions that the model 'ignored': {len(correct_keep_wrong)}")
    click.echo(f"Interventions that left the model 'confused': {len(correct_change_wrong)}")
    click.echo(f"Wrong predictions that the model changed to correct: {len(wrong_change_right)}")
    click.echo(f"Wrong predictions that the model didn't change but that became correct: {len(wrong_keep_right)}")
