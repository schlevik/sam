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

import argparse
import glob
import logging
import os
import random
import timeit

import click
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

from scripts.utils import get_output_predictions_file_name

logger = logging.getLogger(__name__)
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def _is_gpu_available():
    import torch
    try:
        gpu = torch.cuda.is_available()
        if gpu:
            str(torch.rand(1).to(torch.device("cuda")))
    except:
        gpu = False
    return gpu


@click.command()
@click.option('--in-file', type=str)
@click.option('--out-folder', type=str, default='')
@click.option('--model-path', type=str)
@click.option('--model-type', type=str)
@click.option('--no-cuda', type=bool, default=None)
@click.option('--do-lower-case', is_flag=True, default=False)
@click.option('--per-gpu-eval-batch-size', type=int, default=8)
@click.option('--lang-id', type=int, default=0)
@click.option('--v2', is_flag=True, default=False)
@click.option('--n-best-size', type=int, default=5)
@click.option('--max-answer-length', type=int, default=30)
@click.option('--verbose-logging', is_flag=True, default=False)
@click.option('--null-score-diff-threshold', type=float, default=0.0)
def predictions(in_file, model_path, model_type, out_folder, no_cuda, do_lower_case, per_gpu_eval_batch_size,
                lang_id, v2, n_best_size, max_answer_length, verbose_logging, null_score_diff_threshold):
    if no_cuda is None:
        no_cuda = not _is_gpu_available()

    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    n_gpu = 0 if no_cuda else torch.cuda.device_count()

    model = get_model(model_path)
    tokenizer = get_tokenizer(model_path, do_lower_case)

    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)

    # Note that DistributedSampler samples randomly
    dataset, examples, features = load_examples(in_file)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)
    model.to(device)
    # multi-gpu evaluate
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    click.echo(f"Generating predictions for model {click.style(model_path, fg='blue')}, "
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

            if model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * lang_id).to(device)}
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
    logger.info("  Evaluation done in total %f secs (%f sec per example)", eval_time, eval_time / len(dataset))

    out_file = get_output_predictions_file_name(in_file, out_folder)

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    # Compute predictions
    file_name = os.path.basename(out_file)
    output_prediction_file = os.path.join(out_folder, file_name)
    output_nbest_file = os.path.join(out_folder, f"nbest-{file_name}")

    if v2:
        output_null_log_odds_file = os.path.join(out_folder, f"null-odds-{file_name}")
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            n_best_size,
            max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            v2,
            tokenizer,
            verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            n_best_size,
            max_answer_length,
            do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            verbose_logging,
            v2,
            null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    # results = squad_evaluate(examples, predictions)
    # return results


def get_tokenizer(name_or_model_name_or_path, do_lower_case):
    return AutoTokenizer.from_pretrained(
        # args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        name_or_model_name_or_path,
        # do_lower_case=args.do_lower_case,
        do_lower_case=do_lower_case,
        # cache_dir=args.cache_dir if args.cache_dir else None,
        cache_dir=None
    )


def get_model(model_name_or_path):
    config = AutoConfig.from_pretrained(
        # args.config_name if args.config_name else args.model_name_or_path,
        model_name_or_path,
        # cache_dir=args.cache_dir if args.cache_dir else None,
        cache_dir=None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        # args.model_name_or_path,
        model_name_or_path,
        # from_tf=bool(".ckpt" in args.model_name_or_path),
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        # cache_dir=args.cache_dir if args.cache_dir else None,
        cache_dir=None,
    )
    return model


@click.command()
@click.argument("in_files", nargs=-1)
@click.option("--out-folder")
@click.option('--model-path', type=str)
@click.option('--do-lower-case', is_flag=True, default=False)
@click.option('--evaluate', is_flag=True, default=False)
@click.option('--v2', is_flag=True, default=False)
@click.option('--max-seq-length', type=int, default=384)
@click.option('--doc-stride', type=int, default=128)
@click.option('--max-query-length', type=int, default=64)
@click.option('--num-workers', type=int, default=1)
def cache_examples(in_files, out_folder, model_path, do_lower_case, evaluate, v2, max_seq_length, doc_stride,
                   max_query_length, num_workers):
    tokenizer = get_tokenizer(model_path, do_lower_case)
    processor = SquadV2Processor() if v2 else SquadV1Processor()
    for in_file in in_files:
        data_dir = os.path.dirname(in_file)
        file_name = os.path.basename(in_file)
        if evaluate:
            examples = processor.get_dev_examples(data_dir, filename=file_name)
        else:
            examples = processor.get_train_examples(data_dir, filename=file_name)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=num_workers,
        )
        out_file = os.path.join(out_folder, f"{os.path.splitext(file_name)[0]}-"
                                            f"{os.path.basename(os.path.normpath(model_path))}.bin")
        click.echo(f"Saving features into cached file {click.style(out_file, fg='blue')}")
        if os.path.dirname(out_file).replace(".", ""):
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, out_file)


def load_examples(location):
    features_and_dataset = torch.load(location)
    features, dataset, examples = (
        features_and_dataset["features"],
        features_and_dataset["dataset"],
        features_and_dataset["examples"],
    )
    return dataset, examples, features
