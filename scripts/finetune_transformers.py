import json
import logging
import tempfile
from collections import defaultdict
from copy import deepcopy

import click
from loguru import logger
import torch
from ax.service.ax_client import AxClient
from tqdm import trange

from scripts.predict_transformers import evaluate
from scripts.train_transformers import do_train
from scripts.utils_transformers import load_examples, Args, get_model, get_tokenizer, set_seed, \
    debug_features_examples_dataset, load_or_convert
from scripts.utils import write_json, get_baseline_intervention_control_from_baseline
from stresstest.eval_utils import align, evaluate_intervention, get_mean_var_ci_bernoulli, get_mean_var_ci
from stresstest.util import load_json


def train_and_eval_single_step(args: Args, train_dataset, aligned_baseline, aligned_intervention, aligned_control,
                               baseline_dataset, intervention_dataset, control_dataset, baseline_gold_path,
                               intervention_gold_path, control_gold_path, run_nr=0, keep_predictions=False,
                               train=True, num_runs=1):
    # load model
    model = get_model(args.model_path)
    tokenizer = get_tokenizer(args.model_path, do_lower_case=args.do_lower_case)
    print(args.device)
    model.to(args.device)
    # train
    results = []
    for i in range(num_runs):
        if train:
            step, loss = do_train(args, train_dataset, model, tokenizer)
        if keep_predictions:
            args.eval_file = baseline_gold_path
        baseline_predictions = evaluate(args, model, tokenizer, *baseline_dataset, f'baseline-{run_nr}',
                                        return_raw=True)
        if keep_predictions:
            args.eval_file = intervention_gold_path
        intervention_predictions = evaluate(args, model, tokenizer, *intervention_dataset, f'intervention-{run_nr}',
                                            return_raw=True)
        if keep_predictions:
            args.eval_file = control_gold_path
        control_predictions = evaluate(args, model, tokenizer, *control_dataset, f'control-{run_nr}', return_raw=True)

        # obtain predictions on all three of them
        (overall_results, results_baseline, results_intervention, results_control,
         correct_before_intervention, correct_change_correct, correct_keep_wrong, correct_change_wrong,
         wrong_change_right, wrong_keep_right, correct_baseline_control, correct_baseline_control_intervention
         ) = evaluate_intervention(aligned_baseline, aligned_intervention, aligned_control,
                                   baseline_predictions, intervention_predictions, control_predictions)

        mean, *_ = get_mean_var_ci_bernoulli(overall_results)
        # there's no point to evaluate multiple times if not training

        results.append({
            "overall": mean,
            "acc_baseline": sum(results_baseline) / len(results_baseline),
            "acc_intervention": sum(results_intervention) / len(results_intervention),
            "acc_control": sum(results_control) / len(results_control),
            'correct->change->correct': len(correct_change_correct),
            'correct(baseline+control)/correct(baseline):': len(correct_baseline_control) / sum(results_baseline),
            'correct+control->change->correct': len(correct_baseline_control_intervention),
        })
    final_result = {
        "overall": get_mean_var_ci([r['overall'] for r in results]),
        "acc_baseline": get_mean_var_ci([r['acc_baseline'] for r in results]),
        "acc_intervention": get_mean_var_ci([r['acc_intervention'] for r in results]),
        "acc_control": get_mean_var_ci([r['acc_control'] for r in results]),
        'correct->change->correct': get_mean_var_ci([r['correct->change->correct'] for r in results]),
        'correct(baseline+control)/correct(baseline):': get_mean_var_ci(
            [r['correct(baseline+control)/correct(baseline)'] for r in results]),
        'correct+control->change->correct': get_mean_var_ci([r['correct+control->change->correct'] for r in results]),
    }
    logger.info(final_result)

    return final_result


@click.command()
@click.argument("train-file", type=str)
@click.option('--out-file', type=str)
@click.option("--model-path", type=str)
@click.option("--model-type", type=str)
@click.option("--no-cuda", is_flag=True, type=bool, default=None)
@click.option("--do-not-lower-case", is_flag=True, type=bool, default=False)
@click.option("--per-gpu-eval-batch-size", type=int, default=8)
@click.option("--lang-id", type=int, default=0)
@click.option("--v2", type=bool, is_flag=True, default=False)
@click.option("--overwrite-output-dir", is_flag=True, type=bool, default=False)
@click.option("--max-grad-norm", type=float, default=1.0)
@click.option("--fp16", type=bool, default=False)
@click.option("--max-answer-length", type=int, default=30)
@click.option("--verbose-logging", is_flag=True, type=bool, default=False)
@click.option("--null-score-diff-threshold", type=float, default=0.0)
@click.option("--seed", type=int, default=42)
@click.option("--weight-decay", type=float, default=0.0)
@click.option("--warmup-steps", type=int, default=0)
@click.option("--do-eval-after-training", is_flag=True, type=bool, default=False)
@click.option("--eval-all-checkpoints", is_flag=True, type=bool, default=False)
@click.option("--adam-epsilon", type=float, default=1e-8)
@click.option("--learning-rate", type=float, default=5e-5)
@click.option("--gradient-accumulation-steps", type=int, default=1)
@click.option("--max-steps", type=int, default=-1)
@click.option("--per-gpu-train-batch-size", type=int, default=8)
@click.option("--num-train-epochs", type=int, default=3)
@click.option("--hyperparam-opt-runs", type=int, default=1)
@click.option("--baseline-gold-file", type=str)
@click.option("--hyperparams", type=str, default="[]")
@click.option("--keep-predictions", type=str, default="")
@click.option('--stfu', type=bool, is_flag=True, default=False)
@click.option('--max-seq-length', type=int, default=384)
@click.option('--doc-stride', type=int, default=128)
@click.option('--max-query-length', type=int, default=64)
@click.option('--num-workers', type=int, default=8)
@click.option('--debug-features', type=bool, is_flag=True, default=False)
@click.option('--do-not-lower-case', type=bool, is_flag=True, default=False)
@click.option('--runs-per-trial', type=int, default=1)
def finetune(**kwargs):
    runs_per_trial = kwargs.pop('runs_per_trial')
    num_hpopt_runs = kwargs.pop('hyperparam_opt_runs')
    # feature_files = get_baseline_intervention_control_from_baseline(kwargs.pop("baseline_features_file"))
    out_file = kwargs.pop('out_file')
    # load eval datasets for predictions

    gold_files = get_baseline_intervention_control_from_baseline(kwargs.pop("baseline_gold_file"))

    golds = tuple(
        load_json(g) for g in gold_files
    )
    # load eval gold for evaluation
    aligneds = align(*golds, assert_same=True)

    # run single step
    hyper_params = [{
        'name': hp['name'],
        'type': hp.get("type", 'range'),
        'bounds': hp['bounds'],
        'value_type': hp.get('value_type', 'float'),
        'log_scale': hp.get('log_scale', True)
    } for hp in json.loads(kwargs.pop('hyperparams'))]

    logger.info(hyper_params)
    keep_predictions = kwargs.pop('keep_predictions')

    mute = kwargs.pop('stfu')
    args = Args(**kwargs)
    if args.seed:
        set_seed(args)
    args.debug_features = not mute
    tokenizer = get_tokenizer(args.model_path, args.do_lower_case)
    features = []
    for f in gold_files:
        args.eval_file = f
        features.append(load_or_convert(args, tokenizer, evaluate=True))
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        kwargs['n_gpu'] = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        kwargs['n_gpu'] = 1
    kwargs['device'] = device
    args.n_gpu = kwargs['n_gpu']
    args.device = kwargs['device']

    logger.debug(args)

    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # load train dataset

    train_dataset, train_examples, train_features = load_or_convert(args, tokenizer)
    if not mute:
        debug_features_examples_dataset(train_dataset, train_examples, train_features, tokenizer)
    ax_client = AxClient()
    ax_client.create_experiment(
        name=f'{args.model_path}@{args.train_file}',
        parameters=hyper_params,
        objective_name='Evaluation on Intervention',
        minimize=False,
    )
    result = {
        "trials": [],
        "tried_params": defaultdict(list),
        "best_params": ...,
    }
    # first, eval and save what is the performance before training

    result['pre_eval'] = train_and_eval_single_step(args, train_dataset, *aligneds, *features, *gold_files,
                                                    run_nr='eval', keep_predictions=bool(keep_predictions), train=False)
    click.echo(f"Results: {json.dumps(result['pre_eval'], indent=4)}")
    # run hyperparam optimisation
    with tempfile.TemporaryDirectory() as tempdir:
        predictions_folder = keep_predictions
        for i in trange(num_hpopt_runs):
            parameters, trial_index = ax_client.get_next_trial()
            single_step_args = deepcopy(kwargs)
            single_step_args.update(parameters)
            args = Args(**single_step_args)
            args.predictions_folder = str(predictions_folder)
            trial_result = train_and_eval_single_step(args, train_dataset, *aligneds, *features, *gold_files, run_nr=i,
                                                      keep_predictions=bool(keep_predictions), num_runs=runs_per_trial)
            mean = trial_result['overall'][0]
            click.echo(f"Results: {json.dumps(trial_result, indent=4)}")
            result["trials"].append(trial_result)
            result['tried_params'][i].append(parameters)
            ax_client.complete_trial(trial_index=trial_index, raw_data=mean)
    best_params, metrics = ax_client.get_best_parameters()
    result['best_params'] = best_params
    result['best_metrics'] = metrics
    click.echo(f"What is metrics? {metrics}")
    click.echo(json.dumps(result, indent=4))
    write_json(result, out_file)
