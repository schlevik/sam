import json
from collections import defaultdict
from copy import deepcopy

import click
from loguru import logger
import torch
from ax.service.ax_client import AxClient
from tqdm import trange

from scripts.predict_transformers import evaluate
from scripts.train_transformers import do_train
from scripts.utils_transformers import Args, get_model, get_tokenizer, set_seed, \
    debug_features_examples_dataset, load_or_convert
from scripts.utils import write_json, get_baseline_intervention_control_from_baseline
from stresstest.eval_utils import align, evaluate_intervention, get_mean_var_ci_bernoulli, get_mean_var_ci, EMRelaxed
from stresstest.util import load_json


def train_and_eval_single_step(args: Args, train_dataset, aligned_baseline, aligned_intervention, aligned_control,
                               baseline_dataset, intervention_dataset, control_dataset, original_dev_dataset,
                               baseline_gold_path, intervention_gold_path, control_gold_path,
                               run_nr=0, train=True, num_runs=1, evaluate_on='eoi'):
    results = []
    for i in range(num_runs):
        set_seed(args)
        # load model
        model = get_model(args.model_path)
        tokenizer = get_tokenizer(args.model_path, do_lower_case=args.do_lower_case)
        model.to(args.device)
        # train

        if train:
            step, loss = do_train(args, train_dataset, model, tokenizer)

        args.eval_file = baseline_gold_path
        if evaluate_on == 'eoi' or evaluate_on == 'baseline':
            baseline_predictions = evaluate(args, model, tokenizer, *baseline_dataset, f'baseline-{run_nr}',
                                            return_raw=True)
        args.eval_file = intervention_gold_path

        if evaluate_on == 'eoi' or evaluate_on == 'intervention':
            intervention_predictions = evaluate(args, model, tokenizer, *intervention_dataset, f'intervention-{run_nr}',
                                                return_raw=True)

        args.eval_file = control_gold_path
        if evaluate_on == 'eoi':
            control_predictions = evaluate(args, model, tokenizer, *control_dataset, f'control-{run_nr}',
                                           return_raw=True)

            # obtain predictions on all three of them
            (overall_results, results_baseline, results_intervention, results_control,
             correct_before_intervention, correct_change_correct, correct_keep_wrong, correct_change_wrong,
             wrong_change_right, wrong_keep_right, correct_baseline_control, correct_baseline_control_intervention
             ) = evaluate_intervention(aligned_baseline, aligned_intervention, aligned_control,
                                       baseline_predictions, intervention_predictions, control_predictions)

            mean, *_ = get_mean_var_ci_bernoulli(overall_results)
            # there's no point to evaluate multiple times if not training

            result = {
                "overall": mean,
                'consistency': len(correct_change_correct) / len(aligned_baseline),
                'consistency+control': len(correct_baseline_control_intervention) / len(aligned_baseline),

                "acc_baseline": sum(results_baseline) / len(results_baseline),
                "acc_intervention": sum(results_intervention) / len(results_intervention),
                "acc_control": sum(results_control) / len(results_control),
                'correct->change->correct': len(correct_change_correct),
                'correct(baseline+control)/correct(baseline)': len(correct_baseline_control) / sum(results_baseline),
                'correct+control->change->correct': len(correct_baseline_control_intervention),
            }
        elif evaluate_on == 'baseline':
            metric_results = EMRelaxed(max_length=args.max_answer_length)(aligned_baseline, baseline_predictions)
            result = {"EMRelaxed": get_mean_var_ci_bernoulli(metric_results)[0]}
        elif evaluate_on == 'intervention':
            metric_results = EMRelaxed(max_length=args.max_answer_length)(aligned_baseline, baseline_predictions)
            result = {"EMRelaxed": get_mean_var_ci_bernoulli(metric_results)[0]}
        else:
            raise NotImplementedError()
        if original_dev_dataset is not None:
            original_dev_result = evaluate(args, model, tokenizer, *original_dev_dataset, f'original-dev-{run_nr}',
                                           return_raw=False)
            result['original'] = original_dev_result['exact']
        results.append(result)
    if num_runs == 1:
        return results[0]

    if evaluate_on == 'eoi':
        final_result = {
            "overall": get_mean_var_ci([r['overall'] for r in results]),
            "acc_baseline": get_mean_var_ci([r['acc_baseline'] for r in results]),
            "acc_intervention": get_mean_var_ci([r['acc_intervention'] for r in results]),
            "acc_control": get_mean_var_ci([r['acc_control'] for r in results]),
            'correct->change->correct': get_mean_var_ci([r['correct->change->correct'] for r in results]),
            'correct(baseline+control)/correct(baseline):': get_mean_var_ci(
                [r['correct(baseline+control)/correct(baseline)'] for r in results]),
            'correct+control->change->correct': get_mean_var_ci(
                [r['correct+control->change->correct'] for r in results]),
        }
    else:
        final_result = {
            key: get_mean_var_ci([r[key] for r in results]) for key in results[0].keys()
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
@click.option("--max-answer-length", type=int, default=5)
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
@click.option('--mute', type=bool, is_flag=True, default=False)
@click.option('--max-seq-length', type=int, default=384)
@click.option('--doc-stride', type=int, default=128)
@click.option('--max-query-length', type=int, default=64)
@click.option('--num-workers', type=int, default=8)
@click.option('--debug-features', type=bool, is_flag=True, default=False)
@click.option('--do-not-lower-case', type=bool, is_flag=True, default=False)
@click.option('--runs-per-trial', type=int, default=1)
@click.option('--evaluate-on', type=str, default='eoi')
@click.option('--original-dev-dataset', type=str, default=None)
@click.option('--optimize-consistency', is_flag=True, default=False)
def finetune(optimize_consistency, evaluate_on, original_dev_dataset, runs_per_trial, hyperparam_opt_runs, out_file,
             mute, baseline_gold_file, hyperparams, keep_predictions, **kwargs):
    gold_files = get_baseline_intervention_control_from_baseline(baseline_gold_file)

    golds = tuple(
        load_json(g) for g in gold_files
    )
    # load eval gold for evaluation
    aligneds = align(*golds, assert_same=True)

    hyper_params = [{
        'name': hp['name'],
        'type': hp.get("type", 'range'),
        'bounds': hp['bounds'],
        'value_type': hp.get('value_type', 'float'),
        'log_scale': hp.get('log_scale', True)
    } for hp in json.loads(hyperparams)]

    logger.info(hyper_params)

    args = Args(**kwargs)

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
    if args.seed:
        set_seed(args)
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
    if original_dev_dataset:
        args.eval_file = original_dev_dataset
        original_dev_dataset = load_or_convert(args, tokenizer, evaluate=True)
    ax_client = AxClient()
    ax_client.create_experiment(
        name=f'{args.model_path}@{args.train_file}',
        parameters=hyper_params,
        objective_name=evaluate_on,
        minimize=False,
    )
    result = {"trials": [], "tried_params": defaultdict(list), "best_params": ...,
              'pre_eval': train_and_eval_single_step(args, train_dataset, *aligneds, *features, original_dev_dataset,
                                                     *gold_files, run_nr='eval', train=False,
                                                     evaluate_on=evaluate_on)}
    # first, eval and save what is the performance before training

    click.echo(f"Results: {json.dumps(result['pre_eval'], indent=4)}")
    # run hyperparam optimisation
    predictions_folder = keep_predictions
    for i in trange(hyperparam_opt_runs):
        parameters, trial_index = ax_client.get_next_trial()
        logger.info(f"Trying parameters: {parameters}")
        single_step_args = deepcopy(kwargs)
        single_step_args.update(parameters)
        args = Args(**single_step_args)
        args.predictions_folder = str(predictions_folder)
        trial_result = train_and_eval_single_step(args, train_dataset, *aligneds, *features, original_dev_dataset,
                                                  *gold_files, run_nr=i, num_runs=runs_per_trial,
                                                  evaluate_on=evaluate_on)
        #
        if optimize_consistency:
            assert evaluate_on == 'eoi'
            mean = trial_result['consistency']
        else:
            mean = trial_result['overall' if evaluate_on == 'eoi' else 'EMRelaxed']
        if runs_per_trial > 1:
            mean, var, ci = mean
        if original_dev_dataset:
            logger.info(f"Mean: ({mean} * 100 + {trial_result['original']})/2")
            mean = (mean * 100 + trial_result['original']) / 2

        trial_result["mean"] = mean

        logger.info(f"Result: {mean}")
        logger.info(f"Results: {json.dumps(trial_result, indent=4)}")
        result["trials"].append(trial_result)
        result['tried_params'][i].append(parameters)
        ax_client.complete_trial(trial_index=trial_index, raw_data=mean)
    best_params, metrics = ax_client.get_best_parameters()
    result['best_params'] = best_params
    result['best_metrics'] = metrics
    click.echo(f"What is metrics? {metrics}")
    click.echo(json.dumps(result, indent=4))
    write_json(result, out_file)
