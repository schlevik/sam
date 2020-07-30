import random
from itertools import groupby
from typing import List, Type

import click
import numpy as np
from loguru import logger

from scripts.utils import write_json, EvalMetricParam, match_prediction_to_gold, extract_model_name
from stresstest.eval_utils import get_mean_var_ci, get_mean_var_ci_bernoulli, EvalMetric

from stresstest.util import load_json, sample_iter


def _get_score(gold, predictions, metrics: List[Type[EvalMetric]], subsample=None):
    result = dict()
    for metric_class in metrics:
        metric_name = metric_class.__name__
        metric = metric_class()

        metric_results = np.array(metric(gold, predictions))
        if metric.binary:
            mean, var, ci = get_mean_var_ci_bernoulli(metric_results)
        else:
            mean, var, ci = get_mean_var_ci(metric_results)
        printable_result = f'{mean:.4f} +/- {ci:.4f}'

        click.echo(f"Mean under the {click.style(metric_name, fg='green')} metric on "
                   f"{f'subsample of {subsample}' if subsample else 'full sample'}: "
                   f"{click.style(printable_result, fg='green', bold=True)}")

        result[metric_name] = {
            'human_readable': printable_result,
            'mean': mean,
            'variance': var,
            '95ci': ci
        }
    return result


def reasoning_key(d):
    return d.qa['reasoning']


def num_modifier_key(d):
    return d.qa['num_modifications']


@click.command()
@click.argument("gold-files", nargs=-1, type=str)
@click.option("--prediction-folder", type=str, default="data/predictions")
@click.option("--output", type=str, default="metrics/result.json")
@click.option("--metric", type=EvalMetricParam(), default='EM,F1,EMRelaxed')
@click.option("--split-reasoning", is_flag=True, default=False)
@click.option("--split-num-modifier", is_flag=True, default=False)
def evaluate(prediction_folder, gold_files, output, metric, split_reasoning, split_num_modifier):
    eval_metrics = metric
    result = dict()
    for gold_file in gold_files:
        click.echo(f"Evaluating predictions on {click.style(gold_file, fg='blue')}")
        gold = list(sample_iter(load_json(gold_file)))
        gold_descriptor, prediction_files = match_prediction_to_gold(gold_file, prediction_folder)
        result[gold_file] = {'n': len(gold)}
        logger.debug(prediction_files)
        for prediction_file in sorted(prediction_files):
            model_name = extract_model_name(gold_descriptor, prediction_file)
            result[gold_file][model_name] = dict()
            click.echo(f"Evaluating predictions of model {click.style(model_name, fg='green')}")
            predictions = load_json(prediction_file)
            result[gold_file][model_name]['full'] = _get_score(gold, predictions, eval_metrics)
            click.echo()
            if split_reasoning:
                for reasoning, gold_split in groupby(sorted(gold, key=reasoning_key), key=reasoning_key):
                    result[gold_file][model_name][reasoning] = _get_score(list(gold_split), predictions, eval_metrics,
                                                                          reasoning)
                click.echo()
            if split_num_modifier:
                for num_mod, gold_split in groupby(sorted(gold, key=num_modifier_key), key=num_modifier_key):
                    result[gold_file][model_name][num_mod] = _get_score(list(gold_split), predictions, eval_metrics,
                                                                        f"Num modifications: {num_mod}")
            click.echo()
        write_json(result, output)
