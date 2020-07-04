import random
from typing import List, Type

import click
import numpy as np
from loguru import logger

from scripts.utils import write_json, EvalMetricParam, match_prediction_to_gold, extract_model_name
from stresstest.eval_utils import get_mean_var_ci, get_mean_var_ci_bernoulli, EvalMetric

from stresstest.util import load_json, sample_iter, num_questions


def _get_score(gold, predictions, metrics: List[Type[EvalMetric]], subsample=None):
    result = dict()
    for metric_class in metrics:
        metric_name = metric_class.__name__
        metric = metric_class()
        gold_iter = random.sample(list(sample_iter(gold)), subsample) if subsample else sample_iter(gold)
        metric_results = np.array(metric(gold_iter, predictions))
        if metric.binary:
            mean, var, ci = get_mean_var_ci_bernoulli(metric_results)
        else:
            mean, var, ci = get_mean_var_ci(metric_results)
        printable_result = f'{mean:.4f} +/- {ci:.4f}'

        click.echo(f"Mean under the {click.style(metric_name, fg='green')} metric on "
                   f"{f'subsample of {subsample}' if subsample else 'full sample'}: "
                   f"{click.style(printable_result, fg='green', bold=True)}")
        click.echo()
        result[metric_name] = {
            'human_readable': printable_result,
            'mean': mean,
            'variance': var,
            '95ci': ci
        }
    return result


@click.command()
@click.argument("gold_files", nargs=-1, type=str)
@click.option("--prediction-folder", type=str, default="data/predictions")
@click.option("--output", type=str, default="metrics/result.json")
@click.option("--metric", type=EvalMetricParam(), default='EM,F1')
def evaluate(prediction_folder, gold_files, output, metric):
    eval_metrics = metric
    result = dict()
    for gold_file in gold_files:
        click.echo(f"Evaluating predictions on {click.style(gold_file, fg='blue')}")
        gold = load_json(gold_file)

        gold_descriptor, prediction_files = match_prediction_to_gold(gold_file, prediction_folder)

        result[gold_file] = {'n': num_questions(gold)}
        logger.debug(prediction_files)
        for prediction_file in sorted(prediction_files):
            model_name = extract_model_name(gold_descriptor, prediction_file)
            click.echo(f"Evaluating predictions of model {click.style(model_name, fg='green')}")
            predictions = load_json(prediction_file)
            result[gold_file][model_name] = _get_score(gold, predictions, eval_metrics)

        write_json(result, output)
