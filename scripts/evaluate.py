import random
from typing import List, Type

import click
import numpy as np

from scripts.utils import write_json, EvalMetricParam
from stresstest.eval_utils import get_mean_var_ci, get_mean_var_ci_bernoulli, EvalMetric

from stresstest.util import load_json, sample_iter


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
@click.option("--predictions", type=str, default='data/predictions.json')
@click.option("--gold", type=str, default='data/stresstest.json')
@click.option("--output", type=str, default="metrics/score.json")
@click.option("--metric", type=EvalMetricParam(), default='EM,F1')
def evaluate(predictions, gold, output, metric):
    eval_metrics = metric
    gold = load_json(gold)
    predictions = load_json(predictions)
    click.echo(f"Evaluating {click.style(str(len(gold)), fg='green', bold=True)} sample(s).")
    result = {'num_samples': len(predictions), 'full': _get_score(gold, predictions, eval_metrics)}

    sample_sizes = [10, 25, 50, 100, 250, 500]
    for k in sample_sizes:
        too_small = len(gold) < k
        if too_small:
            click.echo(f"Omitting drawing {k} samples (and subsequent), not enough data to sample from!")
            break
        result[str(k)] = _get_score(gold, predictions, eval_metrics, subsample=k)

    write_json(result, output)
