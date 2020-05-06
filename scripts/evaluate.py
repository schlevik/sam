import random

import click

from scripts.utils import write_json
from stresstest.eval_utils import em, get_mean_var_ci, f1
from stresstest.util import load_json, sample_iter, num_questions


def _get_score(gold, predictions, metrics, subsample=None):
    result = dict()
    for metric in metrics:
        metric_name = metric.__name__
        metric_results = []
        for i, (gold_sample, predictions_sample) in enumerate(zip(gold, predictions)):
            gold_sample = \
                random.sample(list(sample_iter(gold_sample)), subsample) if subsample else sample_iter(gold_sample)
            metric_results.append(metric(gold_sample, predictions_sample))
        mean, var, ci = get_mean_var_ci(metric_results)
        printable_result = f'{mean:.4f} +/- {ci:.4f}'

        click.echo(f"Average  under the {click.style(metric_name, fg='green')} metric on "
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
@click.option("--metric", type=str, default='em,f1')
def evaluate(predictions, gold, output, metric):
    eval_metrics = []
    if 'em' in metric:
        eval_metrics.append(em)
    if 'f1' in metric:
        eval_metrics.append(f1)
    if not eval_metrics:
        raise ValueError("Need at list one metric to evaluate! Choose from: {em, f1}")
    gold = load_json(gold)
    predictions = load_json(predictions)
    click.echo(f"Evaluating {click.style(str(len(gold)), fg='green', bold=True)} sample(s).")
    result = {'num_samples': len(predictions), 'full': _get_score(gold, predictions, eval_metrics)}
    # click.echo(f"Results EM: {' '.join(f'{s:.2f}' for s in em_metrics)}")

    sample_sizes = [10, 25, 50, 100, 250]
    for k in sample_sizes:
        too_small = any(num_questions(s) < k for s in gold)
        if too_small:
            click.echo(f"Omitting drawing {k} samples (and subsequent), not enough data to sample from!")
            break
        result[str(k)] = _get_score(gold, predictions, eval_metrics, subsample=k)

    write_json(result, output)
