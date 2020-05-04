import random

import click

from scripts.utils import write_json
from stresstest.eval_utils import eval_em, get_mean_var_ci
from stresstest.util import load_json, sample_iter, num_questions


@click.command()
@click.option("--predictions", type=str, default='data/predictions.json')
@click.option("--gold", type=str, default='data/stresstest.json')
@click.option("--output", type=str, default="metrics/score.json")
@click.option("--metric", type=str, default='em')
def evaluate(predictions, gold, output, metric):
    gold = load_json(gold)
    predictions = load_json(predictions)
    sample_metrics = []
    click.echo(f"Evaluating {click.style(str(len(gold)), fg='green', bold=True)} sample(s).")
    for i, (gold_sample, predictions_sample) in enumerate(zip(gold, predictions)):
        sample_metrics.append(eval_em(sample_iter(gold_sample), predictions_sample))
    click.echo(f"Results: {' '.join(f'{s:.2f}' for s in sample_metrics)}")
    mean, var, ci = get_mean_var_ci(sample_metrics)
    printable_result = f'{mean:.4f} +/- {ci:.4f}'
    click.echo(f"Average  under the {click.style(metric, fg='green')} metric: "
               f"{click.style(printable_result, fg='green', bold=True)}")
    click.echo()
    result = {}
    result['full'] = {
        "num_samples": len(predictions),
        metric: {
            'human_readable': printable_result,
            'mean': mean,
            'variance': var,
            '95ci': ci
        }
    }

    sample_sizes = [10, 25, 50, 100, 250]
    for k in sample_sizes:
        sub_sample_metrics = []

        click.echo(f"Evaluating {click.style(str(len(gold)), fg='green', bold=True)} sample(s), "
                   f"sub-sampling {click.style(str(k), fg='green', bold=True)} entries.")
        too_small = any(num_questions(s) < k for s in gold)
        if too_small:
            click.echo(f"Omitting drawing {k} samples (and subsequent), not enough data to sample from!")
            break
        for i, (gold_sample, predictions_sample) in enumerate(zip(gold, predictions)):
            new_gold_sample = random.sample(list(sample_iter(gold_sample)), k)
            sub_sample_metrics.append(eval_em(new_gold_sample, predictions_sample))
        click.echo(f"Results: {' '.join(f'{s:.2f}' for s in sub_sample_metrics)}")
        mean, var, ci = get_mean_var_ci(sub_sample_metrics)
        printable_result = f'{mean:.4f} +/- {ci:.4f}'
        click.echo(f"Average  under the {click.style(metric, fg='green')} metric: "
                   f"{click.style(printable_result, fg='green', bold=True)}")
        click.echo()
        result[k] = {
            "num_samples": len(predictions),
            metric: {
                'human_readable': printable_result,
                'mean': mean,
                'variance': var,
                '95ci': ci
            }
        }

    write_json(result, output)
