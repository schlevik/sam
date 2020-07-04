import os

import click
from tabulate import tabulate

from scripts.utils import BASELINE, INTERVENTION
from stresstest.util import load_json


@click.command()
@click.option("--diversity", type=str, default='metrics/diversity.json')
@click.option("--count", type=str, default='metrics/count.json')
@click.option("--evaluation", type=str, default='metrics/result.json')
@click.option('--evaluation-intervention', type=str, default='metrics/result-intervention.json')
@click.option("--naturality", type=str, default='metrics/naturality.json')
def report(diversity, count, evaluation, evaluation_intervention, naturality):
    diversity_diff = load_json(diversity)
    count = load_json(count)

    evaluation = load_json(evaluation)
    evaluation_intervention = load_json(evaluation_intervention)
    try:
        naturality_diff = load_json(naturality)
    except:
        naturality_diff = None
    # TODO
    click.secho("Diversity: ", fg='green')
    table = []
    for metric, v in diversity_diff.items():
        table.append([
            click.style(metric, fg='green'),
            str(v['ours']['human_readable']),
            str(v['reference']['human_readable']),
            f"{v['difference']}"
        ])

    click.echo(tabulate(table, headers=[click.style('Index', bold=True), click.style('Stress-test', bold=True),
                                        click.style('Reference', bold=True), click.style('Difference', bold=True)]))
    click.secho("Naturality: ", fg='green')
    table = []
    if naturality_diff:
        for metric, v in naturality_diff.items():
            if metric != 'num_samples':
                table.append([
                    click.style(metric, fg='green'),
                    str(v['ours']['human_readable']),
                    str(v['reference']['human_readable']),
                    f"{v['difference']['difference']}"
                ])
    click.echo(tabulate(table, headers=[click.style('Metric', bold=True), click.style('Stress-test', bold=True),
                                        click.style('Reference', bold=True), click.style('Difference', bold=True)]))

    click.secho("Configuration estimate: ", fg='green')
    for k, v in count.items():
        click.echo(f'For "{k}" event type:')
        click.echo(f"Upper bound: {v['upper']} Lower bound: {v['lower']}")

    baseline_score = next(v for k, v in evaluation.items() if BASELINE in os.path.basename(k))
    intervention_score = next(v for k, v in evaluation.items() if INTERVENTION in os.path.basename(k))

    click.secho("Baseline/Intervention performance isolated: ", fg='green')
    table = []
    headers = []
    for model_name, values in baseline_score.items():
        intervention_values = intervention_score[model_name]
        if model_name != 'n':

            if not headers:
                headers = [click.style(k, bold=True) for k in
                           ['Model'] + list(values.keys()) + [f"Intervention {v}" for v in values.keys()]]
            table.append([model_name] +
                         [values[val]['human_readable'] for val in values] +
                         [intervention_values[val]['human_readable'] for val in intervention_values]
                         )
    click.echo(tabulate(table, headers=headers))
    click.secho("Evaluation by intervention: ", fg='green')
    for model_name, values in evaluation_intervention.items():
        click.echo(f"Model: {model_name}: {values['evaluation_on_intervention']['human_readable']}")

        headers = [click.style(k, bold=True) for k in ['Model Behaviour', f'Occurrences (of {values["n"]})']]
        click.echo(tabulate(list(values['behaviour'].items()), headers=headers))
