import json
from collections import defaultdict

import click
import quickconf
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

from stresstest.classes import Model
from stresstest.util import load_json, sample_iter, num_questions


@click.command()
@click.option("--diversity", type=str, default='metrics/diversity.json')
@click.option("--count", type=str, default='metrics/count.json')
@click.option("--baseline", type=str, default='metrics/score-baseline.json')
@click.option("--intervention", type=str, default='metrics/score-intervention.json')
@click.option("--naturality", type=str, default='metrics/naturality.json')
def report(diversity, count, baseline, intervention, naturality):
    diversity_diff = load_json(diversity)
    count = load_json(count)
    phenomenon_score = load_json(intervention)
    baseline_score = load_json(baseline)
    try:
        naturality_diff = load_json(naturality)
    except:
        naturality_diff = None
    # TODO
    click.secho("Diversity: ", fg='green')
    table = []
    for metric, v in diversity_diff.items():
        if metric != 'num_samples':
            # click.echo(f'For "{metric}" metric:')
            # click.echo(f"Ours: {str(v['ours']['human_readable'])} "
            #           f"Reference: {str(v['reference']['human_readable'])}")
            # click.echo(f"Difference: {v['difference']['difference']}")
            table.append([
                click.style(metric, fg='green'),
                str(v['ours']['human_readable']),
                str(v['reference']['human_readable']),
                f"{v['difference']['difference']}"
            ])
            # click.echo(f"Within Confidence Interval? "
            #           f"{click.style('yes', fg='green') if v['difference']['within_ci']
            #           else click.style('no', fg='red')}")

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

    click.secho("Baseline performance: ", fg='green')
    for metric, v in baseline_score['full'].items():
        click.secho(f"{metric}: {v['human_readable']}")

    click.secho("Performance after Intervention: ", fg='green')
    for metric, v in phenomenon_score['full'].items():
        click.secho(v['human_readable'])
    click.secho("Differences", fg='green')
    for (metric, baseline_v) in baseline_score['full'].items():
        click.secho(f"{metric}: {baseline_v['mean'] - phenomenon_score['full'][metric]['mean']:.4f} ")
    # click.secho("Difference after intervention: ", fg='green')
    # click.echo(f"{phenomenon_diff['difference']:.2f}")
    # TODO well technically we should do a different computation here
    # click.echo(f"Statistically significant? f"{click.style('yes', fg='green')
    # if phenomenon_diff['within_ci'] else click.style('no', fg='red')}")
