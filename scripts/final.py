import click
from tabulate import tabulate

from stresstest.util import load_json


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
        table.append([
            click.style(metric, fg='green'),
            str(v['ours']['human_readable']),
            str(v['reference']['human_readable']),
            f"{v['difference']['difference']}"
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

    click.secho("Baseline performance: ", fg='green')
    for metric, v in baseline_score['full'].items():
        click.secho(f"{metric}: {v['human_readable']}")

    click.secho("Performance after Intervention: ", fg='green')
    for metric, v in phenomenon_score['full'].items():
        click.secho(v['human_readable'])
    click.secho("Differences", fg='green')
    for (metric, baseline_v) in baseline_score['full'].items():
        click.secho(f"{metric}: {baseline_v['mean'] - phenomenon_score['full'][metric]['mean']:.4f} ")
