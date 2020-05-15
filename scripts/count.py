import click

from scripts.utils import write_json, get_templates
from stresstest.realize import SizeEstimator, Processor, Accessor


@click.command()
@click.option("--action", type=str, default=None)
@click.option("-n", type=int, default=None)
@click.option('--output', type=str, default=None)
def count(action, n, output):
    result = {}
    actions, sentences = get_templates(action, n, "Counting")
    for action in actions:
        r = SizeEstimator(processor=Processor(accessor=Accessor(sentences=sentences)))
        upper_bound = r.estimate_size(r.processor.accessor.sentences[action])
        r = SizeEstimator(processor=Processor(accessor=Accessor(sentences=sentences)))
        lower_bound = r.estimate_size(r.processor.accessor.sentences[action], pessimistic=True)
        click.secho(
            f"Pessimistically speaking, you can generate {click.style(str(lower_bound), fg='red', bold=True)} "
            f"distinct sentences!")

        click.secho(
            f"Optimistically speaking, you can generate {click.style(str(upper_bound), fg='green', bold=True)} "
            f"distinct sentences!")
        result[action] = {
            "lower": lower_bound,
            "upper": upper_bound
        }
    if output:
        write_json(result, output)
