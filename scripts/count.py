import click

from scripts.utils import write_json, get_templates, Domain
from stresstest.classes import Bundle
from stresstest.realize import SizeEstimator, Processor, Accessor, RandomChooser


@click.command()
@click.option("--action", type=str, default=None)
@click.option("-n", type=int, default=None)
@click.option('--output', type=str, default=None)
@click.option('--domain', type=Domain(), default='football')
@click.option('--mod', type=str, default=None)
def count(action, n, output, domain: Bundle, mod):
    result = {}
    templates = domain[f"templates_{mod}" if mod else "templates"]
    actions, sentences = get_templates(templates=templates, action=action, n=n, command="Counting")
    for action in actions:
        click.echo(f"For action '{click.style(action, fg='blue')}':")
        r = SizeEstimator(processor=Processor(accessor=Accessor(**templates), chooser=RandomChooser()))
        upper_bound = r.estimate_size(r.processor.accessor.sentences[action])
        r = SizeEstimator(processor=Processor(accessor=Accessor(**templates), chooser=RandomChooser()))
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
