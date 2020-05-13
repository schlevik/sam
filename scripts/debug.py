import json
from math import ceil

import click

from scripts.utils import get_templates
from stresstest.realize import Realizer
from stresstest.resources.templates import sentences
from tests.util import interactive_env


@click.command()
@click.option("--action", type=str, default=None)
@click.option("-n", type=int, default=None)
@click.option("-k", type=int, default=1)
@click.option("--with-questions", is_flag=True, default=False)
def test(action, n, k, with_questions):
    actions, sentences = get_templates(action, n, "Testing")
    if n is not None:
        click.secho(f"Testing: {action}.{n}:", fg='green')
        click.secho(f"{sentences[action][0]}", fg='green', bold=True)

    colorise = action is not None
    for action in actions:
        for i in range(ceil(k / len(actions))):
            try:
                r = Realizer(sentences=sentences, unique_sentences=False)
                generator, cfg, events, realizer, story, all_questions = interactive_env(realizer=r, do_print=False)
                for realised, logical in zip(story, events):
                    if colorise and logical['action'] == action:
                        styled = click.style(realised, fg='red', bold=True)
                    else:
                        styled = realised
                    click.echo(styled)
                if with_questions:
                    labels = ["SSQs", "MSQs", "UAQs", "AQs"]
                    for label, logical_qs in zip(labels, all_questions):
                        click.echo(f"{label}:")
                        for logical in logical_qs:
                            if logical.realized:
                                click.echo(logical)
                                click.echo(logical.realized)
                                # click.echo(realised)
                                click.echo()
                click.secho(20 * "=", bold=True)
            except Exception as e:
                click.secho(str(e), fg='red', bold=True)
                click.secho(f'Error in i={i}')
                raise e


@click.command()
def validate():
    print("Validating...")
    try:
        realizer = Realizer()
    except ValueError as e:
        print(str(e))
        click.secho("FAIL!", fg='red')
        return
    click.secho("SUCCESS!", fg='green')
