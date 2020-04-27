import click

from stresstest.realize import Realizer
from stresstest.resources.templates import sentences
from tests.testutil import interactive_env


@click.command()
@click.argument("action", type=str)
@click.option("-n", type=int, default=-1)
@click.option("-k", type=int, default=1)
def test(action, n, k):
    if n < 0:
        n = len(sentences[action]) - 1

    click.secho(f"Testing: {action}.{n}:", fg='green')
    click.secho(f"{sentences[action][n]}", fg='red', bold=True)

    sentences[action] = [sentences[action][n]]
    for i in range(k):
        try:
            r = Realizer(sentences=sentences, unique_sentences=False)
            _, _, _, story_sents, questions, logical_form = interactive_env(realizer=r, do_print=False)
            for realised, logical in zip(story_sents, logical_form):
                if logical['action'] == action:
                    styled = click.style(realised, fg='red', bold=True)
                else:
                    styled = realised
                click.echo(styled)
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
