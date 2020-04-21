import click
from loguru import logger

from stresstest.realize import Realizer
from stresstest.resources.templates import sentences
from tests.testutil import interactive_env


@click.group()
@click.option('--debug', default=False, is_flag=True)
def cli(debug):
    if not debug:
        logger.remove(0)


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


@click.command()
def generate():
    from tests.testutil import interactive_env
    interactive_env()


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
    for _ in range(k):
        r = Realizer(sentences=sentences)
        _, _, _, story_sents, questions, logical_form = interactive_env(realizer=r, do_print=False)
        for realised, logical in zip(story_sents, logical_form):
            if logical['action'] == action:
                styled = click.style(realised, fg='red', bold=True)
            else:
                styled = realised
            click.echo(styled)


cli.add_command(validate)
cli.add_command(test)
cli.add_command(generate)

if __name__ == '__main__':
    cli()
