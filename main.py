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
@click.option("-k", type=int, default=1)
def generate(k):
    for _ in range(k):
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
@click.argument("action", type=str)
@click.option("-n", type=int, default=-1)
def count(action, n):
    if n >= 0:
        click.echo(f"Counting only for '{action}' sentence {click.style(text=n, fg='green', bold=True)}!")
        sentences[action] = [sentences[action][n]]
    else:
        click.echo(f"Counting only for {click.style(text='all', fg='green', bold=True)} '{action}' sentences!")

    r = Realizer(sentences=sentences)
    size = r.estimate_size(r.sentences[action])
    click.secho(f"Optimistically speaking, you can generate {size} distinct sentences!")


cli.add_command(validate)
cli.add_command(test)
cli.add_command(generate)
cli.add_command(count)

if __name__ == '__main__':
    cli()
