import click

from stresstest.realize import Realizer
from stresstest.resources.templates import sentences


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
    click.secho(
        f"Optimistically speaking, you can generate {click.style(str(size), fg='green', bold=True)} distinct sentences!")
    r = Realizer(sentences=sentences)
    size = r.estimate_size(r.sentences[action], pessimistic=True)
    click.secho(
        f"Pessimistically speaking, you can generate {click.style(str(size), fg='red', bold=True)} distinct sentences!")
