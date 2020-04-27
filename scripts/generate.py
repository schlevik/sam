import click

from tests.testutil import interactive_env


@click.command()
@click.option("-k", type=int, default=1)
def generate(k):
    for _ in range(k):
        interactive_env()
