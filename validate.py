import click
from loguru import logger

from stresstest.realize import Realizer


@click.command()
def validate():
    logger.remove(0)
    print("Validating...")
    try:
        realizer = Realizer()
    except ValueError as e:
        print(str(e))
        click.secho("FAIL!", fg='red')
    click.secho("SUCCESS!", fg='green')


if __name__ == '__main__':
    validate()
