import glob
import importlib

import click
from click import Command
from loguru import logger


@click.group()
@click.option('--debug', default=False, is_flag=True)
def cli(debug):
    if not debug:
        logger.remove(0)


# for folder in scripts: import everything, add to cli
for file in glob.iglob('scripts/*.py'):
    m = importlib.import_module(file[:-3].replace('/', '.'))
    for name, obj in m.__dict__.items():
        if isinstance(obj, Command):
            cli.add_command(obj)

if __name__ == '__main__':
    cli()
