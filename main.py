import glob
import importlib

import click
from click import Command
from loguru import logger
from tqdm import tqdm


@click.group()
@click.option('--debug', default=False, is_flag=True)
def cli(debug):
    if not debug:
        logger.remove(0)
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level='WARNING')
    else:
        logger.remove()
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
        logger.add("./logs/debug.log", level='DEBUG', rotation='50MB', compression="zip")
        logger.debug('Set up logging.')


# for folder in scripts: import everything that is a command, add to main cli
# python magic 🤪 (or more like undocumented interfaces)
for file in glob.iglob('scripts/*.py'):
    m = importlib.import_module(file[:-3].replace('/', '.'))
    for name, obj in m.__dict__.items():
        if isinstance(obj, Command):
            cli.add_command(obj)

if __name__ == '__main__':
    cli()
