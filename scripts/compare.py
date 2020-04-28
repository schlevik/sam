import random

import click

from stresstest.classes import Config
from stresstest.generate import StoryGenerator
from stresstest.realize import Realizer
from tests.testutil import interactive_env


@click.command()
@click.option("--input", type=str, default="data/stresstest.json")
@click.option("--reference")
def compare(input, reference):
    raise NotImplementedError()
