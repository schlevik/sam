import json
import random
import uuid

import click

from stresstest.classes import Config
from stresstest.generate import StoryGenerator
from stresstest.realize import Realizer


@click.command()
@click.option("--config", type=str, default='stresstest/resources/config.json')
@click.option("--output", type=str, default="data/stresstest.json")
@click.option("-n", type=int, default=5)
@click.option("-k", type=int, default=1)
@click.option("--seed", type=int, default=False)
@click.option("--multispan", is_flag=True, default=False)
@click.option("--unanswerable", is_flag=True, default=False)
@click.option("--abstractive", is_flag=True, default=False)
def generate(config, output, n, k, seed, multispan, unanswerable, abstractive):
    """

    Args:
        config: Config for the generator.
        output: Where to save the output.
        n: Number of passages per sample.
        k: Number of samples.

    """
    click.echo(
        f"Generating from '{click.style(config, fg='green')}': {click.style(str(k), fg='green', bold=True)} samples, "
        f"{click.style(str(n), fg='green', bold=True)} passages each.")
    click.echo(f"Saving in {click.style(output, fg='green', bold=True)}.")
    c = Config(config)
    # TODO: allow to select class from config
    # TODO: # something something fix random seed.
    uuid4 = lambda: uuid.UUID(int=random.getrandbits(128)).hex

    results = []

    if seed:
        random.seed(seed)

    for _ in range(k):

        sample = []
        for _ in range(n):
            # TODO: check if we can pull this out... not that it matters
            generator = StoryGenerator(c)
            realizer = Realizer()
            ss = generator.generate_story()
            story, visits = realizer.realise_story(ss, generator.world)
            (single_span_questions, multi_span_questions, unanswerable_questions, abstractive_questions) = \
                generator.generate_questions(ss, visits)
            realised_ssqs = [realizer.realise_question(q) for q in single_span_questions]
            realised_msqs = [realizer.realise_question(q) for q in multi_span_questions]
            realised_uaqs = [realizer.realise_question(q) for q in unanswerable_questions]
            realised_aqs = [realizer.realise_question(q) for q in abstractive_questions]
            # qs = realised_ssqs + realised_msqs + realised_uaqs + realised_aqs
            qs = realised_ssqs
            if multispan:
                qs += realised_msqs
            if unanswerable:
                qs += realised_uaqs
            if abstractive:
                qs += realised_aqs
            qs = [q for q in qs if q]
            sample.append(
                {"id": uuid4(),
                 "passage": ' '.join(story),
                 "qas": [
                     {"id": uuid4(), "question": q, "answer": a} for q, a in qs
                 ]}
            )
        results.append(sample)
    with open(output, "w+") as f:
        json.dump(results, f)
    click.secho("Done!", fg='green', bold=True)
