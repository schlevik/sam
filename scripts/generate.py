import json
import random
import uuid

import click
from quickconf import load_class
from tqdm import trange

from stresstest.classes import Config
from stresstest.generate import StoryGenerator
from stresstest.realize import Realizer


@click.command()
@click.option("--config", type=str, default='stresstest/resources/team-names.json')
@click.option("--output", type=str, default="data/stresstest.json")
@click.option("-n", type=int, default=5)
@click.option("-k", type=int, default=1)
@click.option("--seed", type=int, default=False)
@click.option("--multispan", is_flag=True, default=False)
@click.option("--unanswerable", is_flag=True, default=False)
@click.option("--abstractive", is_flag=True, default=False)
@click.option("--config", default='conf/baseline.json')
def generate(config, output, n, k, seed, multispan, unanswerable, abstractive):
    click.echo(
        f"Generating from '{click.style(config, fg='green')}': {click.style(str(k), fg='green', bold=True)} samples, "
        f"{click.style(str(n), fg='green', bold=True)} passages each.")
    click.echo(f"Saving in {click.style(output, fg='green', bold=True)}.")
    c = Config(config)
    # TODO: allow to select class from config
    uuid4 = lambda: uuid.UUID(int=random.getrandbits(128)).hex

    results = []

    if seed:
        random.seed(seed)

    for _ in trange(k, position=1):

        sample = []
        for _ in trange(n, position=0):
            # TODO: check if we can pull this out... not that it matters
            generator = load_class(c.get('generator.class'), StoryGenerator)(c)
            reasoning = c.get("reasoning", None)
            template_path = c.get("templates", "stresstest.resources.templates")
            templates = {
                name: load_class(f"{template_path}.{name}") for name in
                ['dollar', 'sentences', 'at', 'percent', 'bang', 'question_templates']
            }
            realizer = Realizer(**templates)
            events = generator.generate_story()
            story, visits = realizer.realise_story(events, generator.world)
            (single_span_questions, multi_span_questions, unanswerable_questions, abstractive_questions) = \
                generator.generate_questions(events, visits)
            realised_ssqs = [(q, realizer.realise_question(q, story)) for q in single_span_questions]
            realised_msqs = [(q, realizer.realise_question(q, story)) for q in multi_span_questions]
            realised_uaqs = [(q, realizer.realise_question(q, story)) for q in unanswerable_questions]
            realised_aqs = [(q, realizer.realise_question(q, story)) for q in abstractive_questions]
            qs = realised_ssqs
            if multispan:
                qs += realised_msqs
            if unanswerable:
                qs += realised_uaqs
            if abstractive:
                qs += realised_aqs
            qs = [(l, r) for l, r in qs if r and (reasoning and l.reasoning in reasoning or not reasoning)]
            sample.append({
                "id": uuid4(),
                "passage": ' '.join(story),
                "qas": [
                    {
                        "id": uuid4(),
                        "question": q,
                        "answer": a,
                        "reasoning": l.reasoning,
                        'type': l.type,
                        'target': l.target,
                        'evidence': l.evidence,
                        'event_type': l.event_type,
                        'question_data': l.question_data,
                    } for l, (q, a) in qs
                ]
            })
        results.append(sample)
    with open(output, "w+") as f:
        json.dump(results, f)
    click.secho("Done!", fg='green', bold=True)
