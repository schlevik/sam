from math import ceil

import click
from quickconf import load_class

from scripts.utils import get_templates
from stresstest.classes import Config
from stresstest.generator import StoryGenerator
from stresstest.realize import Realizer


@click.command()
@click.option("--action", type=str, default=None)
@click.option("-n", type=int, default=None)
@click.option("-k", type=int, default=1)
@click.option("--with-questions", is_flag=True, default=False)
@click.option("--config", default='conf/baseline.json')
@click.option("--multispan", is_flag=True, default=False)
@click.option("--unanswerable", is_flag=True, default=False)
@click.option("--abstractive", is_flag=True, default=False)
def test(action, n, k, with_questions, config, multispan, unanswerable, abstractive):
    actions, sentences = get_templates(action, n, "Testing")
    if n is not None:
        click.secho(f"Testing: {action}.{n}:", fg='green')
        click.secho(f"{sentences[action][0]}", fg='green', bold=True)

    colorise = action is not None
    c = Config(config)
    for action in actions:
        for i in range(ceil(k / len(actions))):
            try:
                # TODO: that's a bit bodged the way it is now
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
                all_questions = (single_span_questions, multi_span_questions,
                                 unanswerable_questions, abstractive_questions)
                for q in (q for qs in all_questions for q in qs):
                    realizer.realise_question(q, story)
                for realised, logical in zip(story, events):
                    if colorise and logical['action'] == action:
                        styled = click.style(realised, fg='red', bold=True)
                    else:
                        styled = realised
                    click.echo(styled)
                if with_questions:
                    labels = ["SSQs"]
                    if multispan:
                        labels += ["MSQs"]
                    if unanswerable:
                        labels += ["UAQs"]
                    if abstractive_questions:
                        labels += ['AQs']
                    for label, logical_qs in zip(labels, all_questions):
                        click.echo(f"{label}:")
                        for logical in logical_qs:
                            if logical.realized and (reasoning and logical.reasoning in reasoning or not reasoning):
                                click.echo(logical)
                                click.echo(f"{logical.realized} {logical.answer}")
                                click.echo()
                click.secho(20 * "=", bold=True)
            except Exception as e:
                click.secho(str(e), fg='red', bold=True)
                click.secho(f'Error in i={i}')
                raise e


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
