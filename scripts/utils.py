import json
import click

from stresstest.generate import StoryGenerator
from stresstest.realize import Realizer
from stresstest.resources.templates import sentences


def write_json(result, output):
    with open(output, "w+") as f:
        json.dump(result, f, indent=4, separators=(',', ': '))


def get_templates(action: str = None, n: int = None, command: str = "Executing command"):
    actions = list(sentences.keys())
    if action is not None:
        actions = [action]

    if n is not None:
        for action in actions:
            sentences[action] = [sentences[action][n]]

    actions_str = click.style(', '.join(actions), fg='blue')
    n_str = click.style(text=str(n) if n else "all", fg='green', bold=True)
    click.echo(f"{click.style(command, fg='red')} for actions: '{actions_str}'; sentences: {n_str} !")
    return actions, sentences


def generate(c, generator_class=None):
    generator_class = generator_class or StoryGenerator
    generator = generator_class(c)
    realizer = Realizer()
    ss = generator.generate_story()
    story, visits = realizer.realise_story(ss, generator.world)
    (single_span_questions, multi_span_questions, unanswerable_questions, abstractive_questions) = \
        generator.generate_questions(ss, visits)
    realised_ssqs = [(q, realizer.realise_question(q)) for q in single_span_questions]
    realised_msqs = [(q, realizer.realise_question(q)) for q in multi_span_questions]
    realised_uaqs = [(q, realizer.realise_question(q)) for q in unanswerable_questions]
    realised_aqs = [(q, realizer.realise_question(q)) for q in abstractive_questions]
    qs = realised_ssqs
    ssqs = [(l, r) for l, r in zip(single_span_questions, realised_ssqs) if r]
    msqs = [(l, r) for l, r in zip(multi_span_questions, realised_msqs) if r]
    uaqs = [(l, r) for l, r in zip(unanswerable_questions, realised_uaqs) if r]
    aqs = [(l, r) for l, r in zip(abstractive_questions, realised_aqs) if r]
    return story, ssqs, msqs, uaqs, aqs
