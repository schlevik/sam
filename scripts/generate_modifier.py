import json
import os
import random
import uuid
from typing import List, Dict

import click

from scripts.utils import Domain
from stresstest.classes import Config
from stresstest.realize import Realizer


def _generate(generator_class, question_types, answer_types, templates, first_modification,
              fill_with_modification,
              modify_event_types, modification_distance, total_modifiable_actions, uuid4):
    # generate modified
    story_id = uuid4()
    modified = {"id": story_id, 'qas': []}
    realizer = Realizer(**templates)
    generator = generator_class({}, first_modification=first_modification,
                                fill_with_modification=fill_with_modification, modify_event_types=modify_event_types,
                                modification_distance=modification_distance,
                                total_modifiable_actions=total_modifiable_actions)

    events = generator.generate_story()
    # raise NotImplementedError()
    story, visits = realizer.realise_story(events, generator.world)
    modified["passage"] = ' '.join(story)
    modified['passage_sents'] = story
    choices = realizer.context.choices
    template_choices = realizer.context.chosen_templates
    # print("\n".join(story))
    # print()
    (single_span_questions, multi_span_questions, unanswerable_questions, abstractive_questions) = \
        generator.generate_questions(events, visits)
    for q in single_span_questions + multi_span_questions + unanswerable_questions + abstractive_questions:
        realizer.realise_question(q, story)

    for label, logical_qs in zip(answer_types, (single_span_questions, multi_span_questions,
                                                unanswerable_questions, abstractive_questions)):
        # click.echo(f"{label.upper()}s:")
        for logical in logical_qs:
            if logical.realized and (question_types and logical.reasoning in question_types or not question_types) \
                    and logical.event_type in modify_event_types:
                question_data_str = "/".join(
                    f"{k}:{v}" for k, v in logical.question_data.items() if k not in ['modified', 'easier'])
                modified['qas'].append({
                    "id": f"{story_id}/{logical.reasoning}/{logical.type}/{logical.target}/{logical.event_type}/"
                          f"{question_data_str}",
                    "question": logical.realized,
                    "answer": logical.answer,
                    "reasoning": logical.reasoning,
                    'type': logical.type,
                    'target': logical.target,
                    'evidence': logical.evidence,
                    'event_type': logical.event_type,
                    'question_data': logical.question_data,
                })
                # click.echo(logical)
                # click.echo(f"{logical.event_type}|{logical.target}|{logical.question_data.get('n', None)}")
                # click.echo(f"{logical.realized} {logical.answer}")
                # click.echo()

    question_map = {}
    for question in single_span_questions + multi_span_questions + unanswerable_questions + abstractive_questions:
        question_map[(question.event_type, question.target, question.question_data.get("n", None))] = question

    # remove modifier
    for event in events:
        event.features = []

    # generate baseline
    baseline = {
        "id": story_id,
        "qas": []
    }

    realizer = Realizer(**templates)
    story, visits = realizer.realise_with_choices(events, generator.world, choices, template_choices)
    # print(20 * "===")
    # print("\n".join(story))
    # generator = load_class(config.get('generator.class'), StoryGenerator)(config)
    baseline["passage"] = ' '.join(story)
    baseline['passage_sents'] = story

    generator = generator_class({}, first_modification=first_modification,
                                fill_with_modification=fill_with_modification, modify_event_types=modify_event_types,
                                modification_distance=modification_distance,
                                total_modifiable_actions=total_modifiable_actions)
    (single_span_questions, multi_span_questions, unanswerable_questions, abstractive_questions) = \
        generator.generate_questions(events, visits)
    for q in single_span_questions + multi_span_questions + unanswerable_questions + abstractive_questions:
        try:
            q.realized = question_map[(q.event_type, q.target, q.question_data.get("n", None))].realized
            q.answer = realizer._fix_units(q, story)
        except:
            realizer.realise_question(q, story)

    for label, logical_qs in zip(answer_types, (single_span_questions, multi_span_questions,
                                                unanswerable_questions, abstractive_questions)):
        # click.echo(f"{label.upper()}s:")
        for logical in logical_qs:
            if logical.realized and (question_types and logical.reasoning in question_types or not question_types) \
                    and logical.event_type == 'goal':
                question_data_str = "/".join(
                    f"{k}:{v}" for k, v in logical.question_data.items() if k not in ['modified', 'easier'])
                baseline['qas'].append({
                    "id": f"{story_id}/{logical.reasoning}/{logical.type}/{logical.target}/{logical.event_type}/"
                          f"{question_data_str}",
                    "question": logical.realized,
                    "answer": logical.answer,
                    "reasoning": logical.reasoning,
                    'type': logical.type,
                    'target': logical.target,
                    'evidence': logical.evidence,
                    'event_type': logical.event_type,
                    'question_data': logical.question_data,
                })
                # click.echo(logical)
                # click.echo(f"{logical.event_type}|{logical.target}|{logical.question_data.get('n', None)}")
                # click.echo(f"{logical.realized} {logical.answer}")
                # click.echo()
    return baseline, modified


@click.command()
@click.option("--config", default='conf/modifier.json')
@click.option("--out-path", default='data/modifier/')
@click.option("--seed", type=int, default=False)
@click.option("-n", type=int, default=5)
@click.option("-k", type=int, default=1)
@click.option("--do-print", is_flag=True, default=False)
@click.option("--do-save", is_flag=True, default=False)
@click.option("--domain", type=Domain(), default='football')
def generate_modifier(config, out_path, seed, n, k, do_print, do_save, domain):
    if seed:
        random.seed(seed)

    uuid4 = lambda: uuid.UUID(int=random.getrandbits(128)).hex
    cfg = Config(config)
    answer_types = cfg.get('answer_types', ['ssq'])
    question_types = cfg.get('reasoning', ['retrieval'])
    # k = 5
    # n = 50

    file_name = f"stresstest-{'-'.join(answer_types)}-{{}}-{'-'.join(question_types)}-{n}-{k}.json"
    click.echo(
        f"Generating from '{click.style(config, fg='green')}': {click.style(str(k), fg='green', bold=True)} samples, "
        f"{click.style(str(n), fg='green', bold=True)} passages each.")
    click.echo(
        f"Saving baseline in {click.style(os.path.join(out_path, file_name.format('baseline')), fg='green', bold=True)}.")
    click.echo(
        f"Saving modified in {click.style(os.path.join(out_path, file_name.format('modifier')), fg='green', bold=True)}.")

    samples_baseline = []
    samples_modified = []

    max_sents = cfg["world.num_sentences"]

    templates = domain.templates_modifier

    first_modifications = range(max_sents - 1)
    fill_with_modifications = [True, False]
    modify_event_types = ['goal']

    # num of modifications: f(max_sent) = |modify_event_types| * 1/3 * max_sent * (max_sent - 1) * (max_sent - 2)
    for _ in range(k):
        baseline = []
        modified = []
        for modify_event_type in modify_event_types:
            for first_modification in first_modifications:
                for fill_with_modification in fill_with_modifications:
                    modification_distances = range(1, max_sents - first_modification)
                    for modification_distance in modification_distances:
                        if fill_with_modification:
                            totals = range(modification_distance + 1, max_sents - first_modification)
                        else:
                            totals = range(2, max_sents - (modification_distance + first_modification) + 1)
                        for total_modifiable_actions in totals:
                            b, m = _generate(domain.generator_modifier, question_types, answer_types, templates,
                                             first_modification, fill_with_modification, [modify_event_type],
                                             modification_distance, total_modifiable_actions, uuid4)
                            baseline.append(b)
                            modified.append(m)
                            for qa in m['qas']:
                                modification_data = {
                                    'first_modification': first_modification,
                                    'fill_with_modification': fill_with_modification,
                                    'modify_event_type': modify_event_type,
                                    'modification_distance': modification_distance,
                                    'total_modifiable_actions': total_modifiable_actions
                                }
                                qa['modification_data'] = modification_data
        total_q_b = sum(1 for b in baseline for _ in b['qas'])
        total_q_m = sum(1 for m in modified for _ in m['qas'])
        if do_print:
            for b, m in zip(baseline, modified):
                click.echo("Baseline Story:")
                for i, sent in enumerate(b['passage_sents']):
                    click.echo(f'[{i}]: {sent}')

                click.echo("Baseline Questions:")
                for qa in b['qas']:
                    click.echo(f"{qa['question']}: {qa['answer']} ({qa['evidence']})")
                click.echo("Modified Story:")
                for i, sent in zip(range(max_sents), m['passage_sents']):
                    click.echo(f'[{i}]: {sent}')
                click.echo("Modified Questions:")
                for qa in m['qas']:
                    click.echo(f"{qa['question']}: {qa['answer']} ({qa['evidence']})")
                if m['qas']:
                    click.echo(m['qas'][0]['modification_data'])

                click.echo(20 * "===")

        click.echo(f"Total Passages: {len(baseline)}")
        click.echo(f"Total Questions over baseline passages: {total_q_b}")
        click.echo(f"Total Questions over modified passages: {total_q_m}")
        samples_baseline.extend(baseline)
        samples_modified.extend(modified)

    if do_save:
        with open(os.path.join(out_path, file_name.format('baseline')), "w+") as f:
            json.dump(samples_baseline, f)

        with open(os.path.join(out_path, file_name.format('modifier')), "w+") as f:
            json.dump(samples_modified, f)
