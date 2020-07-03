import json
import os
import random
import uuid
from copy import deepcopy

import click
from joblib import delayed, Parallel
from loguru import logger
from tqdm import tqdm

from scripts.utils import Domain
from stresstest.classes import Config
from stresstest.comb_utils import generate_all_possible_template_choices
from stresstest.realize import Realizer


def _generate_events(generator_class, config, first_modification,
                     fill_with_modification,
                     modify_event_types, modification_distance, total_modifiable_actions):
    generator = generator_class(config, first_modification=first_modification,
                                fill_with_modification=fill_with_modification, modify_event_types=modify_event_types,
                                modification_distance=modification_distance,
                                total_modifiable_actions=total_modifiable_actions)
    events = generator.generate_story()
    world = generator.world
    return events, world


def _realize_events(generator_class, target_event_types, events, world, arranged_sentences, question_types,
                    answer_types,
                    templates, uuid4, modification_data=None):
    logger.remove()
    # realizer = Realizer(**templates)
    story_id = uuid4()
    modified = {"id": story_id, 'qas': []}
    realizer = Realizer(**templates, unique_sentences=True)

    story, visits = realizer.realise_with_sentence_choices(events, world, arranged_sentences)
    modified["passage"] = ' '.join(story)
    modified['passage_sents'] = story
    choices = realizer.context.choices
    template_choices = realizer.context.chosen_templates

    generator = generator_class({})
    (single_span_questions, multi_span_questions, unanswerable_questions, abstractive_questions) = \
        generator.generate_questions(events, visits)
    for q in single_span_questions + multi_span_questions + unanswerable_questions + abstractive_questions:
        realizer.realise_question(q, story)

    for label, logical_qs in zip(answer_types, (single_span_questions, multi_span_questions,
                                                unanswerable_questions, abstractive_questions)):
        for logical in logical_qs:
            if logical.realized and (question_types and logical.reasoning in question_types or not question_types) \
                    and logical.event_type in target_event_types:
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
                    'modification_data': modification_data
                })
                # click.echo(logical)
                # click.echo(f"{logical.event_type}|{logical.target}|{logical.question_data.get('n', None)}")
                # click.echo(f"{logical.realized} {logical.answer}")
                # click.echo()

    question_map = {}
    for question in single_span_questions + multi_span_questions + unanswerable_questions + abstractive_questions:
        question_map[(question.event_type, question.target, question.question_data.get("n", None))] = question

    # remove modifier
    events = deepcopy(events)
    for event in events:
        event.features = []

    # generate baseline
    baseline = {
        "id": story_id,
        "qas": []
    }

    # realizer = Realizer(**templates)
    realizer = Realizer(**templates, unique_sentences=True)
    story, visits = realizer.realise_with_choices(events, world, choices, template_choices)
    # print(20 * "===")
    # print("\n".join(story))
    # generator = load_class(config.get('generator.class'), StoryGenerator)(config)
    baseline["passage"] = ' '.join(story)
    baseline['passage_sents'] = story

    generator = generator_class({})
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
                    and logical.event_type in target_event_types:
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
                    'modification_data': modification_data
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
@click.option("--subsample", type=int, default=0)
@click.option("--do-print", is_flag=True, default=False)
@click.option("--do-save", is_flag=True, default=False)
@click.option("--domain", type=Domain(), default='football')
@click.option("--num-workers", type=int, default=8)
def generate_modifier(config, out_path, seed, subsample, do_print, do_save, domain, num_workers):
    if seed:
        random.seed(seed)
    cfg = Config(config)
    max_sents = cfg["world.num_sentences"]

    first_modifications = range(max_sents - 1)
    modify_event_types = ['goal']
    fill_with_modifications = [True, False]
    n = int(len(modify_event_types) * 1 / 3 * max_sents * (max_sents - 1) * (max_sents - 2))

    templates = domain.templates_modifier

    uuid4 = lambda: uuid.UUID(int=random.getrandbits(128)).hex

    answer_types = cfg.get('answer_types', ['ssq'])
    question_types = cfg.get('reasoning', ['retrieval'])

    subsample_str = subsample if subsample else 'full'
    file_name = f"stresstest-{'-'.join(answer_types)}-{{}}-{'-'.join(question_types)}-{n}-{subsample_str}.json"
    click.echo(
        f"Generating from '{click.style(config, fg='green')}': {click.style(str(n), fg='green', bold=True)} passages, "
        f"{click.style(str(subsample_str), fg='green', bold=True)} realisation per passage.")
    click.echo(f"Saving baseline in "
               f"{click.style(os.path.join(out_path, file_name.format('baseline')), fg='green', bold=True)}.")
    click.echo(f"Saving modified in "
               f"{click.style(os.path.join(out_path, file_name.format('modifier')), fg='green', bold=True)}.")

    samples_baseline = []
    samples_modified = []

    all_template_choices = []
    # num of modifications: f(max_sent) = |modify_event_types| * 1/3 * max_sent * (max_sent - 1) * (max_sent - 2)
    with tqdm(total=n) as progress_bar:
        # for _ in range(k):
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
                            # bs, ms = _generate(domain.generator_modifier, cfg, question_types, answer_types,
                            #                    templates, first_modification, fill_with_modification,
                            #                    [modify_event_type], modification_distance,
                            #                    total_modifiable_actions, uuid4)
                            modification_data = {
                                'first_modification': first_modification,
                                'fill_with_modification': fill_with_modification,
                                'modify_event_type': modify_event_type,
                                'modification_distance': modification_distance,
                                'total_modifiable_actions': total_modifiable_actions
                            }

                            events, world = _generate_events(
                                domain.generator_modifier, cfg,
                                first_modification=first_modification,
                                fill_with_modification=fill_with_modification,
                                modify_event_types=[modify_event_type],
                                modification_distance=modification_distance,
                                total_modifiable_actions=total_modifiable_actions
                            )
                            template_choices = generate_all_possible_template_choices(
                                events,
                                templates['sentences'],
                                [modify_event_type]
                            )
                            if subsample:
                                template_choices = random.sample(template_choices, subsample)

                            all_template_choices.extend(
                                (events, world, [modify_event_type], c, modification_data)
                                for c in template_choices
                            )

                            # for choice in tqdm(template_choices, position=1):
                            #     # TODO: do some parallelism here
                            #     b, m = _realize_events(domain.generator_modifier, [modify_event_type],
                            #                            events=events, world=world, arranged_sentences=choice,
                            #                            question_types=question_types, answer_types=answer_types,
                            #                            templates=templates, uuid4=uuid4)
                            #     # bs, ms =

                            progress_bar.update()

    all_realised = Parallel(n_jobs=num_workers)(
        delayed(_realize_events)(
            domain.generator_modifier, event_type_targets,
            events=events, world=world, arranged_sentences=choice,
            question_types=question_types, answer_types=answer_types,
            templates=templates, uuid4=uuid4, modification_data=modification_data
        ) for (events, world, event_type_targets, choice, modification_data) in tqdm(all_template_choices)
    )

    baseline, modified = zip(*all_realised)
    assert len(baseline) == len(set(b['passage'] for b in baseline))
    assert len(modified) == len(set(b['passage'] for b in modified))
    total_q_b = sum(1 for b in baseline for _ in b['qas'])
    total_q_m = sum(1 for m in modified for _ in m['qas'])
    # click.echo(f"Can generate different realised forms for this set.")
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


def _generate(generator_class, config, question_types, answer_types, templates, first_modification,
              fill_with_modification,
              modify_event_types, modification_distance, total_modifiable_actions, uuid4):
    # generate modified

    generator = generator_class(config, first_modification=first_modification,
                                fill_with_modification=fill_with_modification, modify_event_types=modify_event_types,
                                modification_distance=modification_distance,
                                total_modifiable_actions=total_modifiable_actions)
    events = generator.generate_story()
    world = generator.world

    choices = generate_all_possible_template_choices(events, templates['sentences'], modify_event_types)
    all_baseline = []
    all_modified = []
    for choice in tqdm(choices, position=1):
        story_id = uuid4()
        modified = {"id": story_id, 'qas': []}

        # realizer = Realizer(**templates)
        realizer = Realizer(**templates, unique_sentences=True)

        story, visits = realizer.realise_with_sentence_choices(events, world, choice)
        modified["passage"] = ' '.join(story)
        modified['passage_sents'] = story
        choices = realizer.context.choices
        template_choices = realizer.context.chosen_templates
        (single_span_questions, multi_span_questions, unanswerable_questions, abstractive_questions) = \
            generator.generate_questions(events, visits)
        for q in single_span_questions + multi_span_questions + unanswerable_questions + abstractive_questions:
            realizer.realise_question(q, story)

        for label, logical_qs in zip(answer_types, (single_span_questions, multi_span_questions,
                                                    unanswerable_questions, abstractive_questions)):
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

        # realizer = Realizer(**templates)
        realizer = Realizer(**templates, unique_sentences=True)
        story, visits = realizer.realise_with_choices(events, world, choices, template_choices)
        # print(20 * "===")
        # print("\n".join(story))
        # generator = load_class(config.get('generator.class'), StoryGenerator)(config)
        baseline["passage"] = ' '.join(story)
        baseline['passage_sents'] = story

        generator = generator_class({}, first_modification=first_modification,
                                    fill_with_modification=fill_with_modification,
                                    modify_event_types=modify_event_types,
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
        all_baseline.append(baseline)
        all_modified.append(modified)
    return all_baseline, all_modified
