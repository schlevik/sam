import sys
from collections import defaultdict
from copy import deepcopy
from functools import partial
from operator import attrgetter
import random
from typing import List

from joblib import delayed, Parallel
from loguru import logger
from tqdm import tqdm

from stresstest.classes import Reasoning
from stresstest.comb_utils import generate_one_possible_template_choice
from stresstest.football.generate_with_modifier import PlannedFootballModifierGenerator
from stresstest.realize import Realizer


def generate_and_realise(bundle, config, modify_event_type, modifier_type, max_sents, per_modify_distance_per_reasoning,
                         reasonings, max_modifiers, use_mod_distance=False, mute=False, num_workers=8,
                         deterministic=True):
    # TODO: do parallel
    result = generate_balanced(
        modify_event_type, config, bundle, max_sents, reasonings, modifier_type, per_modify_distance_per_reasoning,
        max_modifiers, use_mod_distance, mute, num_workers=num_workers
    )

    templates = bundle.templates_modifier
    # validate templates here real quick
    realizer = Realizer(**templates, validate=True)
    if deterministic:
        seeds = [random.randint(0, sys.maxsize) for _ in result]
    else:
        seeds = [None for _ in result]
    if num_workers > 1:
        realized = Parallel(num_workers)(
            (delayed(_do_realize)
             (config, event_plan, events, modifier_type,
              template_choices, templates, world, seed)
             ) for (event_plan, events, template_choices, world), seed in
            zip(tqdm(result, desc='Realising...', disable=mute), seeds))
    else:
        realized = [
            _do_realize(
                config, event_plan, events, modifier_type,
                template_choices, templates, world, seed
            ) for (event_plan, events, template_choices, world), seed in
            zip(tqdm(result, desc='Realising...', disable=mute), seeds)]
    return [(*z1, *z2) for z1, z2 in zip(result, realized)]


def _do_realize(config, event_plan, events, modifier_type, template_choices, templates, world, seed=None):
    if seed:
        random.seed(seed)
    realizer = Realizer(**templates, validate=False)
    story, visits = realizer.realise_with_sentence_choices(events, world, template_choices)
    choices = realizer.context.choices
    events = deepcopy(events)
    for event in events:
        event.features = []
    realizer = Realizer(**templates, unique_sentences=True)
    baseline_story, baseline_visits = realizer.realise_with_choices(events, world, choices, template_choices)
    generator = partial(PlannedFootballModifierGenerator, config=config,
                        modifier_type=modifier_type)
    generator_instance: PlannedFootballModifierGenerator = generator(event_plan=event_plan)
    # this is for single span extraction only atm
    qs = generator_instance.generate_questions_from_plan(event_plan, events)[0]
    mqs = generator_instance.generate_questions_from_plan(event_plan, events, True)[0]

    for q, mq in zip(qs, mqs):
        try:
            realizer.realise_question(q, story, ignore_missing_keys=False)
        except IndexError as e:
            print(f"{q}\n{baseline_story}\n{event_plan.event_types}\n{event_plan.must_haves}\n{template_choices}")
        mq.realized = q.realized
        assert mq.realized, f"{mq}\n{story}"
        try:
            mq.answer = realizer._fix_units(mq, story)
        except IndexError as e:
            print(f"{mq}\n{story}\n{event_plan.event_types}\n{event_plan.must_haves}\n{template_choices}")
            raise NotImplementedError(e)
        assert q.answer in " ".join(
            story), f"{q}\n{baseline_story}\n{event_plan.event_types}\n{event_plan.must_haves}\n{template_choices}"
        assert mq.answer in " ".join(
            story), f"{mq}\n{story}\n{event_plan.event_types}\n{event_plan.must_haves}\n{template_choices}"
    return baseline_story, mqs, qs, story


def generate_balanced(modify_event_type, config, bundle, max_sents, reasonings: List[Reasoning],
                      modifier_type, per_modify_distance_per_reasoning=10,
                      max_modifiers=4, use_mod_distance=False, mute=False, num_workers=8):
    attr = attrgetter('modification_distance' if use_mod_distance else 'num_modifications')
    result = []
    with tqdm(disable=mute, total=per_modify_distance_per_reasoning * max_modifiers * len(reasonings)) as pbar:
        for reasoning in reasonings:
            all_event_plans = reasoning.generate_all_event_plans(max_sents, modify_event_type,
                                                                 bundle.reasoning_map[reasoning.name])
            event_plans_by_num_modifications = defaultdict(list)
            for ep in all_event_plans:
                value = attr(ep)
                if value <= max_modifiers:
                    event_plans_by_num_modifications[value].append(ep)
            for num_modifiers, event_plans in event_plans_by_num_modifications.items():
                template_choices = []
                if len(event_plans) > per_modify_distance_per_reasoning:
                    eps = random.sample(event_plans, per_modify_distance_per_reasoning)
                else:
                    eps = event_plans
                    while len(eps) < per_modify_distance_per_reasoning:
                        eps.append(random.choice(eps))

                seeds = [random.randint(0, sys.maxsize) for _ in eps]
                if num_workers > 1:
                    stories_and_worlds = Parallel(num_workers)(
                        delayed(_do_generate)(
                            PlannedFootballModifierGenerator, config=config,
                            modifier_type=modifier_type, ep=ep, mute=True, seed=seed)
                        for ep, seed in zip(eps, seeds))
                else:
                    stories_and_worlds = [
                        _do_generate(
                            PlannedFootballModifierGenerator, config=config,
                            modifier_type=modifier_type, ep=ep, mute=True, seed=seed)
                        for ep, seed in zip(eps, seeds)
                    ]
                stories, worlds = zip(*stories_and_worlds)
                for story, ep in zip(stories, eps):
                    template_choice = generate_one_possible_template_choice(
                        story, bundle.templates_modifier['sentences'], ep.must_haves, bundle.has_template_attribute
                    )
                    # TODO patience!
                    while template_choice in template_choices:
                        template_choice = generate_one_possible_template_choice(
                            story, bundle.templates_modifier['sentences'], ep.must_haves, bundle.has_template_attribute
                        )
                        print("Yup, we're in a forever-loop....")
                    pbar.update()
                    template_choices.append(template_choice)
                try:
                    assert len(stories) == len(eps) == len(template_choices) == len(worlds)
                except AssertionError:
                    print(len(stories), len(eps), len(template_choices), len(worlds))
                    raise NotImplementedError()
                assert len(stories) == len(eps) == len(template_choices) == len(worlds)
                result.extend(zip(eps, stories, template_choices, worlds))

    return result


def _do_generate(generator_class, config, modifier_type, ep, mute, seed=None):
    if seed:
        random.seed(seed)
    if mute:
        logger.remove()
    generator_instance = generator_class(config=config, modifier_type=modifier_type, event_plan=ep)
    return generator_instance.generate_story(return_world=True)
