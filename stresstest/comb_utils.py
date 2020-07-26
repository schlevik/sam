from collections import Counter, defaultdict
from copy import deepcopy
from itertools import groupby, permutations, product
from math import factorial
from operator import attrgetter
from random import shuffle, sample
from typing import List, Dict, Tuple

from loguru import logger

from stresstest.classes import Event, Choices, YouIdiotException

key = attrgetter('event_type')


def split(templates: Dict[str, Dict[str, List[str]]], event_types_to_split: List[str], split_ratio: float):
    first_split = deepcopy(templates)
    second_split = deepcopy(templates)
    for event_type, sents in templates['sentences'].items():
        sents = deepcopy(sents)
        if event_type in event_types_to_split:
            shuffle(sents)
            split_idx = round(split_ratio * len(sents))
            first_split['sentences'][event_type] = sents[split_idx:]
            second_split['sentences'][event_type] = sents[:split_idx]
            assert len(first_split['sentences'][event_type]) + len(second_split['sentences'][event_type]) == len(
                templates['sentences'][event_type])
        else:
            second_split['sentences'][event_type] = sents
            first_split['sentences'][event_type] = sents
    return first_split, second_split


def calculate_num_of_permutations(events: List[Event], templates, events_to_permute):
    permutations_per_event_type = []
    for k, g in groupby(sorted(events, key=key), key):
        logger.debug(f"Event type: {k}")
        if k in events_to_permute:
            n = len(templates[k])
            r = len(list(g))
            npr = int(factorial(n) / factorial(n - r))
            logger.debug(f"{n}P{r} = {npr}")
            permutations_per_event_type.append(npr)

    result = 1
    for p in permutations_per_event_type:
        result *= p
    return result


def generate_one_possible_template_choice(events: List[Event], templates, must_haves, has_template_attribute):
    if any(must_haves):
        must_have_per_event = Counter()
        for e, mh in zip(events, must_haves):
            et = e.event_type
            if mh:
                must_have_per_event[(et, mh)] += 1
        for (e, mh), needed in must_have_per_event.items():
            has = len([t for t in templates[e] if has_template_attribute(t, mh)])
            if not has >= needed:
                raise YouIdiotException(f"{e}.{mh} does not have enough templates! has: {has} needed: {needed} ")

        choices = defaultdict(list)
        # noinspection PyTypeChecker
        result: List[Tuple[str, int]] = [None] * len(events)
        # do the one with must_haves first
        for i, (event, must_have) in enumerate(zip(events, must_haves)):
            et = event.event_type
            logger.debug("Choices[et]", choices[et])
            if must_have:
                candidate_choices = [i for i, t in enumerate(templates[et])
                                     if has_template_attribute(t, must_have) and i not in choices[et]]
                logger.debug(f"Candidates: {candidate_choices}")
                choice = Choices(candidate_choices).random()
                assert choice is not None, must_haves
                logger.debug(f"Choice: {choice}")
                choices[et].append(choice)
                result[i] = (et, choice)
                logger.debug(f"Result: {result}")
        # do the others later
        for i, (event, must_have) in enumerate(zip(events, must_haves)):
            et = event.event_type
            logger.debug(f"Choices[et]: {choices[et]}")
            if not must_have:
                candidate_choices = [i for i, t in enumerate(templates[et])
                                     if i not in choices[et]]
                logger.debug(f"Candidates {candidate_choices}")
                choice = Choices(candidate_choices).random()
                assert choice is not None, must_haves
                choices[et].append(choice)
                result[i] = (et, choice)
        return result
    else:
        event_types = Counter()
        for e in events:
            event_types[e.event_type] += 1
        choice_iter = {e: iter(sample(range(len(templates[e])), n)) for e, n in event_types.items()}
        result = []
        for e in events:
            result.append((e.event_type, next(choice_iter[e.event_type])))
        return result


def generate_all_possible_template_choices(events: List[Event], templates, events_to_permute):
    permutations_per_event_type = []
    event_types = [e.event_type for e in events]
    event_order_map = dict()
    for i, (k, g) in enumerate(groupby(sorted(events, key=key), key)):
        event_order_map[k] = i
        g = list(g)
        r = len(g)
        n = range(len(templates[k]))
        if k in events_to_permute:
            logger.debug(f"{k}: {len(templates[k])}P{r}")
            permutations_per_event_type.append(permutations(n, r))
        else:
            logger.debug(f"{k}: 1*{r}")
            permutations_per_event_type.append([['<any>' for _ in g]])
    all_possible_combinations = product(*permutations_per_event_type)
    logger.debug(event_order_map)
    result = []
    for combination in all_possible_combinations:
        iters = [iter(x) for x in combination]
        result.append([(et, next(iters[event_order_map[et]])) for et in event_types])
    return result
