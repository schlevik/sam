from itertools import groupby, permutations, product
from math import factorial
from operator import attrgetter
from random import shuffle
from typing import List, Dict

from loguru import logger

from stresstest.classes import Event

key = attrgetter('event_type')


def split(templates: Dict[str, List[str]], event_types_to_split: List[str], split_ratio: float):
    first_split = dict()
    second_split = dict()
    for et, sents in templates.items():
        if et in event_types_to_split:
            sents = sents[:]
            shuffle(sents)
            split_idx = round(split_ratio * len(sents))
            first_split[et] = sents[split_idx:]
            second_split[et] = sents[:split_idx]
            assert len(first_split[et]) + len(second_split[et]) == len(sents)
        else:
            second_split[et] = sents
            first_split[et] = sents
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
