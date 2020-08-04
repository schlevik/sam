from collections import Counter, defaultdict
from copy import deepcopy
from itertools import groupby, permutations, product
from math import factorial
from operator import attrgetter
from random import shuffle, sample
from typing import List, Dict, Tuple, Optional, Callable

from loguru import logger
from pyhocon import ConfigFactory

from stresstest.classes import Event, Choices, YouIdiotException

key = attrgetter('event_type')


def paths(key, path=()):
    if isinstance(key, list):
        # key is leaf
        yield path, key
    else:
        for n, s in key.items():
            for p in paths(s, path + (n,)):
                yield p


def split(templates: Dict[str, Dict[str, List[str]]], event_types_to_split: List[str], split_ratio: float,
          split_q_templates: Optional[Callable] = None):
    first_split = deepcopy(templates)
    second_split = deepcopy(templates)
    split_q_templates = split_q_templates or []
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
    if split_q_templates:
        first_q_predicate = split_q_templates
        first_split, second_split = filter_question_templates(first_q_predicate,
                                                              templates['question_templates'])
    return first_split, second_split


def filter_question_templates(predicate, question_templates):
    hits = ConfigFactory.from_dict({})
    rest = ConfigFactory.from_dict({})
    for path, q_templates in paths(question_templates):
        logger.debug(path)
        logger.debug(q_templates)
        q_templates_hits = []
        q_templates_rest = []
        for i, q_template in enumerate(q_templates):
            if predicate(i, q_template):
                q_templates_hits.append(q_template)
            else:
                q_templates_rest.append(q_template)
        assert len(q_templates_hits) + len(q_templates_rest) == len(q_templates)
        assert set(q_templates_hits + q_templates_rest) == set(q_templates)
        assert q_templates_hits
        assert q_templates_rest
        hits.put('.'.join(path), q_templates_hits)
        rest.put('.'.join(path), q_templates_rest)
    hits = hits.as_plain_ordered_dict()
    rest = rest.as_plain_ordered_dict()
    assert [k for k, _ in paths(hits)] == \
           [k for k, _ in paths(rest)] == \
           [k for k, _ in paths(question_templates)]
    return hits, rest


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
