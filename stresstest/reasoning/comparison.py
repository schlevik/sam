from functools import partial
from itertools import permutations, combinations
import random
from typing import List, cast, Callable, Any, Dict, Tuple

from loguru import logger

from stresstest.classes import EventPlan, Question, QuestionTypes, Reasoning, Event


def _is(et, cmd, arg):
    try:
        c, args = et
    except (TypeError, ValueError) as e:
        return False
    if ((c == cmd) or (cmd == '_')) and ((args == arg) or (arg == '_')):
        return True
    return _is(args[0], cmd, arg)


def to_question(events: List[Event], is_modified, generator, event_types,
                comparison_attribute, modify_event_type, target='actor', reverse=False) -> Question:
    assert target == 'actor'
    non_modified_events = [
        i for i, e in enumerate(event_types) if
        _is(e, EventPlan.Just, modify_event_type)
    ]
    modified_events = [
        i for i, e in enumerate(event_types) if
        _is(e, EventPlan.Mod, modify_event_type)
    ]
    evidence = non_modified_events + modified_events
    logger.debug(f"Evidence: {evidence}")
    logger.debug(f"len(events): {len(events)}")
    target_of_modified_events = events[modified_events[0]].actor \
        if target == 'actor' \
        else events[modified_events[0]].attributes[target]
    if target == 'actor':
        target_of_non_modified_events = next(
            events[i] for i in non_modified_events if events[i].actor != target_of_modified_events
        ).actor
    else:
        target_of_non_modified_events = next(
            events[i] for i in non_modified_events if events[i].attributes[target] != target_of_modified_events
        ).attributes[target]
    answer = target_of_non_modified_events if is_modified else target_of_modified_events
    return Question(
        type=QuestionTypes.OVERALL,
        target=target,
        evidence=sorted(evidence),
        event_type=modify_event_type,
        reasoning=f"{comparison.name}{'-reverse-' if reverse else '-'}{comparison_attribute}",
        question_data={
            "target-of-modified-events": target_of_modified_events,
            'target-of-non-modified-events': target_of_non_modified_events
        },
        answer=answer,
    )


def order(cmd, argmax_attribute, order):
    return EventPlan.Order, (cmd, argmax_attribute, order)


def modified(modify_event_type):
    return EventPlan.Mod, modify_event_type


def just(modify_event_type):
    return EventPlan.Just, modify_event_type


def same_actor_as(cmd, same_as_what):
    return EventPlan.SameActor, (cmd, same_as_what)


def generate_most_comparison_plans(
        max_sents: int, modify_event_type: str, comparison_attribute: List[Tuple[str, bool]], reverse=False
) -> List[EventPlan]:
    event_plans = []
    for attribute, temp_ordered in comparison_attribute:
        all_event_types = []
        # if temp_ordered:
        #    ...
        # else:
        comb_function = combinations  # if temp_ordered else permutations
        for j in range(2, max_sents):
            for indices in comb_function(list(range(max_sents)), j + 1):
                if not temp_ordered:
                    indices = list(indices)
                    random.shuffle(indices)
                if not reverse and temp_ordered:
                    indices = indices[::-1]
                *first, middle, last = indices
                assert len(indices) >= 3
                first_occurrence = min(i for i in indices if i != middle)
                event_types = [(EventPlan.Any, "_")] * max_sents

                for f in first:
                    event_types[f] = same_actor_as(modified(modify_event_type), first_occurrence)
                event_types[middle] = just(modify_event_type)

                event_types[last] = same_actor_as(just(modify_event_type), first_occurrence)

                if not temp_ordered:
                    for f in first:
                        event_types[f] = order(event_types[f],
                                               argmax_attribute=attribute, order=2 if not reverse else 0)
                    event_types[middle] = order(event_types[middle], argmax_attribute=attribute, order=1)
                    event_types[last] = order(event_types[last],
                                              argmax_attribute=attribute, order=0 if not reverse else 2)

                all_event_types.append((event_types, len(first)))
        for event_types, num_modifications in all_event_types:
            logger.debug('\n'.join(str(e) for e in event_types))
            logger.debug(f"Num Modifications: {num_modifications}")
            event_plans.append(EventPlan(
                event_types=tuple(event_types),
                num_modifications=num_modifications,
                modification_distance=...,
                first_modification=...,
                modify_event_type=modify_event_type,
                reasoning_type=comparison if not reverse else comparison_reverse,
                question_target='actor',
                to_question=cast(Callable[[List[Event], bool, Any], Question],
                                 partial(to_question,
                                         event_types=event_types,
                                         modify_event_type=modify_event_type,
                                         comparison_attribute=attribute,
                                         reverse=reverse
                                         )
                                 ),
                must_haves=[
                    attribute if _is(et, '_', modify_event_type) else None for et in event_types
                ]
            ))
    return event_plans


comparison = Reasoning(
    name='comparison',
    cardinality_event_plans=lambda max_sent: (max_sent - 1) * (max_sent - 1),
    questions_per_event_plan=3,
    generate_all_event_plans=generate_most_comparison_plans,
)

comparison_reverse = Reasoning(
    name='comparison-reverse',
    cardinality_event_plans=lambda max_sent: (max_sent - 1) * (max_sent - 1),
    questions_per_event_plan=3,
    generate_all_event_plans=partial(generate_most_comparison_plans, reverse=True),
)
