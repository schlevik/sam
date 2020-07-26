import json
from functools import partial
from typing import List

from loguru import logger

from stresstest.classes import EventPlan, Question, QuestionTypes, Reasoning, Event
from stresstest.reasoning.retrieval import get_event_types, get_modification_data


def to_question(events: List[Event], is_modified, generator, et, met, target='actor', reverse=False):
    #
    evidence = [
        next(iter([i for i, (mod, _) in enumerate(et) if mod == 'just'][::-1 if reverse else 1]))
    ]
    if is_modified:
        evidence += [
            next(i for i, (mod, _) in enumerate(et) if mod == 'just' and i not in evidence)
        ]
    else:
        evidence += [
            next(iter([i for i, (mod, _) in enumerate(et) if mod == 'modified'][::-1 if reverse else 1]))
        ]
    evidence = sorted(evidence)
    answer_event = events[evidence[-1] if not reverse else evidence[0]]
    bridge_event = events[evidence[0] if not reverse else evidence[-1]]
    answer = answer_event.actor if target == 'actor' else answer_event.attributes[target]
    return Question(
        type=QuestionTypes.DIRECT,
        target=target,
        evidence=evidence,
        event_type=met,
        reasoning=bridge_reverse.name,
        question_data={
            "bridge-event": bridge_event,
        },
        answer=answer,
    )


def generate_all_bridge_event_plans(max_sents, modify_event_type, attributes, reverse=False) -> List[EventPlan]:
    """
    Cardinality: (max_sents-1)^2 * |attributes|
    Args:
        reasoning_type:
        max_sents:
        modify_event_type:

    Returns:

    """
    modified = (EventPlan.MOD, modify_event_type)
    other = (EventPlan.NOT, modify_event_type)
    either = (EventPlan.ANY, "_")
    non_modified = (EventPlan.JUST, modify_event_type)
    just_any = (EventPlan.JUST, '_')
    event_plans = []
    all_event_types = []
    j = 1
    while j <= max_sents:
        logger.debug(f"j: {j}")
        ets = [get_event_types(modify_event_type, max_sents - j, *md, reverse=False) for md in
               get_modification_data(max_sents - j)]
        if ets:
            front_ets = [[either] * (j - 1) + [just_any] + et for et in ets]
            all_event_types.extend(front_ets)
        j += 1

    assert all(len(et) == max_sents for et in all_event_types)
    assert len(set(tuple(e) for e in all_event_types)) == len(all_event_types)
    event_plans: List[EventPlan] = []

    for event_types in all_event_types:

        if reverse:
            print(event_types)
            print('turns into')
            event_types = event_types[::-1]
            print(event_types)
            print(20 * '====')
        event_plans.append(EventPlan(
            event_types=tuple(event_types),
            num_modifications=sum(et == modified for et in event_types),
            modification_distance=...,
            first_modification=...,
            modify_event_type=modify_event_type,
            reasoning_type=bridge,
            question_target='actor',
            to_question=partial(to_question, et=event_types, met=modify_event_type, reverse=reverse),
            must_haves=[],
            # ),
        ))
        for attribute in attributes:
            must_haves = [
                attribute if et == modify_event_type and (mod == EventPlan.JUST or mod == EventPlan.MOD)
                else None for mod, et in event_types
            ]

            event_plans.append(EventPlan(
                event_types=tuple(event_types),
                num_modifications=sum(et == modified for et in event_types),
                modification_distance=...,
                first_modification=...,
                modify_event_type=modify_event_type,
                reasoning_type=bridge,
                question_target=attribute,
                to_question=partial(to_question, et=event_types, met=modify_event_type, target=attribute,
                                    reverse=reverse),
                must_haves=must_haves, ))

    return event_plans


bridge = Reasoning(
    name='bridge',
    cardinality_event_plans=lambda max_sent: (max_sent - 1) * (max_sent - 1),
    questions_per_event_plan=3,
    generate_all_event_plans=generate_all_bridge_event_plans,
)
bridge_reverse = Reasoning(
    name='bridge-reverse',
    cardinality_event_plans=lambda max_sent: (max_sent - 1) * (max_sent - 1),
    questions_per_event_plan=3,
    generate_all_event_plans=partial(generate_all_bridge_event_plans, reverse=True),
)
