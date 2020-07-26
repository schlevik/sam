import json
from functools import partial
from typing import List

from loguru import logger

from stresstest.classes import EventPlan, Question, QuestionTypes, Reasoning
from stresstest.reasoning.retrieval import get_event_types, get_modification_data


def generate_all_retrieval_event_plans(max_sents, modify_event_type, attributes, reverse=False) -> List[EventPlan]:
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
    event_plans = []
    all_event_types = []
    j = 1
    while j <= max_sents:
        ets = [get_event_types(modify_event_type, max_sents - j, *md, reverse) for md in
               get_modification_data(max_sents - j)]
        if ets:
            front_ets = [[other] * (j - 1) + [non_modified] + et for et in ets]
            back_ets = [et + [non_modified] + [other] * (j - 1) for et in ets]
            all_event_types.extend(front_ets)
            all_event_types.extend(back_ets)
        j += 1

    assert all(len(et) == max_sents for et in all_event_types)
    assert len(set(tuple(e) for e in all_event_types)) == len(all_event_types)

    for event_types in all_event_types:
        # if modification distance == 1 then there's nothing to fill so they become identical
        # if modification_distance > 1 or fill_with_modification:
        # python closures are python closures
        # first_modification = next(i for i, m in enumerate(event_types) if m == modified)
        # modification_distance = next(
        #    i for i, m in enumerate(event_types[first_modification:], first_modification) if
        #    m == non_modified) - first_modification

        def to_question(events, is_modified, generator, ets=event_types,
                        met=modify_event_type):
            if is_modified:
                if reverse:
                    evidence = [i for i, m in enumerate(ets) if m == non_modified][-2:]
                else:
                    evidence = [i for i, m in enumerate(ets) if m == non_modified][:2]
            else:
                if reverse:
                    evidence = [i for i, m in enumerate(ets) if m == non_modified or m == modified][-2:]
                else:
                    evidence = [i for i, m in enumerate(ets) if m == non_modified or m == modified][:2]
            assert len(evidence) == 2
            answer = events[evidence[-1]].actor if not reverse else events[evidence[0]].actor
            return Question(
                type=QuestionTypes.DIRECT,
                target='actor',
                evidence=evidence,
                event_type=modify_event_type,
                reasoning='retrieval' if not reverse else 'retrieval-reverse',
                question_data={"n": 2},
                answer=answer,
            )

        event_plans.append(EventPlan(
            event_types=tuple(event_types),
            num_modifications=sum(et == modified for et in event_types),
            modification_distance=...,
            first_modification=...,
            modify_event_type=modify_event_type,
            reasoning_type=retrieval_two if not reverse else retrieval_two_reverse,
            question_target='actor',
            to_question=to_question,
            must_haves=[],
            # ),
        ))
        for attribute in attributes:
            # python closures are python closures
            def to_question(events, is_modified, generator, ets=event_types,
                            # fm=first_modification, md=modification_distance,
                            attr=attribute):
                if is_modified:
                    if reverse:
                        evidence = [i for i, m in enumerate(ets) if m == non_modified][-2:]
                    else:
                        evidence = [i for i, m in enumerate(ets) if m == non_modified][:2]
                else:
                    if reverse:
                        evidence = [i for i, m in enumerate(ets) if m == non_modified or m == modified][-2:]
                    else:
                        evidence = [i for i, m in enumerate(ets) if m == non_modified or m == modified][:2]
                answer = events[evidence[-1]].attributes[attr] if not reverse else events[evidence[0]].attributes[
                    attr]
                return Question(
                    type=QuestionTypes.DIRECT,
                    target=attr,
                    evidence=evidence,
                    event_type=modify_event_type,
                    reasoning='retrieval' if not reverse else 'retrieval-reverse',
                    question_data={"n": 2},
                    answer=answer,
                )

            must_haves = [
                attribute if et == modify_event_type and (mod == EventPlan.JUST or mod == EventPlan.MOD)
                else None for mod, et in event_types
            ]
            # first must_have is not must have
            # for i, _ in enumerate(must_haves):
            #     if event_types[i] == non_modified:
            #         must_haves[i] = None
            #         break
            event_plans.append(EventPlan(
                event_types=tuple(event_types),
                num_modifications=sum(et == modified for et in event_types),
                modification_distance=...,
                first_modification=...,
                modify_event_type=modify_event_type,
                reasoning_type=retrieval_two if not reverse else retrieval_two_reverse,
                question_target=attribute,
                to_question=to_question,
                must_haves=must_haves, ))

    return event_plans


retrieval_two = Reasoning(
    name='retrieval-two',
    cardinality_event_plans=lambda max_sent: (max_sent - 1) * (max_sent - 1),
    questions_per_event_plan=3,
    generate_all_event_plans=generate_all_retrieval_event_plans,
)

retrieval_two_reverse = Reasoning(
    name='retrieval-two-reverse',
    cardinality_event_plans=lambda max_sent: (max_sent - 1) * (max_sent - 1),
    questions_per_event_plan=3,
    generate_all_event_plans=partial(generate_all_retrieval_event_plans, reverse=True),
)
