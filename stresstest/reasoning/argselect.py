from functools import partial
from itertools import permutations
from typing import List, cast, Callable, Any, Dict

from loguru import logger

from stresstest.classes import EventPlan, Question, QuestionTypes, Reasoning, Event


def to_question(events: List[Event], is_modified, generator, event_types,
                argselect_attribute,
                modify_event_type, target='actor', reverse=False) -> Question:
    if not is_modified:

        evidence = [
            i for i, (_, ((mod, _), _, order)) in enumerate(event_types) if
            mod == EventPlan.Mod and order == (0 if reverse else 3)
        ]
    else:
        evidence = [i for i, (_, ((mod, _), _, order)) in enumerate(event_types) if mod == EventPlan.Just]

    answer = events[evidence[0]].actor if target == 'actor' else events[evidence[0]].attributes[target]
    return Question(
        type=QuestionTypes.OVERALL,
        target=target,
        evidence=evidence,
        event_type=modify_event_type,
        reasoning=f"{argmin.name if reverse else argmax.name}-{argselect_attribute}",
        question_data=dict(),
        answer=answer,
    )


def either(argmax_attribute, order):
    return EventPlan.Order, ((EventPlan.Any, "_"), argmax_attribute, order)


def modified(modify_event_type, argmax_attribute, order):
    return EventPlan.Order, ((EventPlan.Mod, modify_event_type), argmax_attribute, order)


def just(modify_event_type, argmax_attribute, order):
    return EventPlan.Order, ((EventPlan.Just, modify_event_type), argmax_attribute, order)


def generate_all_argselect_event_plans(
        max_sents: int, modify_event_type: str, attributes: Dict[str, List[str]], reverse=False
) -> List[EventPlan]:
    all_event_types = []
    for argmax_attribute, target_attributes in attributes.items():

        for i in range(1, max_sents):
            for indices in permutations(list(range(max_sents)), i + 1):
                event_types = [either(argmax_attribute, 3 if reverse else 0)] * max_sents
                event_types[indices[0]] = modified(modify_event_type, argmax_attribute, 0 if reverse else 3)
                for idx in indices[1:-1]:
                    event_types[idx] = modified(modify_event_type, argmax_attribute, 1 if reverse else 2)
                event_types[indices[-1]] = just(modify_event_type, argmax_attribute, 2 if reverse else 1)
                all_event_types.append(event_types)
        logger.debug(f"Have {len(all_event_types)} event type plans")
        for et in all_event_types:
            logger.debug(f"{et}")

        event_plans = []
        for event_types in all_event_types:
            event_plans.append(EventPlan(
                event_types=tuple(event_types),
                num_modifications=sum(mod == EventPlan.Mod for _, ((mod, et), *_) in event_types),
                modification_distance=...,
                first_modification=...,
                modify_event_type=modify_event_type,
                reasoning_type=argmin if reverse else argmax,
                question_target='actor',
                to_question=cast(Callable[[List[Event], bool, Any], Question],
                                 partial(to_question,
                                         event_types=event_types,
                                         modify_event_type=modify_event_type,
                                         argselect_attribute=argmax_attribute,
                                         reverse=reverse
                                         )
                                 ),
                must_haves=[
                    argmax_attribute if et == modify_event_type and (mod == EventPlan.Just or mod == EventPlan.Mod)
                    else None for _, ((mod, et), *_) in event_types],
                # ),
            ))
            for attribute in target_attributes:
                must_have = (attribute, argmax_attribute) if (attribute != argmax_attribute) else argmax_attribute
                must_haves = [must_have
                              if et == modify_event_type and (mod == EventPlan.Just or mod == EventPlan.Mod)
                              else None for _, ((mod, et), *_) in event_types]
                event_plans.append(EventPlan(
                    event_types=tuple(event_types),
                    num_modifications=sum(mod == EventPlan.Mod for _, ((mod, et), *_) in event_types),
                    modification_distance=...,
                    first_modification=...,
                    modify_event_type=modify_event_type,
                    reasoning_type=argmin if reverse else argmax,
                    question_target=attribute,
                    to_question=cast(Callable[[List[Event], bool, Any], Question],
                                     partial(to_question,
                                             event_types=event_types,
                                             modify_event_type=modify_event_type,
                                             argselect_attribute=argmax_attribute,
                                             target=attribute, reverse=reverse
                                             )
                                     ),
                    must_haves=must_haves, ))

        return event_plans


argmax = Reasoning(
    name='argmax',
    cardinality_event_plans=lambda max_sent: (max_sent - 1) * (max_sent - 1),
    questions_per_event_plan=3,
    generate_all_event_plans=generate_all_argselect_event_plans,
)
argmin = Reasoning(
    name='argmin',
    cardinality_event_plans=lambda max_sent: (max_sent - 1) * (max_sent - 1),
    questions_per_event_plan=3,
    generate_all_event_plans=partial(generate_all_argselect_event_plans, reverse=True),
)
