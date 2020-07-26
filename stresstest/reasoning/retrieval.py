import json
from functools import partial
from typing import List

from loguru import logger

from stresstest.classes import EventPlan, Question, QuestionTypes, Reasoning


def get_modification_data(max_sents):
    first_modifications = list(range(max_sents - 1))
    fill_with_modifications = [True, False]
    result = []
    for first_modification in first_modifications:
        for fill_with_modification in fill_with_modifications:
            modification_distances = range(1, max_sents - first_modification)
            for modification_distance in modification_distances:
                if modification_distance > 1 or fill_with_modification:
                    result.append((
                        first_modification,
                        fill_with_modification,
                        modification_distance,
                    ))
    return result


def get_event_types(modify_event_type, max_sents, first_modification, fill_with_modification, modification_distance,
                    reverse):
    modified = (EventPlan.MOD, modify_event_type)
    other = (EventPlan.NOT, modify_event_type)
    either = (EventPlan.ANY, "_")
    non_modified = (EventPlan.JUST, modify_event_type)
    event_types = [either] * max_sents
    event_types[first_modification] = modified
    event_types[first_modification + modification_distance] = non_modified
    for i in range(1, modification_distance):
        event_types[first_modification + i] = modified if fill_with_modification else other
    for i in range(first_modification):
        event_types[i] = other
    logger.debug(event_types)
    if reverse:
        event_types = event_types[::-1]
    return event_types


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
    # fill_with_modifications = [True, False]
    event_plans: List[EventPlan] = []
    # first_modifications = list(range(max_sents - 1))
    # logger.debug(f"First modification to appear in: {first_modifications}")
    # for first_modification in first_modifications:
    #     for fill_with_modification in fill_with_modifications:
    #         modification_distances = range(1, max_sents - first_modification)
    #         for modification_distance in modification_distances:
    #             modification_data = {
    #                 'first_modification': first_modification,
    #                 'fill_with_modification': fill_with_modification,
    #                 # 'modify_event_type': modify_event_type,
    #                 'modification_distance': modification_distance,
    #             }
    for first_modification, fill_with_modification, modification_distance in get_modification_data(max_sents):
        # logger.debug(json.dumps(modification_data, indent=4))
        # event_types = [either] * max_sents
        # event_types[first_modification] = modified
        # event_types[first_modification + modification_distance] = non_modified
        # for i in range(1, modification_distance):
        #     event_types[first_modification + i] = modified if fill_with_modification else other
        # for i in range(first_modification):
        #     event_types[i] = other
        # logger.debug(event_types)
        # if reverse:
        #     event_types = event_types[::-1]
        event_types = get_event_types(modify_event_type, max_sents, first_modification, fill_with_modification,
                                      modification_distance, reverse)

        # if modification distance == 1 then there's nothing to fill so they become identical
        # if modification_distance > 1 or fill_with_modification:
        # python closures are python closures
        def to_question(events, is_modified, fm=first_modification, md=modification_distance):
            if reverse:
                evidence = len(events) - 1 - (fm if not is_modified else fm + md)
            else:
                evidence = fm if not is_modified else fm + md
            answer = events[evidence]['actor']
            return Question(
                type=QuestionTypes.DIRECT,
                target='actor',
                evidence=[evidence],
                event_type=modify_event_type,
                reasoning=(retrieval_reverse if reverse else retrieval).name,
                question_data={"n": 1},
                answer=answer,
            )

        event_plans.append(EventPlan(
            event_types=tuple(event_types),
            num_modifications=sum(et == modified for et in event_types),
            modification_distance=modification_distance,
            first_modification=first_modification,
            modify_event_type=modify_event_type,
            reasoning_type=(retrieval_reverse if reverse else retrieval),
            question_target='actor',
            to_question=to_question,
            must_haves=[],
            # ),
        ))
        for attribute in attributes:
            # python closures are python closures
            def to_question(events, is_modified, fm=first_modification, md=modification_distance,
                            attr=attribute):
                assert len(events) == 5
                if reverse:
                    evidence = len(events) - 1 - (fm if not is_modified else fm + md)
                else:
                    evidence = fm if not is_modified else fm + md
                answer = events[evidence]['attributes'][attr]
                return Question(
                    type=QuestionTypes.DIRECT,
                    target=attr,
                    evidence=[evidence],
                    event_type=modify_event_type,
                    reasoning=(retrieval_reverse if reverse else retrieval).name,
                    question_data={"n": 1},
                    answer=answer,
                )

            must_haves = [
                attribute if et == modify_event_type and (mod == EventPlan.JUST or mod == EventPlan.MOD)
                else None for mod, et in event_types
            ]
            event_plans.append(EventPlan(
                event_types=tuple(event_types),
                num_modifications=sum(et == modified for et in event_types),
                modification_distance=modification_distance,
                first_modification=first_modification,
                modify_event_type=modify_event_type,
                reasoning_type=(retrieval_reverse if reverse else retrieval),
                question_target=attribute,
                to_question=to_question,
                must_haves=must_haves, ))

    return event_plans


retrieval = Reasoning(
    name='retrieval',
    cardinality_event_plans=lambda max_sent: (max_sent - 1) * (max_sent - 1),
    questions_per_event_plan=3,
    generate_all_event_plans=generate_all_retrieval_event_plans,
)

retrieval_reverse = Reasoning(
    name='retrieval-reverse',
    cardinality_event_plans=lambda max_sent: (max_sent - 1) * (max_sent - 1),
    questions_per_event_plan=3,
    generate_all_event_plans=partial(generate_all_retrieval_event_plans, reverse=True),
)
