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
                comparison_attribute, modify_event_type, target='actor', reverse=False, temp_ordered=False) -> Question:
    assert target == 'actor'
    for e in event_types:
        logger.debug(e)
    logger.debug(f"temp ordered: {temp_ordered}")
    logger.debug(f"is modified: {is_modified}")
    logger.debug(f"comparison attribute: {comparison_attribute}")
    logger.debug(f"reverse: {reverse}")
    idx_of_true_when_modified = \
        [i for i, e in enumerate(event_types) if _is(e, EventPlan.Just, '_') and not _is(e, EventPlan.SameActor, '_')]
    assert len(idx_of_true_when_modified) == 1
    idx_of_true_when_modified = idx_of_true_when_modified[0]
    if temp_ordered:
        idx_of_true_when_not_modified = [i for i, e in enumerate(event_types) if _is(e, EventPlan.SameActor, '_')]
        if is_modified:
            idx_of_true_when_not_modified = [
                i for i in idx_of_true_when_not_modified if _is(event_types[i], EventPlan.Just, '_')
            ]
            assert len(idx_of_true_when_not_modified) == 1
            idx_of_true_when_not_modified = idx_of_true_when_not_modified[0]
        else:
            idx_of_true_when_not_modified = idx_of_true_when_not_modified[0 if reverse else -1]
    else:
        if is_modified:
            idx_of_true_when_not_modified = [
                i for i, e in enumerate(event_types) if
                _is(e, EventPlan.Just, '_') and _is(e, EventPlan.SameActor, '_')
            ]
        else:
            idx_of_true_when_not_modified = [
                i for i, e in enumerate(event_types) if _is(e, EventPlan.SameActor, '_') and
                                                        e[0] == EventPlan.Order and e[1][2] == (-99 if reverse else 99)
            ]
        assert len(idx_of_true_when_not_modified) == 1
        idx_of_true_when_not_modified = idx_of_true_when_not_modified[0]
    evidence = [idx_of_true_when_modified, idx_of_true_when_not_modified]

    logger.debug(f"Evidence: {evidence}")
    logger.debug(f"len(events): {len(events)}")

    # if not (isinstance(idx_of_true_when_modified, int) and isinstance(idx_of_true_when_not_modified, int) and len(
    #         evidence) == 2):
    #     print(idx_of_true_when_modified)
    #     print(idx_of_true_when_not_modified)
    #     print(event_types)
    #     for e in event_types:
    #         print(e)
    #     print(reverse)  # false
    #     print(is_modified)  # True
    #     print(temp_ordered)  # false
    #     raise NotImplementedError()
    assert len(evidence) == 2
    answer_when_modified = events[idx_of_true_when_modified].actor
    answer_when_not_modified = events[idx_of_true_when_not_modified].actor
    # if not answer_when_modified != answer_when_not_modified:
    #     print(event_types)
    #     for e in event_types:
    #         print(e)
    #     print(idx_of_true_when_modified)
    #     print(idx_of_true_when_not_modified)
    #     raise NotImplementedError()
    answer = answer_when_modified if is_modified else answer_when_not_modified
    return Question(
        type=QuestionTypes.OVERALL,
        target=target,
        evidence=sorted(evidence),
        event_type=modify_event_type,
        reasoning=f"{comparison.name}{'-reverse-' if reverse else '-'}{comparison_attribute}",
        question_data={
            "answer-when-modified": answer_when_modified,
            'answer-when-not-modified': answer_when_not_modified
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
                    ff, *fs = first
                    event_types[ff] = order(event_types[f],
                                            argmax_attribute=attribute, order=99 if not reverse else -99)
                    for f in fs:
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
                                         reverse=reverse,
                                         temp_ordered=temp_ordered,
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
