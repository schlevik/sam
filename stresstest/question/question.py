import logging
from ailog import Loggable
from typing import List, Optional
from stresstest.classes import Path, Choices
from stresstest.question.independent_conditions import QuestionCondition


class Question(Loggable):
    """
    Container class for the question.

    Attributes:
        target (str):
            Target type in the path (i.e. player, team)
        action (str):
            Action type in the path (i.e. goal, free-kick)
        answer_position (int):
            The position of the answer in the path.
        path (Path):
            The passage in logical form (a path in the content graph)
            the question asks about.

    """

    def __init__(self, target: str, action: str, answer_position: int,
                 path: Path):
        self.target = target
        self.action = action
        self.answer_position = answer_position
        self.path = path

    def __repr__(self):
        return (f"{self.target} {self.action}: "
                f"{self.path[self.answer_position], self.answer_position}")


def create_question(target: str, action: str, path: Path,
                    conditions: List[QuestionCondition]) -> Optional[Question]:
    """
    Creates a :class:`Question` for a path given a concrete target and
    action if possible.

    The created question satisfies given conditions.

    Returns `None` if the question is ambiguous (more than 1 answer exists) or
    unanswerable.

    Args:
        target:
            Concrete question target (e.g. player)
        action:
            Concrete question action (e.g. goal)
        path:
            Passage in logical form (a path in the content graph)
            the generated question should ask about.
        conditions:
             Conditions the generated question should conform to.

    Returns:
        A :class:`Question` if a question with given action and target
        is unambiguously answerable from the given path and suffices
        given conditions.
    """
    candidates = Choices([i for i, _ in enumerate(path)])
    for c in conditions:
        candidates = c(path=path, choices=candidates, target=target,
                       action=action)
    if not candidates:
        return None
    if len(candidates) > 1:
        logging.getLogger(__name__).info(
            f"{target}, {action} is ambiguous under "
            f"{[f.__class__.__name__ for f in conditions]}: "
            f"{[path[c] for c in candidates]}")
        return None
    else:
        q = Question(target, action, candidates[0], path)
        return q


def generate_question(path: Path,
                      targets: List[str],
                      actions: List[str],
                      conditions: List[QuestionCondition],
                      ) -> Optional[Question]:
    """
    Generates a random question that can be asked given the path and
    conforms to the given conditions, if possible. If not, returns 0.

    Generated questions are per definition `unique` (i.e. there cannot
    be two correct answers) and answerable (i.e. there cannot be
    no answer).

    Args:
        path:
            Passage in logical form (a path in the content graph)
            the generated question should ask about.
        targets:
            Possible question target types. (e.g. player, team, etc)
        actions:
            Possible question action types. (e.g. free kick, goal, etc)
        conditions:
            Conditions the generated question should conform to.

    Returns:
        A random question about the path that suffices all given
        conditions. If no question can be asked, returns ``None``.

    """
    logger = logging.getLogger("generate_question")
    logger.info(f"Generating question from: {path}")
    questions = (create_question(t, a, path, conditions) for t in targets for a
                 in actions)
    possible_choices = Choices(q for q in questions if q)
    if not possible_choices:
        return None
    question = possible_choices.random()
    return question
