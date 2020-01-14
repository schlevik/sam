import logging
from ailog import Loggable
from typing import List, Callable, Optional, Set
from stresstest.classes import Path, Choices


class Question(Loggable):
    def __init__(self, target, action, answer_position, path):
        self.target = target
        self.action = action
        self.answer_position = answer_position
        self.path = path

    def __repr__(self):
        return f"{self.target} {self.action}: {self.path[self.answer_position], (self.answer_position)}"


def create_question(target, action, path, conditions) -> Optional[Question]:
    candidates = [i for i, _ in enumerate(path)]
    for c in conditions:
        candidates = c(target, action, path, candidates)
    if not candidates:
        return None
    if len(candidates) > 1:
        logging.getLogger(__name__).info(
            f"{target}, {action} is ambiguous under "
            f"{[f.__name__ for f in conditions]}: "
            f"{[path[c] for c in candidates]}")
        return None
    else:
        q = Question(target, action, candidates[0], path)
        return q


def generate_question(path: Path,
                      targets: List[str],
                      actions: List[str],
                      conditions: List[Callable[[str, str, Path], List[int]]],
                      dependent_conditions: List[
                          Callable[[str, str, Path, Choices], Set[int]]]
                      ) -> Optional[Question]:
    logger = logging.getLogger("generate_question")
    logger.info(f"Generating question from: {path}")
    questions = (create_question(t, a, path, conditions) for t in targets for a
                 in actions)
    possible_choices = Choices(q for q in questions if q)
    if not possible_choices:
        return None
    print(possible_choices.choices)
    question = possible_choices.random()
    print(question)
    return question
