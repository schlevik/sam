import random
from enum import Enum, auto

from ailog import Loggable

from stresstest.graph import Path


class QuestionType(Enum):
    Score   = auto()


class QuestionTarget(Enum):
    Overall = auto()
    Team = auto()
    Player = auto()


class Answer(Loggable):
    ...


class Question(Loggable):
    def __init__(self, question_type: QuestionType,
                 question_target: QuestionTarget, answer: Answer):
        ...


def is_applicable_type(question_type: QuestionType, path: Path) -> bool:
    return True


def is_applicable_target(question_target: QuestionTarget,
                         question_type: QuestionType,
                         path: Path) -> bool:
    return True


def instantiate():
    ...


def generate_question(path: Path) -> Question:
    # get random question type, question target
    question_type = random.sample(
        [t for t in QuestionType if
         is_applicable_type(t, path)], 1)[0]
    question_target = random.sample(
        [t for t in QuestionType if
         is_applicable_target(t, question_type, path)], 1)[0]
    # instantiate

    # generate answer
    answer = Answer()
    # return question
    return Question(question_type, question_target, answer)
