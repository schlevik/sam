import random

from ailog import Loggable

from stresstest.graph import Path
from stresstest.stringify import Stringifier
from stresstest.util import alphnum


class Question(Loggable):
    def __init__(self, question_type, answer_type):
        self.question_type = question_type
        self.answer_type = answer_type

    def stringify(self, stringifier: Stringifier):
        return stringifier.to_string_question(self)

    def __repr__(self):
        return f"{self.question_type} {self.answer_type}"


def generate_question(path: Path, question_conf) -> Question:
    alphnum_path = [alphnum(p) for p in path.steps]
    # get random question type, question target
    question_type = random.choice(
        [alphnum(e) for e in question_conf['question-type'].keys() if
         alphnum(e) in alphnum_path])
    answer_type = random.choice(
        [alphnum(e) for e in question_conf['answer-type'].keys() if
         alphnum(e) in alphnum_path])
    q = Question(question_type, answer_type)
    # generate answer
    # return question
    return q


def generate_answer(path: Path, question: Question):
    return path.stringified[question.answer_type]
