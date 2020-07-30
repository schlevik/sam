import string
from typing import List
import names

from stresstest.classes import Question, Event
from stresstest.football.classes import Player

MASK = '🤪'

_names = None

def mask(passage: str, keep=None):
    keep = keep or set()
    return " ".join(t if t in keep else MASK for t in passage.split(" "))


def mask_question(question: Question, keep_question_mark=False):
    keep = {"?"} if keep_question_mark else None
    return mask(question.realized, keep)


def mask_passage(question: Question, passage: List[str], events: List[Event] = None, keep_answer_types=False,
                 keep_punctuation=False):
    if keep_answer_types:
        assert events
        if any(d in question.answer for d in string.digits):
            keep = set(str(a) for e in events for a in e.attributes.values() if isinstance(a, int))
        else:
            keep_f = [a.first for e in events for a in list(e.attributes.values()) + [e.actor] if isinstance(a, Player)]
            keep_l = [a.last for e in events for a in list(e.attributes.values()) + [e.actor] if isinstance(a, Player)]
            keep = set(keep_f + keep_l)
    else:
        keep = set()
    if keep_punctuation:
        keep = keep.union({d for d in string.punctuation})
    return [mask(p, keep) for p in passage]
