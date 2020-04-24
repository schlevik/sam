from stresstest.classes import F
from tests.testutil import TestRealizer, only

sentences = {
    "test": [
        "%A",  # 12
        "%B $a",  # 3*3 = 9
        "%B %B",  # 9
        "!C",  # 2
        "!D",  # 9
        "$a !E",  # 3*1 = 3
        "$d !F $d",  # 3*2*3 = 18
        "!G"  # 12
    ]

}
dollar = {
    "a": "1 2 3".split(),  # 3
    "b": "4 5 6".split(),  # 3
    "c": "7 8 9".split(),  # 3
    "d": "0 x y".split()  # 3
}

percent = {
    "A": {  # 12
        "condition": lambda ctx: ...,
        "one": ["$a 1", "$d 2"],  # 6
        "two": ["$b"],  # 3
        "three": ['$c']  # 3
    },
    "B": {
        "condition": lambda ctx: ...,
        "one": ["$a"]  # 3
    }
}


class TestC(F):
    options = None
    number = 2

    def __call__(self, ctx: dict) -> str:
        pass


class TestD(F):
    options = ["$a", "$b", "$c"]

    def __call__(self, ctx: dict) -> str:
        pass


bang = {
    "C": TestC,
    "D": TestD,
    "E": (lambda ctx: "4"),
    "F": (lambda ctx: ..., 2),
    "G": (lambda ctx: ..., ['$b', '$a'])  # 6
}


def test_estimate_size_works_per_sentence():
    sents = only(sentences, 0)  # flat
    r = TestRealizer(sentences=sents, dollar=dollar, percent=percent)
    assert r.estimate_size(r.sentences['test']) == 12


def test_bang_works_with_explicitly_defined():
    sents = only(sentences, 3)
    r = TestRealizer(sentences=sents, dollar=dollar, percent=percent, bang=bang)
    assert r.estimate_size(r.sentences['test']) == 2


def test_bang_works_with_options_when_explicitly_defined():
    sents = only(sentences, 4)
    r = TestRealizer(sentences=sents, dollar=dollar, percent=percent, bang=bang)
    assert r.estimate_size(r.sentences['test']) == 9


def test_bang_works_with_number_when_implicitly_defined_no_args():
    sents = only(sentences, 5)
    r = TestRealizer(sentences=sents, dollar=dollar, percent=percent, bang=bang)
    assert r.estimate_size(r.sentences['test']) == 3


def test_bang_works_with_number_when_implicitly_defined_with_args():
    sents = only(sentences, 6)
    r = TestRealizer(sentences=sents, dollar=dollar, percent=percent, bang=bang)
    assert r.estimate_size(r.sentences['test']) == 18


def test_bang_works_with_options_when_implicitly_defined_with_args():
    sents = only(sentences, 7)
    r = TestRealizer(sentences=sents, dollar=dollar, percent=percent, bang=bang)
    assert r.estimate_size(r.sentences['test']) == 6


def test_full_grammar():
    r = TestRealizer(sentences=sentences, dollar=dollar, percent=percent, bang=bang)
    assert r.estimate_size(r.sentences['test']) == 68
