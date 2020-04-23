from stresstest.classes import F
from tests.testutil import TestRealizer, only

sentences = {
    "test": [
        "%A",  # 12
        "%B $a",  # 3*(3-1) = 6
        "%B %B",
        "!C",
        "!D",
        "$a !E",  # 3*1 = 3
        "$d !F $d",  # 3*2*2 = 12
        "!G"
    ]

}
dollar = {
    "a": "1 2 3".split(),  # 3
    "b": "4 5 6".split(),  # 3
    "c": "7 8 9".split(),  # 3
    "d": "0 0 0".split()  # 3
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
    "G": (lambda ctx: ..., ['$b', '$a'])  # 12
}


def test_estimate_size_works_per_sentence():
    sents = only(sentences, 0)  # flat
    r = TestRealizer(sentences=sents, dollar=dollar, percent=percent)
    assert r.estimate_size(r.sentences['test']) == 12


def test_estimate_size_works_with_replacement():
    sents = only(sentences, 1)  # flat
    r = TestRealizer(sentences=sents, dollar=dollar, percent=percent)
    assert r.estimate_size(r.sentences['test']) == 6


def test_estimate_size_works_with_replacement_when_accessed_indirectly():
    sents = only(sentences, 2)  # flat
    r = TestRealizer(sentences=sents, dollar=dollar, percent=percent)
    assert r.estimate_size(r.sentences['test']) == 6


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
    assert r.estimate_size(r.sentences['test']) == 12


def test_bang_works_with_options_when_implicitly_defined_with_args():
    sents = only(sentences, 7)
    r = TestRealizer(sentences=sents, dollar=dollar, percent=percent, bang=bang)
    assert r.estimate_size(r.sentences['test']) == 6
