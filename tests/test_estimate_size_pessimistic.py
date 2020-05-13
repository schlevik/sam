from stresstest.classes import F, Context
from stresstest.realize import Accessor, Processor, SizeEstimator
from tests.util import  only

sentences = {
    "test": [
        "%A",  # 20 || 4
        "!C",  # 2 || 1
        "!D !C",  # 8 * 2 = 16 || 1 * 2 = 2
        "%A $b !D"  # 20 * 2 * 8 = 320 || 4 * 2 * 2 = 16
    ]

}
dollar = {
    "a": "1".split(),  # 1
    "b": "2 3".split(),  # 2
    "c": "4 5 6".split(),  # 3
}

percent = {
    "A": {  # 20 || 4
        "condition": lambda ctx: ...,
        "one": ["$a $b $c", "$a"],  # 6+1 = 7
        "two": ["$b $c", "$c"],  # 6 + 3 = 9
        "three": ['$a', "$c d"]  # 1 + 3 = 4
    },
}


# 2 || 1
class TestC(F):
    number = 2

    def __call__(self, ctx: dict) -> str:
        pass


# 2 + 6 = 8 || 2
class TestD(F):
    options = ['$a $b', "$c $a $b"]

    def __call__(self, ctx: dict) -> str:
        pass


bang = {
    "C": TestC,
    "D": TestD,
}


def test_pessimistic_bang():
    sents = only(sentences, 1)
    a = Accessor(context=Context(), sentences=sents, dollar=dollar, percent=percent, bang=bang)
    r = SizeEstimator(Processor(a))
    assert r.estimate_size(a.sentences['test']) == 2

    a = Accessor(context=Context(), sentences=sents, dollar=dollar, percent=percent, bang=bang)
    r = SizeEstimator(Processor(a))
    assert r.estimate_size(a.sentences['test'], pessimistic=True) == 1

    sents = only(sentences, 2)
    a = Accessor(context=Context(), sentences=sents, dollar=dollar, percent=percent, bang=bang)
    r = SizeEstimator(Processor(a))
    assert r.estimate_size(a.sentences['test']) == 16

    a = Accessor(context=Context(), sentences=sents, dollar=dollar, percent=percent, bang=bang)
    r = SizeEstimator(Processor(a))
    assert r.estimate_size(a.sentences['test'], pessimistic=True) == 2


def test_pessimistic_condition():
    sents = only(sentences, 0)
    a = Accessor(context=Context(), sentences=sents, dollar=dollar, percent=percent, bang=bang)
    r = SizeEstimator(Processor(a))
    assert r.estimate_size(a.sentences['test']) == 20

    a = Accessor(context=Context(), sentences=sents, dollar=dollar, percent=percent, bang=bang)
    r = SizeEstimator(Processor(a))
    assert r.estimate_size(a.sentences['test'], pessimistic=True) == 4


def test_combined():
    a = Accessor(context=Context(), sentences=sentences, dollar=dollar, percent=percent, bang=bang)
    r = SizeEstimator(Processor(a))

    assert r.estimate_size(a.sentences['test']) == 358

    a = Accessor(context=Context(), sentences=sentences, dollar=dollar, percent=percent, bang=bang)
    r = SizeEstimator(Processor(a))

    assert r.estimate_size(a.sentences['test'], pessimistic=True) == 23
