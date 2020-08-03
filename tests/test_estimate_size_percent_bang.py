from stresstest.classes import F, Context
from stresstest.realize import SizeEstimator, Processor, Accessor, RandomChooser
from stresstest.util import only

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

templates = {
    'dollar': dollar, 'sentences': sentences, 'at': {}, 'percent': percent, 'bang': bang, 'question_templates': {}
}




def test_estimate_size_works_per_sentence():
    only_templates = only(templates, 0)
    a = Accessor(context=Context(), **only_templates)
    r = SizeEstimator(Processor(a, RandomChooser()))
    assert r.estimate_size(a.sentences['test']) == 12


def test_bang_works_with_explicitly_defined():
    only_templates = only(templates, 3)
    a = Accessor(context=Context(), **only_templates)
    r = SizeEstimator(Processor(a, RandomChooser()))
    assert r.estimate_size(a.sentences['test']) == 2


def test_bang_works_with_options_when_explicitly_defined():
    only_templates = only(templates, 4)
    a = Accessor(context=Context(), **only_templates)
    r = SizeEstimator(Processor(a, RandomChooser()))
    assert r.estimate_size(a.sentences['test']) == 9


def test_bang_works_with_number_when_implicitly_defined_no_args():
    only_templates = only(templates, 5)
    a = Accessor(context=Context(), **only_templates)
    r = SizeEstimator(Processor(a, RandomChooser()))
    assert r.estimate_size(a.sentences['test']) == 3


def test_bang_works_with_number_when_implicitly_defined_with_args():
    only_templates = only(templates, 6)
    a = Accessor(context=Context(), **only_templates)
    r = SizeEstimator(Processor(a, RandomChooser()))
    assert r.estimate_size(a.sentences['test']) == 18


def test_bang_works_with_options_when_implicitly_defined_with_args():
    only_templates = only(templates, 7)
    a = Accessor(context=Context(), **only_templates)
    r = SizeEstimator(Processor(a, RandomChooser()))
    assert r.estimate_size(a.sentences['test']) == 6


def test_full_grammar():
    a = Accessor(context=Context(), **templates)
    r = SizeEstimator(Processor(a, RandomChooser()))
    assert r.estimate_size(a.sentences['test']) == 68
