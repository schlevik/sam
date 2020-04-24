from tests.testutil import TestRealizer, only

sentences = {
    "test": [
        "$a",  # 3
        "$a $a",  # 3 * (3-1) = 6
        "$a $a $a",  # 3 * (3-1) * (3-2) = 6
        "$b.a $c",  # 4 * (2-1) = 4
        "a [$a|$c] c",  # 1 * (3 + 2) * 1 = 5
        "a $a ($c b) c"  # 1 * 3 * (1+2*1) *1 = 9
    ]

}
dollar = {
    "a": [  # 1 + 1 + 1 = 3
        '1 b c',  # 1 * 1 * 1
        '2 b c',  # 1 * 1 * 1
        '3'  # 1 * 1 * 1
    ],
    "b": {
        "a": "4 5 $c".split(),  # 1 + 1 + 2 = 4
        "c": "6 7 $b.a".split()  # 1 + 1 + 4 = 6
    },
    "c": "8 9".split()  # 1 + 1 = 2
}


def test_estimate_size_works_per_sentence():
    sents = only(sentences, 0)  # flat
    r = TestRealizer(sentences=sents, dollar=dollar)
    assert r.estimate_size(r.sentences['test']) == 3
    assert r._estimate_words(r.sentences['test'][0]) == 3


def test_estimate_size_works_with_alternative():
    sents = only(sentences, 4)
    r = TestRealizer(sentences=sents, dollar=dollar)
    assert r.estimate_size(r.sentences['test']) == 5
    assert r._estimate_words(r.sentences['test'][0]) == 5


def test_estimate_size_works_with_option():
    sents = only(sentences, 5)
    r = TestRealizer(sentences=sents, dollar=dollar)
    assert r.estimate_size(r.sentences['test']) == 9
    assert r._estimate_words(r.sentences['test'][0]) == 9


def test_estimate_size_works():
    sents = sentences
    r = TestRealizer(sentences=sents, dollar=dollar)
    assert r.estimate_size(r.sentences['test']) == 61
