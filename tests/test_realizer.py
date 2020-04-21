from copy import deepcopy

from flaky import flaky

from stresstest.generate import Sentence
from tests.resources.templates import sentences
from tests.testutil import TestRealizer, only


@flaky(max_runs=5, min_passes=5)
def test_different_dollar_templates_flat():
    sents = only(sentences, 0)  # flat
    r = TestRealizer(sentences=sents)
    logic_sents = [Sentence(0)]
    logic_sents[0].action = 'test'
    world = {}
    realised_sents, visits = r.realise_story(logic_sents, world)
    assert '1' in realised_sents[0]
    assert '2' in realised_sents[0]
    assert '3' in realised_sents[0]


@flaky(max_runs=5, min_passes=5)
def test_different_dollar_templates_flat_with_multiple_sentences():
    sents = only(sentences, 0)  # flat
    r = TestRealizer(sentences=sents)
    logic_sents = [Sentence(0), Sentence(1)]
    logic_sents[0].action = 'test'
    logic_sents[1].action = 'test'
    world = {}
    realised_sents, visits = r.realise_story(logic_sents, world)
    assert '1' in realised_sents[0]
    assert '2' in realised_sents[0]
    assert '3' in realised_sents[0]


@flaky(max_runs=5, min_passes=5)
def test_different_dollar_templates_flat_with_multiple_sentences_when_not_leaf():
    sents = only(sentences, 2)  # flat
    r = TestRealizer(sentences=sents)
    logic_sents = [Sentence(0), Sentence(1)]
    logic_sents[0].action = 'test'
    logic_sents[1].action = 'test'
    world = {}
    realised_sents, visits = r.realise_story(logic_sents, world)
    assert '1' in realised_sents[0]
    assert '2' in realised_sents[0]
    assert 'b' in realised_sents[0]


@flaky(max_runs=5, min_passes=5)
def test_different_dollar_templates_nested():
    sents = only(sentences, 1)  # nested
    r = TestRealizer(sentences=sents)
    logic_sents = [Sentence(0)]
    logic_sents[0].action = 'test'
    world = {}
    realised_sents, visits = r.realise_story(logic_sents, world)
    assert '1' in realised_sents[0]
    assert '2' in realised_sents[0]
    assert '3' in realised_sents[0]
