from flaky import flaky

from stresstest.classes import Event
from tests.resources.templates import sentences
from tests.util import TestRealizer, only


@flaky(max_runs=5, min_passes=5)
def test_different_dollar_templates_flat():
    sents = only(sentences, 0)  # flat
    r = TestRealizer(sentences=sents)
    logic_sents = [Event(0)]
    logic_sents[0].event_type = 'test'
    world = {}
    realised_sents, visits = r.realise_story(logic_sents, world)
    assert '1' in realised_sents[0]
    assert '2' in realised_sents[0]
    assert '3' in realised_sents[0]


@flaky(max_runs=5, min_passes=5)
def test_different_dollar_templates_flat_with_multiple_sentences():
    sents = only(sentences, 0)  # flat
    r = TestRealizer(sentences=sents, unique_sentences=False)
    logic_sents = [Event(0), Event(1)]
    logic_sents[0].event_type = 'test'
    logic_sents[1].event_type = 'test'
    world = {}
    realised_sents, visits = r.realise_story(logic_sents, world)
    assert '1' in realised_sents[0]
    assert '2' in realised_sents[0]
    assert '3' in realised_sents[0]


@flaky(max_runs=5, min_passes=5)
def test_different_dollar_templates_flat_with_multiple_sentences_when_not_leaf():
    sents = only(sentences, 2)  # flat
    r = TestRealizer(sentences=sents, unique_sentences=False)
    logic_sents = [Event(0), Event(1)]
    logic_sents[0].event_type = 'test'
    logic_sents[1].event_type = 'test'
    world = {}
    realised_sents, visits = r.realise_story(logic_sents, world)
    assert '1' in realised_sents[0]
    assert '2' in realised_sents[0]
    assert 'b' in realised_sents[0]


@flaky(max_runs=5, min_passes=5)
def test_different_dollar_templates_nested():
    sents = only(sentences, 1)  # nested
    r = TestRealizer(sentences=sents)
    logic_sents = [Event(0)]
    logic_sents[0].event_type = 'test'
    world = {}
    realised_sents, visits = r.realise_story(logic_sents, world)
    assert '1' in realised_sents[0]
    assert '2' in realised_sents[0]
    assert '3' in realised_sents[0]


@flaky(max_runs=5, min_passes=5)
def test_different_sentences():
    sents = sentences  # nested
    r = TestRealizer(sentences=sents)
    ii = [0, 1, 2, 3]
    logic_sents = [Event(i) for i in ii]
    for i in ii:
        logic_sents[i].event_type = "unique_sentence_test"
    world = {}
    realised_sents, visits = r.realise_story(logic_sents, world)
    for i in ii:
        assert any(str(i) in sent for sent in realised_sents)
