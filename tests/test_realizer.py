from flaky import flaky

from stresstest.classes import Event
from stresstest.realize import Realizer
from tests.resources.templates import sentences, templates
from tests.util import only


@flaky(max_runs=5, min_passes=5)
def test_different_dollar_templates_flat():
    # sents = only(sentences, 0)  # flat
    only_templates = only(templates, 0)
    r = Realizer(**only_templates)
    # r = TestRealizer(sentences=sents)
    logic_sents = [Event(0)]
    logic_sents[0].event_type = 'test'
    world = {}
    realised_sents, visits = r.realise_story(logic_sents, world)
    assert '1' in realised_sents[0]
    assert '2' in realised_sents[0]
    assert '3' in realised_sents[0]


@flaky(max_runs=5, min_passes=5)
def test_different_dollar_templates_flat_with_multiple_sentences():
    only_templates = only(templates, 0)
    r = Realizer(unique_sentences=False, **only_templates)
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
    only_templates = only(templates, 2)
    r = Realizer(**only_templates, unique_sentences=False)
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
    only_templates = only(templates, 1)
    r = Realizer(**only_templates, unique_sentences=False)
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
    r = Realizer(**templates)
    ii = [0, 1, 2, 3]
    logic_sents = [Event(i) for i in ii]
    for i in ii:
        logic_sents[i].event_type = "unique_sentence_test"
    world = {}
    realised_sents, visits = r.realise_story(logic_sents, world)
    for i in ii:
        assert any([str(i) in sent for sent in realised_sents]), f"{i} not in story!"
