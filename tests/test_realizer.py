from flaky import flaky

from stresstest.classes import Event, Question
from stresstest.realize import Realizer
from tests.resources.templates import sentences, templates
from stresstest.util import only


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


def test_fix_units():
    q = Question(type='overall', target='distance', evidence=[3], event_type='goal', reasoning='argmin-distance',
                 answer=str(14),
                 question_data={}, realized='The closest goal was scored from how far away ?')
    p = [
        "Tricia Lusk almost opened the action when she nearly slotted in a 15 metres goal "
        "from Tracy Hahn 's soft clearance .",

        'The stadium went wild as Pok Formosa was withdrawn in minute 29 with her ankle in a brace following a '
        'harsh challenge from Margaretta Sabins .',

        'Further pressure on the attack resulted in Caryl Yacullo fouling Devin Mockler '
        'for an auspiciously looking free-kick chance for her opponents .',

        "14 minutes after that Wendy Miners nearly scored on the 49 th minute , all but putting in "
        "the ball from 14 metres away under the bar after she ran 11 metres and intercepted "
        "Dynamo Whalesharks goalkeeper's goal kick .",

        'Rita Sander scored the next goal for Red-blue Elephants from 18 metres to '
        'continue where they left off after Pauline Hunter played the ball into her path .',
        'The stadium went wild seeing Claudia Johnson winning the ball out wide for Red-blue '
        'Elephants and drawing a foul play from Sharon Schoolfield .'
    ]
    realizer = Realizer(**templates, validate=False)
    answer = realizer._fix_units(q, p)
    assert answer == '14 metres'


def test_fix_units_2():
    q = Question(type='direct', target='time', evidence=[3], event_type='goal', reasoning='retrieval',
                 answer=str(42), question_data={'n': 1}, realized='When did they score the 1 st goal ?')

    passage = [
        'The match started as Terra Miller scythed down Maria Forest '
        'for a promisingly looking free-kick opportunity for her opponents .',

        'On the 11 th minute a spectacular 12 metres strike from Susan White almost '
        'flying in the lower left corner past the woman between the posts for her 2 nd league '
        'goal of the season advanced the action .',

        'The stadium went wild as Lajuana Loader fouled Pearle Giebel on the 35 th minute .',

        "7 minutes after that Trish Oieda scored in minute 42 , hitting the ball from 14 "
        "metres away off the post and in the middle of the goal after she intercepted "
        "FC Monkeys goalkeeper's goal kick .",

        'Things proceeded with Marlene Croom winning the ball in the attacking '
        'third and drawing a foul play from Mellisa Winnett .',

        "Dynamo Whalesharks advanced the action with a 12 metres goal as "
        "Silvana Waugaman put in Tabetha Bowe 's risky through ball ."]
    realizer = Realizer(**templates, validate=False)
    answer = realizer._fix_units(q, passage)
    assert answer == 'minute 42'


def test_fix_units_3():
    q = Question(type='direct', target='distance', evidence=[1, 2], event_type='goal', reasoning='bridge',
                        answer=str(31),
                        question_data=dict(), realized='After the foul on Mary Millwood , '
                                                       'from how far away was the next goal scored ?')
    passage = [
        "Ethelyn Capello scored Arctic Monkeys 's first goal from 20 metres away "
        "to set the tone for the match after Annmarie Dibiase inadvertently prodded the ball into her path .",
        'Things proceeded with Mary Millwood being withdrawn in the 31 st minute with her hip '
        'in a brace following a challenge from Amanda Testa .',
        "Pale Lilac Elephants almost advanced the action with a 31 metres goal as Carol Nehls "
        "all but curled in Cynthia Kittredge 's soft clearance .",
        "Shannon Garber almost added more insult to the injury when she almost slotted in a "
        "21 metres goal from Virginia Sheekey 's pass .",
        "In the 52 nd minute a soft clearance went to Pale Lilac Elephants 's Ida Webb on the flank "
        "and the player swept low to the 6-yard-area for Mamie Swart to poke past the goalkeeper "
        "for a wonderful 25 metres goal .",
        'Things proceeded with Judith Odougherty winning the ball in the middle field for '
        'Pale Lilac Elephants and drawing a foul from Brenda Uttech .'
    ]
    realizer = Realizer(**templates, validate=False)
    answer = realizer._fix_units(q, passage)
    assert answer == '31 metres'