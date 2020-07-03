from stresstest.classes import Event, World
from stresstest.comb_utils import generate_all_possible_template_choices, calculate_num_of_permutations
from stresstest.realize import Realizer

sentences = {
    "t1": [
        "$a $a $a",
        '$b.c $b.c $b.c',
        '$c'
    ],
    "t2": [
        "0",
        "1",
        "2",
        "3",
    ]
}

dollar = {
    "a": "a1 a2 a3".split(),  # flat
    'b': {  # nested
        "c": "bc1 bc2 bc3".split()
    },
    'c': ['$d cb $d'],
    'd': ['d1', 'd2']
}

test_events = [Event(event_type='t2') for _ in range(3)] + [Event(event_type='t1')] * 2


def test_num_permutations_works():
    num_permutations = calculate_num_of_permutations(test_events, sentences, ['t1'])
    assert num_permutations == 6
    num_permutations = calculate_num_of_permutations(test_events, sentences, ['t2'])
    assert num_permutations == 24
    num_permutations = calculate_num_of_permutations(test_events, sentences, ['t1', 't2'])
    assert num_permutations == 144


def test_generate_choices_works():
    for to_permute in (['t1'], ['t2'], ['t1', 't2']):
        choices = generate_all_possible_template_choices(test_events, sentences, to_permute)
        num_permutations = calculate_num_of_permutations(test_events, sentences, to_permute)
        assert len(choices) == len(set(tuple(t) for t in choices))  # convert to tuple because unhashable
        assert len(choices) == num_permutations


def test_realise_choices_works():
    for to_permute in (['t1'], ['t2'], ['t1', 't2']):
        choices = generate_all_possible_template_choices(test_events, sentences, to_permute)
        sf = []
        w = World()
        for c in choices:
            r = Realizer(sentences, {}, dollar, {}, {}, {}, True, False)
            story, visits = r.realise_with_sentence_choices(test_events, w, c)
            sf.append(" ".join(story))
        assert len(sf) == len(set(sf))
