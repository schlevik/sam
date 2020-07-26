import random

from flaky import flaky

from stresstest.classes import Event
from stresstest.comb_utils import generate_one_possible_template_choice
from stresstest.football import bundle


@flaky(max_runs=1000, min_passes=1000)
def test_generate_one_possible_template_choice_works_for_football_bundle():
    must_haves = [random.choice(['coactor', 'time', 'distance', None]) for _ in range(5)]
    # events = (('modified', 'goal'), ('modified', 'goal'), ('just', 'goal'), ('any', '_'), ('any', '_'))
    events = [Event(i, et) for i, et in enumerate(['goal', 'goal', 'goal', 'goal', 'goal'])]
    templates = bundle.templates_modifier['sentences']
    one_possible_template_choice = \
        generate_one_possible_template_choice(events, templates, must_haves, bundle.has_template_attribute)
    for mh, (et, tc) in zip(must_haves, one_possible_template_choice):
        if mh:
            template = templates[et][tc]
            assert bundle.has_template_attribute(template, mh)
