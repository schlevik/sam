from stresstest.classes import Bundle
from stresstest.football.football_generator import FootballGenerator
from stresstest.football.generate_with_modifier import FootballModifierGenerator

from stresstest.util import do_import


def _get_templates_dict(module):
    return {
        name: do_import(name, f"{module.__name__}.{name}") for name in
        ['dollar', 'sentences', 'at', 'percent', 'bang', 'question_templates']
    }


def _reload_bundle():
    # some dynamic reimport stuff
    from stresstest.football.resources import modifier as modifier, baseline
    bundle = Bundle(
        generator=FootballGenerator,
        templates=_get_templates_dict(baseline),
        generator_modifier=FootballModifierGenerator,
        templates_modifier=_get_templates_dict(modifier)
    )
    del modifier, baseline
    return bundle


bundle = _reload_bundle()
