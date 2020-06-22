from stresstest.classes import Bundle
from stresstest.football.football_generator import FootballGenerator
from stresstest.football.generate_with_modifier import FootballModifierGenerator
from stresstest.football.resources import templates
from stresstest.football.resources import templates_modifier


def _get_templates_dict(module):
    return {
        name: getattr(module, name) for name in
        ['dollar', 'sentences', 'at', 'percent', 'bang', 'question_templates']
    }


bundle = Bundle(
    generator=FootballGenerator,
    templates=_get_templates_dict(templates),
    generator_modifier=FootballModifierGenerator,
    templates_modifier=_get_templates_dict(templates_modifier)
)
