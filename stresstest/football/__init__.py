from typing import List, Union

from loguru import logger

from stresstest.classes import Bundle
from stresstest.football.classes import FootballWorld
from stresstest.football.football_generator import FootballGenerator
from stresstest.football.generate_with_modifier import FootballModifierGenerator, PlannedFootballModifierGenerator

from stresstest.util import do_import


def _get_templates_dict(module):
    return {
        name: do_import(name, f"{module.__name__}.{name}") for name in
        ['dollar', 'sentences', 'at', 'percent', 'bang', 'question_templates']
    }


def _has(template: str, attribute: Union[str, List[str]]):
    # hacky sack ~
    if not isinstance(template, str):
        logger.debug(type(template))
        raise NotImplementedError()
    if not isinstance(attribute, str):
        return all(_has(template, a) for a in attribute)
    if not attribute:
        return True
    if attribute == 'time':
        return '.time ' in template
    if attribute == 'coactor':
        return '.coactor ' in template or 'S.goal-cause ' in template or 'PP.goal-cause-coref ' in template
    if attribute == 'distance':
        return '.distance ' in template
    else:
        raise NotImplementedError()


def _reload_bundle() -> Bundle:
    # some dynamic reimport stuff
    from stresstest.football.resources import modifier as modifier
    bundle = Bundle(
        generator=FootballGenerator,
        world=FootballWorld,
        # templates=_get_templates_dict(baseline),
        generator_modifier=FootballModifierGenerator,
        planned_generator=PlannedFootballModifierGenerator,
        templates_modifier=_get_templates_dict(modifier),
        reasoning_map={
            'retrieval': ['time', 'distance', 'coactor'],
            'retrieval-reverse': ['time', 'distance', 'coactor'],
            'retrieval-two': ['time', 'distance', 'coactor'],
            'retrieval-two-reverse': ['time', 'distance', 'coactor'],
            'bridge': ['time', 'distance', 'coactor'],
            'bridge-reverse': ['time', 'distance', 'coactor']
        },
        has_template_attribute=_has
    )
    del modifier,  # baseline
    return bundle


bundle: Bundle = _reload_bundle()
