from loguru import logger
from stresstest.football import FootballGenerator
from stresstest.football.classes import FootballWorld
from stresstest.generator_modifier import ModifierGenerator
from stresstest.planned_generator import PlannedModifierGenerator


class FootballModifierGenerator(FootballGenerator, ModifierGenerator):

    def __init__(self, config, get_world=FootballWorld,
                 team_names: str = 'stresstest/football/resources/team-names.json',
                 first_modification=0,
                 fill_with_modification=None,
                 modify_event_types=None,
                 modification_distance=1, total_modifiable_actions=2, modifier_type=None):
        """


        Args:
            config: The config.
            first_modification: Which event to modify first.
            modification_distance: How many events away the new first event is from the modified one.
            fill_with_modification:
                True: Fill with modified events; False: Don't fill with modified events: None: decide randomly
            modify_event_types: Which event types to modify
            total_modifiable_actions: How many actions to modify
        """
        logger.debug(f"{FootballModifierGenerator.__name__} entering constructor")
        modify_event_types = modify_event_types or ['goal']

        super().__init__(config=config, get_world=get_world, team_names=team_names,
                         first_modification=first_modification, modifier_type=modifier_type,
                         fill_with_modification=fill_with_modification, modify_event_types=modify_event_types,
                         modification_distance=modification_distance, total_modifiable_actions=total_modifiable_actions,
                         )

        logger.debug(f"{FootballModifierGenerator.__name__} finish constructor")


class PlannedFootballModifierGenerator(FootballGenerator, PlannedModifierGenerator):


    def __init__(
            self,
            config,
            event_plan,
            modifier_types,
            get_world=FootballWorld,
            team_names: str = 'stresstest/football/resources/team-names.json'
    ):
        super().__init__(config=config, get_world=get_world, team_names=team_names, event_plan=event_plan,
                         modifier_types=modifier_types)
