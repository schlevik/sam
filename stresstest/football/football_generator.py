import random

import names
from loguru import logger
from scipy.stats import expon

from stresstest.classes import Choices, Config
from stresstest.football.classes import Team, Player, FootballWorld
from stresstest.generator import StoryGenerator
from stresstest.util import fmt_dict


class FootballGenerator(StoryGenerator):
    EVENT_TYPES = Choices(['goal', 'foul'])
    ATTRIBUTES = Choices(['time', 'distance', 'coactor'])

    CAUSES = Choices(['error', 'run', 'freekick'])
    EFFECTS = Choices(['penalty'])

    POSITIONS = Choices(['forward', 'defender', 'midfielder'])
    world: FootballWorld

    def __init__(self, config, get_world=FootballWorld,
                 team_names: str = 'stresstest/football/resources/team-names.json', *args, **kwargs):
        logger.debug(f"{FootballGenerator.__name__} entering constructor")
        logger.debug(fmt_dict(locals()))
        super().__init__(config, FootballWorld, *args, **kwargs)
        logger.debug(f"{team_names}")
        self.team_names = Config(team_names)
        logger.debug(f"{FootballGenerator.__name__} finish")

    def do_set_world(self):
        num_players = self.cfg.get("world.num_players", 5)
        gender = self.cfg.get("world.gender", True)
        self.world.gender = self.world.FEMALE if gender else self.world.MALE

        t1_first = self.team_names.as_choices("first").random()
        t1_second = self.team_names.as_choices("second").random()

        t2_first = (self.team_names.as_choices("first") - t1_first).random()
        t2_second = (self.team_names.as_choices("second") - t1_second).random()

        self.world.teams = (
            Team(**{
                "id": "team1",
                "name": " ".join((t1_first, t1_second))
            }),
            Team(**{
                "id": "team2",
                "name": " ".join((t2_first, t2_second))

            })
        )

        self.world.num_players = num_players
        self.world.players = []
        self.world.players_by_id = dict()
        # TODO: unique names (actually non unique would be funny too)

        for i in range(1, num_players + 1):
            p1 = Player(**{
                "id": f"player{i}",
                "first": names.get_first_name(self.world.gender),
                "last": names.get_last_name(),
                'team': self.world['teams'][0],
                "position": self.POSITIONS.random()
            })
            p2 = Player(**{
                "id": f"player{i + num_players}",
                "first": names.get_first_name(self.world.gender),
                "last": names.get_last_name(),
                'team': self.world['teams'][1],
                "position": self.POSITIONS.random()
            })
            self.world.players.extend((p1, p2))
            self.world.players_by_id[p1['id']] = p1
            self.world.players_by_id[p2['id']] = p2

    def get_actor_choices(self) -> Choices:
        return Choices(self.world['players'])

    def filter_attributes(self, choices):
        if self.current_event.event_type == 'foul':
            choices = choices - ['distance']
        return choices

    def create_attribute(self, name):
        if name == 'distance':
            return random.choice(list(range(18, 35)))

        if name == 'time':
            last_ts = 0
            for sentence in self.sentences[::-1]:
                last_ts = sentence.attributes.get("time", 0)
                if last_ts:
                    break

            rv = expon(scale=20)
            # TODO: if too slow, change to linear...
            if last_ts + 1 >= 90:
                return last_ts + 1
            p = [rv.pdf(x) for x in range(last_ts + 1, 90)]
            sum_p = sum(p)
            p_norm = [x / sum_p for x in p]
            return random.choices(list(range(last_ts + 1, 90)), weights=p_norm, k=1)[0]

        if name == 'coactor':
            if self.current_event.event_type == 'foul':
                player = Choices(
                    p['id'] for p in self.world['players'] if
                    p['team'] != self.current_event.actor['team']).random()

            elif self.current_event.event_type == 'goal':
                player = Choices(p['id'] for p in self.world['players'] if
                                 p['team'] == self.current_event.actor[
                                     'team'] and p != self.current_event.actor).random()
            else:
                raise NotImplementedError()
            return self.world['players_by_id'][player]

    def post_process_attribute_answers(self, attribute_name, attribute_value):
        if attribute_name == 'coactor':
            return " ".join((attribute_value['first'], attribute_value['last']))
        return attribute_value

    def post_process_actor_answer(self, actor):
        return " ".join((actor['first'], actor['last']))
