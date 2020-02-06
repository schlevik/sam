from typing import Dict, List, Tuple
import random
import names
from quickconf import quickconf, format_config
from quicklog import Loggable

from stresstest.classes import Choices, Config


class Sentence:
    def __init__(self):
        self.action: str = ""
        self.attributes: Dict[str, str] = dict()
        self.actor: Dict[str, str] = dict()
        self.cause: str = ""
        self.effect: str = ""
        self.modes: List[Tuple[str, str]] = []
        self.features: List[str] = []

    def __repr__(self):
        return (f"{self.action} by {self.actor['id']}: with {self.attributes}, "
                f"caused by {self.cause} resulting in {self.effect}. "
                f"Modes: {self.modes}, Features: {self.features}")


class StoryGenerator(Loggable):
    ACTIONS = Choices(['goal', 'foul'])
    ATTRIBUTES = Choices(['time', 'distance', 'coactor'])
    # ACTORS = Choices(['player'])
    # ASSIST = Choices(['player'])
    CAUSES = Choices(['error', 'run', 'freekick'])
    EFFECTS = Choices(['penalty'])
    MODES = Choices([''])
    FEATURES = Choices(['modifier'])

    def __init__(self, config: Config):
        self.logger.debug("cfg:")
        self.cfg = config
        self.logger.debug(self.cfg.pprint())

    def set_world(self):
        world = dict()
        num_players = self.cfg.get("world.num_players", 5)
        gender = self.cfg.get("world.gender", True)
        world['gender'] = "female" if gender else "male"
        world['num_sentences'] = self.cfg.get("world.num_sentences", 5)

        t1_first = self.cfg.as_choices("team.name.first").random()
        t1_second = self.cfg.as_choices("team.name.second").random()

        t2_first = (self.cfg.as_choices("team.name.first") - (
            t1_first)).random()
        t2_second = (self.cfg.as_choices("team.name.second") - (
            t1_second)).random()

        world['teams'] = [
            {
                "id": "team1",
                "name": " ".join((t1_first, t1_second))
            },
            {
                "id": "team2",
                "name": " ".join((t2_first, t2_second))

            }]

        world['num_players'] = num_players
        world['players'] = []
        world['players_by_id'] = dict()
        # TODO: unique names
        # TODO: actually non_unique would be funny too
        for i in range(1, num_players + 1):
            # TODO: Positions maybe
            p1 = {
                "id": f"player{i}",
                "first": names.get_first_name(world['gender']),
                "last": names.get_last_name(),
                'team': f"team1"
            }
            p2 = {
                "id": f"player{i + num_players}",
                "first": names.get_first_name(world['gender']),
                "last": names.get_last_name(),
                'team': f"team2"
            }
            world['players'].extend((p1, p2))
            world['players_by_id'][p1['id']] = p1
            world['players_by_id'][p2['id']] = p2
        self.world = world
        self.logger.info("World:")
        self.logger.info(self.world)

    def set_action(self):
        self.sentence.action = self.ACTIONS.random()

    def handle_attribute(self, name):
        if name == 'distance':
            return random.choice(list(range(18, 35)))
        if name == 'time':
            last_ts = 0
            for sentence in self.sentences[::-1]:
                last_ts = sentence.attributes.get("time", 0)
                if last_ts:
                    break
            # TODO: this needs to be more biased towards earlier times
            return random.choice(list(range(last_ts, 90)))
        if name == 'coactor':
            if self.sentence.action == 'foul':
                return Choices(p['id'] for p in self.world['players'] if
                               p['team'] != self.sentence.actor[
                                   'team']).random()
            elif self.sentence.actor == 'goal':
                return Choices(p['id'] for p in self.world['players'] if
                               p['team'] == self.sentence.actor[
                                   'team']).random()

    def set_attributes(self):

        choices = self.ATTRIBUTES
        if self.sentence.action == 'foul':
            choices = choices - ['distance']
        for attribute in choices:
            self.sentence.attributes[attribute] = self.handle_attribute(
                attribute)

    def set_actor(self):
        self.sentence.actor = Choices.random(self.world['players'])

    def set_modes(self):
        ...

    def set_anything_else(self):
        self.sentence.cause = self.CAUSES.random()
        # TODO: logics here
        self.sentence.effect = self.EFFECTS.random()

    def generate_sentence(self):
        self.sentence = Sentence()
        self.set_action()
        self.set_actor()
        self.set_attributes()
        self.set_anything_else()
        self.set_modes()
        self.sentences.append(self.sentence)

    def generate_story(self) -> List[Sentence]:
        self.set_world()
        self.sentences: List[Sentence] = []
        for _ in range(self.world['num_sentences']):
            self.generate_sentence()
        return self.sentences
