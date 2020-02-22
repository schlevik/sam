from typing import Dict, List, Tuple
import random
import names
from quickconf import quickconf, format_config
from quicklog import Loggable

from stresstest.classes import Choices, Config


class Sentence(dict):
    def __init__(self, i):
        super().__init__()
        self.sentence_nr = i
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
            dict({
                "id": "team1",
                "name": " ".join((t1_first, t1_second))
            }),
            dict({
                "id": "team2",
                "name": " ".join((t2_first, t2_second))

            })]

        world['num_players'] = num_players
        world['players'] = []
        world['players_by_id'] = dict()
        # TODO: unique names
        # TODO: actually non_unique would be funny too
        for i in range(1, num_players + 1):
            # TODO: Positions maybe
            p1 = dict({
                "id": f"player{i}",
                "first": names.get_first_name(world['gender']),
                "last": names.get_last_name(),
                'team': world['teams'][0]
            })
            p2 = dict({
                "id": f"player{i + num_players}",
                "first": names.get_first_name(world['gender']),
                "last": names.get_last_name(),
                'team': world['teams'][1]
            })
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
                player = Choices(
                    p['id'] for p in self.world['players'] if
                    p['team'] != self.sentence.actor['team']).random()

            elif self.sentence.action == 'goal':
                player = Choices(p['id'] for p in self.world['players'] if
                                 p['team'] == self.sentence.actor[
                                     'team']).random()
            else:
                raise NotImplementedError()
            return self.world['players_by_id'][player]

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
        if self.sentence.action != "goal":
            self.sentence.effect = self.EFFECTS.random()

    def generate_sentence(self, sentence_nr):
        self.sentence = Sentence(sentence_nr)
        self.set_action()
        self.set_actor()
        self.set_attributes()
        self.set_anything_else()
        self.set_modes()
        self.sentences.append(self.sentence)

    def generate_story(self) -> List[Sentence]:
        self.set_world()
        self.sentences: List[Sentence] = []
        for i in range(self.world['num_sentences']):
            self.generate_sentence(i)
        return self.sentences

    def generate_questions(self, story: List[Sentence],
                           visits: Dict[int, List[str]]):
        # extractive
        single_span_questions = []
        multi_span_questions = []
        unanswerable_questions = []
        abstractive_questions = []

        # per-sentence action questions
        for action in self.ACTIONS:
            print(action)
            for ith, sent in enumerate(s for s in story if s.action == action):
                single_span_questions.append({
                    "type": "direct",
                    "target": "actor",
                    "n": ith + 1,
                    "action": action,
                    "answer": sent.actor

                })

                # attribute questions
                for attribute in self.ATTRIBUTES:
                    q = {
                        "type": "direct",
                        "target": attribute,
                        "n": ith + 1,
                        "action": f"{action}"

                    }
                    print(any(f"sent.attributes.{attribute}" in v for v in
                              visits[sent.sentence_nr]))
                    print(visits[sent.sentence_nr])
                    if any(f"sent.attributes.{attribute}" in v for v in
                           visits[sent.sentence_nr]):
                        q["answer"] = sent.attributes[attribute]
                        single_span_questions.append(q)
                    else:
                        q['answer'] = None
                        unanswerable_questions.append(q)

            # overall questions

            # target = actor
            q = {
                "type": "overall",
                "target": "actor",
                "action": action,
            }
            num_actions = sum(s.action == action for s in story)
            if num_actions > 1:
                q['answer'] = [s.actor for s in story if s.action == action]
                multi_span_questions.append(q)
            elif num_actions == 1:
                q['answer'] = next(s.actor for s in story if s.action == action)
                single_span_questions.append(q)
            elif num_actions < 1:
                q['answer'] = None
                unanswerable_questions.append(q)

            # target = attribute
            for attribute in self.ATTRIBUTES:
                q = {
                    "type": "overall",
                    "target": attribute,
                    "action": action

                }

                def condition(s):
                    return any(
                        f"sent.attributes.{attribute}" in v for v in
                        visits[s.sentence_nr]
                    ) and s.action == action

                num_actions = sum(1 for s in story if condition(s))
                answers = [s.attributes[attribute] for s in story if
                           condition(s)]
                if num_actions > 1:
                    q['answer'] = answers
                    multi_span_questions.append(q)
                elif num_actions == 1:
                    q['answer'] = answers[0]
                    single_span_questions.append(q)
                elif num_actions < 1:
                    q['answer'] = None
                    unanswerable_questions.append(q)

        return (single_span_questions, multi_span_questions,
                unanswerable_questions, abstractive_questions)
