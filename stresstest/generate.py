from typing import Dict, List
import random
import names
from loguru import logger

from stresstest.classes import Choices, Event, World, Team, Player, Question, QuestionTypes, ReasoningTypes


class StoryGenerator:
    EVENT_TYPES = Choices(['goal', 'foul'])
    ATTRIBUTES = Choices(['time', 'distance', 'coactor'])
    # ACTORS = Choices(['player'])
    # ASSIST = Choices(['player'])
    CAUSES = Choices(['error', 'run', 'freekick'])
    EFFECTS = Choices(['penalty'])
    MODES = Choices([''])
    FEATURES = Choices(['modifier'])
    POSITIONS = Choices(['forward', 'defender', 'midfielder'])

    def __init__(self, config):
        logger.debug("cfg:")
        # TODO: change to dynaconf maybe?
        self.cfg = config
        logger.debug(self.cfg.pprint())

    def set_world(self):
        world = World()
        num_players = self.cfg.get("world.num_players", 5)
        gender = self.cfg.get("world.gender", True)
        world.gender = World.FEMALE if gender else World.MALE
        world.num_sentences = self.cfg.get("world.num_sentences", 5)

        t1_first = self.cfg.as_choices("team.name.first").random()
        t1_second = self.cfg.as_choices("team.name.second").random()

        t2_first = (self.cfg.as_choices("team.name.first") - (
            t1_first)).random()
        t2_second = (self.cfg.as_choices("team.name.second") - (
            t1_second)).random()

        world.teams = (
            Team(**{
                "id": "team1",
                "name": " ".join((t1_first, t1_second))
            }),
            Team(**{
                "id": "team2",
                "name": " ".join((t2_first, t2_second))

            })
        )

        world.num_players = num_players
        world.players = []
        world.players_by_id = dict()
        # TODO: unique names (actually non unique would be funny too)

        for i in range(1, num_players + 1):
            p1 = Player(**{
                "id": f"player{i}",
                "first": names.get_first_name(world.gender),
                "last": names.get_last_name(),
                'team': world['teams'][0],
                "position": self.POSITIONS.random()
            })
            p2 = Player(**{
                "id": f"player{i + num_players}",
                "first": names.get_first_name(world.gender),
                "last": names.get_last_name(),
                'team': world['teams'][1],
                "position": self.POSITIONS.random()
            })
            world.players.extend((p1, p2))
            world.players_by_id[p1['id']] = p1
            world.players_by_id[p2['id']] = p2
        self.world = world

        logger.info("World:")
        logger.info(self.world)

    def set_action(self):
        self.sentence.event_type = self.EVENT_TYPES.random()

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
            if self.sentence.event_type == 'foul':
                player = Choices(
                    p['id'] for p in self.world['players'] if
                    p['team'] != self.sentence.actor['team']).random()

            elif self.sentence.event_type == 'goal':
                player = Choices(p['id'] for p in self.world['players'] if
                                 p['team'] == self.sentence.actor['team'] and p != self.sentence.actor).random()
            else:
                raise NotImplementedError()
            return self.world['players_by_id'][player]

    def set_attributes(self):
        self.sentence.attributes = dict()
        choices = self.ATTRIBUTES
        if self.sentence.event_type == 'foul':
            choices = choices - ['distance']
        for attribute in choices:
            self.sentence.attributes[attribute] = self.handle_attribute(attribute)

    def set_actor(self):
        self.sentence.actor = Choices.random(self.world['players'])

    def set_modes(self):
        ...

    def set_anything_else(self):
        self.sentence.cause = self.CAUSES.random()
        # TODO: logics here
        if self.sentence.event_type != "goal":
            self.sentence.effect = self.EFFECTS.random()

    def generate_sentence(self, sentence_nr):
        self.sentence = Event(sentence_nr)
        self.set_action()
        self.set_actor()
        self.set_attributes()
        self.set_anything_else()
        self.set_modes()
        self.sentences.append(self.sentence)

    def generate_story(self) -> List[Event]:
        self.set_world()
        self.sentences: List[Event] = []
        for i in range(self.world['num_sentences']):
            self.generate_sentence(i)
        return self.sentences

    def generate_questions(self, story: List[Event],
                           visits: Dict[int, List[str]]):
        # extractive
        single_span_questions = []
        multi_span_questions = []
        unanswerable_questions = []
        abstractive_questions = []

        # per-sentence action questions
        for event_type in self.EVENT_TYPES:
            for ith, event in enumerate(s for s in story if s.event_type == event_type):
                q = Question(
                    type=QuestionTypes.DIRECT,
                    target="actor",
                    evidence=event.sentence_nr,
                    event_type=event_type,
                    # TODO: WHAT IF COREF ETC
                    answer=" ".join((event.actor['first'], event.actor['last'])),
                    reasoning=ReasoningTypes.Retrieval if ith == 0 else ReasoningTypes.OrderingEasy,
                    question_data={"n": ith + 1}
                )
                if any(f"sent.actor" in v for v in visits[event.sentence_nr]):
                    single_span_questions.append(q)
                else:
                    q.answer = None
                    unanswerable_questions.append(q)

                # attribute questions
                for attribute in self.ATTRIBUTES:
                    q = Question(
                        type=QuestionTypes.DIRECT,
                        target=attribute,
                        event_type=event_type,
                        evidence=event.sentence_nr,
                        reasoning=ReasoningTypes.Retrieval if ith == 0 else ReasoningTypes.OrderingEasy,
                        question_data={"n": ith + 1},

                    )
                    if any(f"sent.attributes.{attribute}" in v for v in visits[event.sentence_nr]):
                        if attribute == 'coactor':
                            q.answer = " ".join(
                                (event.attributes['coactor'].first, event.attributes['coactor'].last))
                        else:

                            q.answer = event.attributes[attribute]
                        single_span_questions.append(q)
                    else:
                        q.answer = None
                        unanswerable_questions.append(q)

            # overall questions

            # target = actor
            q = Question(
                type=QuestionTypes.OVERALL,
                target='actor',
                event_type=event_type,

            )
            events = sum(s.event_type == event_type for s in story)
            q.evidence = [s.sentence_nr for s in story if s.event_type == event_type]
            if events > 1:
                q.reasoning = ReasoningTypes.MultiRetrieval
                q.answer = [" ".join((s.actor['first'], s.actor['last'])) for s in story if s.event_type == event_type]
                multi_span_questions.append(q)
            elif events == 1:
                q.reasoning = ReasoningTypes.Retrieval
                q.answer = next(
                    " ".join((s.actor['first'], s.actor['last'])) for s in story if s.event_type == event_type)
                single_span_questions.append(q)
            elif events < 1:
                q.answer = None
                unanswerable_questions.append(q)

            # target = attribute
            for attribute in self.ATTRIBUTES:
                q = Question(
                    type=QuestionTypes.OVERALL,
                    target=attribute,
                    event_type=event_type

                )

                def condition(s):
                    return any(
                        f"sent.attributes.{attribute}" in v for v in
                        visits[s.sentence_nr]
                    ) and s.event_type == event_type

                events = sum(1 for s in story if condition(s))
                q.evidence = [s.sentence_nr for s in story if condition(s)]
                answers = [s.attributes[attribute] for s in story if condition(s)]
                if attribute == 'coactor':
                    answers = [" ".join((a['first'], a['last'])) for a in answers]
                if events > 1:
                    q.reasoning = ReasoningTypes.MultiRetrieval
                    q.answer = answers
                    multi_span_questions.append(q)
                elif events == 1:
                    q.reasoning = ReasoningTypes.Retrieval
                    q.answer = answers[0]
                    single_span_questions.append(q)
                elif events < 1:
                    q.answer = None
                    unanswerable_questions.append(q)

        return (single_span_questions, multi_span_questions,
                unanswerable_questions, abstractive_questions)
