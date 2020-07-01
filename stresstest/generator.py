from abc import abstractmethod, ABC
from typing import Dict, List, Tuple, Optional, Callable, Any
from loguru import logger

from stresstest.classes import Choices, Event, World, Question, QuestionTypes, ReasoningTypes, Config, YouIdiotException
from stresstest.util import fmt_dict


class StoryGenerator(ABC):
    current_event: Optional[Event]
    world: World
    sentences: List[Event]

    @property
    @abstractmethod
    def EVENT_TYPES(self) -> Choices:
        ...

    @property
    @abstractmethod
    def ATTRIBUTES(self) -> Choices:
        ...

    def __init__(self, config, get_world: Callable[[], World], *args, **kwargs):

        logger.debug(f"{StoryGenerator.__name__} entering constructor")
        logger.debug(fmt_dict(locals()))
        self.cfg = config
        self.get_world = get_world
        self.unique_actors = config.get('unique_actors', False)
        self.chosen_actors = []

        logger.debug("cfg:")
        logger.debug(fmt_dict(self.cfg))
        logger.debug(f"{StoryGenerator.__name__} finish constructor")

    def set_world(self):
        self.world = self.get_world()
        self.world.num_sentences = self.cfg.get("world.num_sentences", 5)

        self.do_set_world()

    def set_action(self):
        self.current_event.event_type = self.EVENT_TYPES.random()

    @abstractmethod
    def create_attribute(self, name: str):
        ...

    def set_attributes(self):
        self.current_event.attributes = dict()
        attributes = self.filter_attributes(self.ATTRIBUTES)
        for attribute in attributes:
            self.current_event.attributes[attribute] = self.create_attribute(attribute)

    @abstractmethod
    def get_actor_choices(self) -> Choices:
        ...

    def set_actor(self):
        if self.unique_actors:
            remaining_pool = self.get_actor_choices() - self.chosen_actors
            logger.debug(f"Chosen Actors: {[a.id for a in self.chosen_actors]}")
            logger.debug(f"Remaining pool: {[a.id for a in remaining_pool]}")
            self.current_event.actor = remaining_pool.random()
            self.chosen_actors.append(self.current_event.actor)
        else:
            self.current_event.actor = self.get_actor_choices().random()

    def set_everything_else(self):
        ...

    def generate_sentence(self, sentence_nr):
        logger.debug(f"Generating Sentence #{sentence_nr}")
        self.current_event = Event(sentence_nr)
        logger.debug("Setting Action...")
        self.set_action()
        logger.debug(f"Action: {self.current_event.event_type}")
        logger.debug("Setting Actor...")
        self.set_actor()
        logger.debug(f"Actor: {self.current_event.actor}")
        logger.debug("Setting Attributes...")
        self.set_attributes()
        logger.debug(f"Attributes: {self.current_event.attributes}")
        logger.debug("Setting Anything Else...")
        self.set_everything_else()
        logger.debug(f"Event so far: {self.current_event}")
        logger.debug("Done!")
        self.sentences.append(self.current_event)

    def generate_story(self) -> List[Event]:
        self.set_world()
        self.sentences: List[Event] = []
        for i in range(self.world['num_sentences']):
            self.generate_sentence(i)
        return self.sentences

    @abstractmethod
    def do_set_world(self):
        pass

    @abstractmethod
    def filter_attributes(self, attributes: Choices[Any]) -> List[Any]:
        """
        Filters attributes choices when generating an event.

        Implement a filter or leave empty to have no filter.
        Args:
            attributes: Attributes to filter

        Returns:
            List of filtered attributes

        """
        pass

    def is_realised(self, attribute: str, event: Event):
        assert getattr(self, 'visits', False), "Visits not set..."
        return any(f"sent.attributes.{attribute}" in v for v in self.visits[event.sentence_nr])

    def generate_questions(self, events: List[Event],
                           visits: Dict[int, List[str]]) -> Tuple[
        List[Question], List[Question], List[Question], List[Question]]:
        # extractive
        single_span_questions = []
        multi_span_questions = []
        unanswerable_questions = []
        abstractive_questions = []
        self.all_events = events
        self.event_type = None
        # per sentence per attribute
        self.visits = visits
        # per-sentence action questions
        for self.event_type in self.EVENT_TYPES:
            self.relevant_events = self.get_relevant_events()
            for ith, event in enumerate(self.relevant_events):
                # actor
                q = Question(
                    type=QuestionTypes.DIRECT,
                    target="actor",
                    evidence=[event.sentence_nr],
                    event_type=self.event_type,
                    # TODO: WHAT IF COREF ETC
                    answer=self.post_process_actor_answer(event.actor),
                    reasoning=ReasoningTypes.Retrieval if ith == 0 else ReasoningTypes.OrderingEasy,
                    question_data={"n": ith + 1}
                )
                self.post_process_question(q)

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
                        event_type=self.event_type,
                        reasoning=ReasoningTypes.Retrieval if ith == 0 else ReasoningTypes.OrderingEasy,
                        question_data={"n": ith + 1},

                    )
                    if self.is_realised(attribute, event):
                        q.answer = self.post_process_attribute_answers(attribute, event.attributes[attribute])
                        q.evidence = [event.sentence_nr]
                        single_span_questions.append(q)
                    else:
                        q.answer = None
                        q.evidence = []
                        unanswerable_questions.append(q)

                    self.post_process_question(q)
            # overall questions

            # target = actor
            q = Question(
                type=QuestionTypes.OVERALL,
                target='actor',
                event_type=self.event_type,

            )
            # events = self.get_relevant_events(event_type, story)  # sum(s.event_type == event_type for s in story)
            # [s.sentence_nr for s in story if s.event_type == event_type]
            q.evidence = [e.sentence_nr for e in self.relevant_events]

            if len(self.relevant_events) > 1:
                q.reasoning = ReasoningTypes.MultiRetrieval
                q.answer = [self.post_process_actor_answer(s.actor) for s in self.relevant_events]
                multi_span_questions.append(q)
            elif len(self.relevant_events) == 1:
                q.reasoning = ReasoningTypes.Retrieval
                q.answer = self.post_process_actor_answer(self.relevant_events[0].actor)
                single_span_questions.append(q)
            elif len(self.relevant_events) < 1:
                q.answer = None
                unanswerable_questions.append(q)

            self.post_process_question(q)
            # target = attribute
            for attribute in self.ATTRIBUTES:
                q = Question(
                    type=QuestionTypes.OVERALL,
                    target=attribute,
                    event_type=self.event_type

                )

                # def condition(s):
                #     return any(f"sent.attributes.{attribute}" in v for v in visits[s.sentence_nr]) and \
                #            s.event_type == event_type

                # events = sum(1 for s in story if condition(s))
                visited_events = [event for event in self.relevant_events if self.is_realised(attribute, event)]
                # q.evidence = [e.sentence_nr for s in story if condition(s)]
                q.evidence = [e.sentence_nr for e in visited_events]
                answers = [self.post_process_attribute_answers(attribute, event.attributes[attribute]) for event in
                           visited_events]

                if len(visited_events) > 1:
                    q.reasoning = ReasoningTypes.MultiRetrieval
                    q.answer = answers
                    multi_span_questions.append(q)

                elif len(visited_events) == 1:
                    q.reasoning = ReasoningTypes.Retrieval
                    q.answer = answers[0]
                    single_span_questions.append(q)

                elif len(visited_events) < 1:
                    q.answer = None
                    unanswerable_questions.append(q)
                self.post_process_question(q)
        return (single_span_questions, multi_span_questions,
                unanswerable_questions, abstractive_questions)

    @abstractmethod
    def post_process_actor_answer(self, actor):
        pass

    @abstractmethod
    def post_process_attribute_answers(self, attribute: str, answer: Any) -> str:
        ...

    def get_relevant_events(self) -> List[Event]:
        return [s for s in self.all_events if s.event_type == self.event_type]

    def post_process_question(self, q: Question):
        pass
