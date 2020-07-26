from abc import abstractmethod, ABC
from typing import List, Optional, Callable
from loguru import logger

from stresstest.classes import Event, World, EventPlan
from stresstest.generator import StoryGenerator


class PlannedModifierGenerator(StoryGenerator, ABC):
    current_event: Optional[Event]
    world: World
    sentences: List[Event]

    def __init__(self, config, get_world: Callable[[], World], event_plan: EventPlan, modifier_type, *args, **kwargs):
        super().__init__(config, get_world, *args, **kwargs)
        self.event_plan = event_plan
        self.modifier_type = modifier_type

    def set_action(self):
        mod, event_type = self.event_plan.event_types[self.current_event.sentence_nr]
        logger.debug(f"mod: {mod}, event_type: {event_type}")
        if mod == EventPlan.ANY:
            super().set_action()
        elif mod == EventPlan.JUST or mod == EventPlan.MOD:
            # exactly event-type
            logger.debug(f"Setting EXACTLY {event_type}")
            if event_type == '_':
                super().set_action()
            else:
                self.current_event.event_type = event_type
        elif mod == EventPlan.NOT:
            # all but event-type
            logger.debug(f"Setting ALL BUT {event_type}")
            self.current_event.event_type = (self.EVENT_TYPES - event_type).random()

        else:
            raise NotImplementedError()

        logger.debug(f"action: {self.current_event.event_type}")

    @abstractmethod
    def create_attribute(self, name: str):
        ...

    def set_attributes(self):
        self.current_event.attributes = dict()
        attributes = self.filter_attributes(self.ATTRIBUTES)
        for attribute in attributes:
            self.current_event.attributes[attribute] = self.create_attribute(attribute)

    def set_everything_else(self):
        mod, _ = self.event_plan.event_types[self.current_event.sentence_nr]
        if mod == EventPlan.MOD:
            self.current_event.features.append(self.modifier_type)

    def generate_questions_from_plan(self, event_plan: EventPlan, events: List[Event], modified=False):
        questions = []
        question = event_plan.to_question(events, modified, self)
        self.post_process_question(question)
        if event_plan.question_target == 'actor':
            question.answer = self.post_process_actor_answer(question.answer)
        elif event_plan.question_target in self.ATTRIBUTES:
            question.answer = self.post_process_attribute_answers(event_plan.question_target, question.answer)
        else:
            raise NotImplementedError()
        questions.append(question)

        return questions, None, None, None
