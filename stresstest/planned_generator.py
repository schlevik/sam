from abc import abstractmethod, ABC
from collections import defaultdict, OrderedDict
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
        self.ordered_attribute_map = defaultdict(dict)

    def _init_order_map(self, order_attr):
        logger.debug(f"Setting up order map for attribute '{order_attr}'")
        orders = sorted(
            list((o, i) for i, (m, (_, _, o)) in enumerate(self.event_plan.event_types) if m == EventPlan.ORDER)
        )
        attributes = []
        while len(attributes) < len(orders):
            new_attribute = self.create_attribute(order_attr)
            if new_attribute not in attributes:
                attributes.append(new_attribute)
        attributes = sorted(attributes)
        logger.debug(attributes)
        logger.debug(orders)
        for attribute, (_, idx) in zip(attributes, orders):
            self.ordered_attribute_map[order_attr][idx] = attribute
        logger.debug(self.event_plan.event_types)
        logger.debug(self.ordered_attribute_map[order_attr])
        # raise NotImplementedError()

    def _do_set_action(self, mod, event_type):
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
            if mod == EventPlan.MOD:
                self.current_event.features.append(self.modifier_type)
        elif mod == EventPlan.NOT:
            # all but event-type
            logger.debug(f"Setting ALL BUT {event_type}")
            self.current_event.event_type = (self.EVENT_TYPES - event_type).random()

        elif mod == EventPlan.ORDER:
            (mod, event_type), order_attr, order = event_type
            self._do_set_action(mod, event_type)
            if not self.ordered_attribute_map[order_attr]:
                self._init_order_map(order_attr)
            # raise NotImplementedError()

    def set_action(self):
        mod, event_type = self.event_plan.event_types[self.current_event.sentence_nr]
        self._do_set_action(mod, event_type)

        logger.debug(f"action: {self.current_event.event_type}")

    @abstractmethod
    def create_attribute(self, name: str):
        ...

    def set_attributes(self):
        self.current_event.attributes = dict()
        attributes = self.filter_attributes(self.ATTRIBUTES)
        for attribute in attributes:
            preset_attr = self.ordered_attribute_map[attribute].get(self.current_event.sentence_nr, None)
            if preset_attr is not None:
                logger.debug(f"Setting {attribute} to {preset_attr}")
                self.current_event.attributes[attribute] = preset_attr
            else:
                self.current_event.attributes[attribute] = self.create_attribute(attribute)

    def set_everything_else(self):
        ...

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
