from abc import abstractmethod, ABC
from collections import defaultdict, OrderedDict
from typing import List, Optional, Callable
from loguru import logger
from overrides import overrides

from stresstest.classes import Event, World, EventPlan, YouIdiotException, Choices
from stresstest.generator import StoryGenerator


class PlannedModifierGenerator(StoryGenerator, ABC):
    current_event: Optional[Event]
    world: World
    sentences: List[Event]

    def __init__(self, config, get_world: Callable[[], World], event_plan: EventPlan, modifier_types, *args, **kwargs):
        super().__init__(config, get_world, *args, **kwargs)
        self.same_actors_map = defaultdict(lambda: None)
        self.event_plan = event_plan
        self.modifier_type: Choices[str] = Choices(modifier_types).random()
        self.ordered_attribute_map = defaultdict(dict)

    def _init_order_map(self, order_attr):
        logger.debug(f"Setting up order map for attribute '{order_attr}'")
        # technically this should be recursive?
        orders = sorted(
            list((args[2], i) for i, (m, args) in enumerate(self.event_plan.event_types) if m == EventPlan.Order)
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
        if mod == EventPlan.Any:
            super().set_action()
        elif mod == EventPlan.Just or mod == EventPlan.Mod:
            # exactly event-type
            logger.debug(f"Setting EXACTLY {event_type}")
            if event_type == '_':
                super().set_action()
            else:
                self.current_event.event_type = event_type
            if mod == EventPlan.Mod:
                self.current_event.features.append(self.modifier_type)
        elif mod == EventPlan.Not:
            # all but event-type
            logger.debug(f"Setting ALL BUT {event_type}")
            self.current_event.event_type = (self.EVENT_TYPES - event_type).random()

        elif mod == EventPlan.Order:
            (mod, event_type), order_attr, order = event_type
            self._do_set_action(mod, event_type)
            if not self.ordered_attribute_map[order_attr]:
                self._init_order_map(order_attr)
            # raise NotImplementedError()
        elif mod == EventPlan.SameActor:
            (mod, event_type), same_actor_idx = event_type
            self._do_set_action(mod, event_type)
            self.same_actors_map[self.current_event.sentence_nr] = same_actor_idx

    def set_action(self):
        mod, event_type = self.event_plan.event_types[self.current_event.sentence_nr]
        self._do_set_action(mod, event_type)

        logger.debug(f"action: {self.current_event.event_type}")

    @abstractmethod
    def create_attribute(self, name: str):
        ...

    @overrides
    def set_actor(self):
        same_actor_idx = self.same_actors_map[self.current_event.sentence_nr]
        if same_actor_idx is not None and same_actor_idx != self.current_event.sentence_nr:
            self.current_event.actor = self.sentences[same_actor_idx].actor
        else:
            super_cls = super()
            super_cls.set_actor()

    def set_attributes(self):
        self.current_event.attributes = dict()
        attributes = self.filter_attributes(self.ATTRIBUTES)
        for attribute in attributes:
            preset_attr = self.ordered_attribute_map[attribute].get(self.current_event.sentence_nr, None)
            if preset_attr is not None:
                logger.debug(f"Setting {attribute} to {preset_attr}")
                self.current_event.attributes[attribute] = preset_attr
            else:
                if not self.unique_actors:
                    self.current_event.attributes[attribute] = self.create_attribute(attribute)
                else:
                    attr = self.create_attribute(attribute)
                    patience = 100

                    while any(e.attributes[attribute] == attr for e in self.sentences if attribute in e.attributes):
                        # unique attributes
                        attr = self.create_attribute(attribute)
                        patience -= 1
                        if not patience:
                            raise YouIdiotException("You tried to generate 100 'random' "
                                                    "attributes but they were all the same!")
                    self.current_event.attributes[attribute] = attr

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
        assert question.answer, f"{question}, {event_plan}, modified={modified}"
        questions.append(question)

        return questions, None, None, None
