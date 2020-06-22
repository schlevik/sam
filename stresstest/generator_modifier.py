import random
from abc import ABC
from typing import List, Dict

from loguru import logger
from overrides import overrides

from stresstest.football import FootballGenerator
from stresstest.classes import Event, QuestionTypes, ReasoningTypes, Question, Choices, YouIdiotException
from stresstest.generator import StoryGenerator
from stresstest.util import fmt_dict


class ModifierGenerator(StoryGenerator, ABC):
    def __init__(self, config, get_world, first_modification=0, fill_with_modification=None, modify_event_types=None,
                 modification_distance=1, total_modifiable_actions=2):
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
        logger.debug(f"{ModifierGenerator.__name__} entering constructor")
        super().__init__(config, get_world)
        self.first_modification = first_modification
        self.fill_with_modifications = fill_with_modification
        assert modify_event_types
        self.modify_event_type = modify_event_types
        self.modification_distance = modification_distance
        self.total_modifiable_actions = 2
        self.current_modifiable_actions = 0
        if not total_modifiable_actions > modification_distance and fill_with_modification:
            raise YouIdiotException("Can't have less modifiable actions than modification distance tho!")
        self.in_modification_distance = False
        logger.debug(f"{ModifierGenerator.__name__} finish constructor")

    MODIFIER = "modifier"

    def _is_current_event_before_first_modify(self):
        return self.current_event.sentence_nr < self.first_modification

    def _is_current_event_first_event_to_modify(self):
        return self.current_event.sentence_nr == self.first_modification

    def _is_current_event_first_nonmodified_event(self):
        return self.current_event.sentence_nr == self.first_modification + self.modification_distance

    def _is_current_event_after_first_nonmodified_event(self):
        return self.current_event.sentence_nr > self.first_modification + self.modification_distance

    def _determine_if_modify(self):

        if (
                self._is_current_event_first_nonmodified_event() or
                self._is_current_event_before_first_modify() or
                self._is_current_event_after_first_nonmodified_event()
        ):
            return False
        if self._is_current_event_first_event_to_modify() or self.in_modification_distance:
            return True
        raise NotImplementedError()

    @overrides
    def set_actor(self):
        # check whether the currently processed event is the actual first non-modified one
        if self._is_current_event_first_nonmodified_event():
            # if so, select an actor that did not appear in any modified event before the actual event
            modified_actors = [sent.actor for sent in self.sentences if
                               sent.event_type in self.modify_event_type and self.MODIFIER in sent.features]
            self.current_event.actor = Choices(
                [p for p in self.get_actor_choices() if p not in modified_actors]).random()
        else:
            super().set_actor()

    @overrides
    def set_action(self):
        # determine, whether to have events of modified event type between first modification and actual event
        # or whether to have events of a different type
        fill_with_modification = random.choice((True, False)) \
            if self.fill_with_modifications is None \
            else self.fill_with_modifications
        logger.debug(fmt_dict(locals()))

        # if is first event to modify or first event to non-modify after that
        if self._is_current_event_first_nonmodified_event() or self._is_current_event_first_event_to_modify():
            if self._is_current_event_first_event_to_modify():
                self.in_modification_distance = True
            elif self._is_current_event_first_nonmodified_event():
                self.in_modification_distance = False
            # then event type must be the event type we're modifying
            self.current_event.event_type = self.EVENT_TYPES.keep_only(*self.modify_event_type).random()
            self.current_modifiable_actions += 1

        # if in modification distance or before the first event to modify
        elif self.in_modification_distance or self._is_current_event_before_first_modify():
            # if also fill_with modification (see above)
            if self.in_modification_distance and fill_with_modification:
                # event type must be the event type we're modifying
                self.current_event.event_type = self.EVENT_TYPES.keep_only(*self.modify_event_type).random()
                self.current_modifiable_actions += 1
            else:
                # otherwise it must be different event type
                self.current_event.event_type = (self.EVENT_TYPES - self.modify_event_type).random()
        else:
            # if we still want more events with modifications
            if self.current_modifiable_actions < self.total_modifiable_actions:
                # if the number of generated modified events and the number of events left to generate
                # is smaller than (or equal to) the desired number of total modified events
                sentences_left = self.world.num_sentences - self.current_event.sentence_nr
                if sentences_left <= self.total_modifiable_actions - self.current_modifiable_actions:
                    self.EVENT_TYPES.keep_only(*self.modify_event_type).random()
                else:
                    super().set_action()
                if self.current_event.event_type in self.modify_event_type:
                    self.current_modifiable_actions += 1
            else:
                self.current_event.event_type = (self.EVENT_TYPES - self.modify_event_type).random()

    @overrides
    def set_everything_else(self):
        logger.debug(f"Modify event #{self.current_event.sentence_nr}?")
        if self.current_event.event_type in self.modify_event_type and self._determine_if_modify():
            logger.debug("Modified!")
            self.current_event.features.append(self.MODIFIER)
        else:
            logger.debug("Not modified!")

    def get_relevant_events(self) -> List[Event]:
        return [e for e in StoryGenerator.get_relevant_events(self) if self.MODIFIER not in e.features]

    def post_process_question(self, q: Question):
        base_relevant = [e for e in StoryGenerator.get_relevant_events(self)]
        modified_relevant = [e for e in self.get_relevant_events()]
        if q.type == QuestionTypes.DIRECT:
            n = q.question_data['n'] - 1
            modified = base_relevant[n].sentence_nr != modified_relevant[n].sentence_nr
            q.question_data['modified'] = modified
        elif q.type == QuestionTypes.OVERALL:
            if q.target in self.ATTRIBUTES:
                base_relevant = [e for e in base_relevant if self.is_realised(q.target, e)]
                modified_relevant = [e for e in modified_relevant if self.is_realised(q.target, e)]

            # base number of events for question type is different than the number of effects if including modifier
            # why not just compare len(get_relevant_events)? because if multiple modifiable actions, overall number
            # can be different while for a specific event it is still the same
            num_base_events = sum(1 for e in base_relevant if e.event_type == q.event_type)
            num_modified_events = sum(1 for e in modified_relevant if e.event_type == q.event_type)
            modified = num_base_events != num_modified_events
            q.question_data['modified'] = modified
            # q.question_data['easier'] = (
            #         len(base_relevant) > 1 and len(modified_relevant) == 1 and base_relevant[0] == modified_relevant[0]
            # )

        else:
            raise NotImplementedError()

        # q.question_data['modify_reasoning'] = (
        #         len(base_relevant) > 1 and len(modified_relevant) == 1 or
        #         len(base_relevant) == 1 and len(modified_relevant) == 0
        # )
