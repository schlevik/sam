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

    def generate_questions(self, story: List[Event],
                           visits: Dict[int, List[str]]):
        # extractive
        single_span_questions = []
        multi_span_questions = []
        unanswerable_questions = []
        abstractive_questions = []
        # TODO: refactor with conditions
        for event_type in self.EVENT_TYPES:
            base_events = [event for event in story if event.event_type == event_type]
            events = (event for event in story if
                      event.event_type == event_type and self.MODIFIER not in event.features)

            # per-sentence action questions
            for ith, event in enumerate(events):
                modified = base_events[ith].sentence_nr != event.sentence_nr
                q = Question(
                    type=QuestionTypes.DIRECT,
                    target="actor",
                    evidence=[event.sentence_nr],
                    event_type=event_type,
                    # TODO: WHAT IF COREF ETC
                    answer=self.post_process_actor_answer(event.actor),
                    reasoning=ReasoningTypes.Retrieval if ith == 0 else ReasoningTypes.OrderingEasy,
                    question_data={"n": ith + 1, "modified": modified}
                )
                logger.debug(f"Question: {q}")
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
                        reasoning=ReasoningTypes.Retrieval if ith == 0 else ReasoningTypes.OrderingEasy,
                        question_data={"n": ith + 1, "modified": modified},

                    )
                    if any(f"sent.attributes.{attribute}" in v for v in visits[event.sentence_nr]):
                        q.answer = self.post_process_attribute_answers(attribute, event.attributes[attribute])
                        q.evidence = [event.sentence_nr]
                        single_span_questions.append(q)
                    else:
                        q.answer = None
                        q.evidence = []
                        unanswerable_questions.append(q)

            # overall questions

            # target = actor
            q = Question(
                type=QuestionTypes.OVERALL,
                target='actor',
                event_type=event_type,

            )

            base_events = sum(s.event_type == event_type for s in story)
            events = sum(s.event_type == event_type and self.MODIFIER not in s.features for s in story)

            q.question_data['modified'] = base_events != events
            q.evidence = [s.sentence_nr for s in story if
                          s.event_type == event_type and self.MODIFIER not in s.features]
            if events > 1:
                q.reasoning = ReasoningTypes.MultiRetrieval
                q.answer = [self.post_process_actor_answer(s.actor) for s in story if
                            s.event_type == event_type and self.MODIFIER not in s.features]
                multi_span_questions.append(q)
            elif events == 1:
                q.reasoning = ReasoningTypes.Retrieval
                q.answer = next(self.post_process_actor_answer(s.actor) for s in story if
                                s.event_type == event_type and self.MODIFIER not in s.features)
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
                    return any(f"sent.attributes.{attribute}" in v for v in visits[s.sentence_nr]) and \
                           s.event_type == event_type and self.MODIFIER not in s.features

                def base_condition(s):
                    return any(f"sent.attributes.{attribute}" in v for v in visits[s.sentence_nr]) and \
                           s.event_type == event_type

                base_events = sum(1 for s in story if base_condition(s))
                events = sum(1 for s in story if condition(s))
                q.question_data['modified'] = base_events != events
                q.evidence = [s.sentence_nr for s in story if condition(s)]
                # answers = [s.attributes[attribute] for s in story if condition(s)]
                answers = [self.post_process_attribute_answers(attribute, s.attributes[attribute])
                           for s in story if condition(s)]
                # if attribute == 'coactor':
                #    answers = [" ".join((a['first'], a['last'])) for a in answers]
                if events > 1:
                    q.reasoning = ReasoningTypes.MultiRetrieval
                    q.answer = answers
                    multi_span_questions.append(q)
                elif events == 1:
                    q.reasoning = ReasoningTypes.Retrieval
                    q.answer = answers[0]
                    if next(e.sentence_nr for e in story if condition(e)) == next(
                            e.sentence_nr for e in story if base_condition(e)):
                        q.question_data['easier'] = True
                    single_span_questions.append(q)
                elif events < 1:
                    q.answer = None
                    unanswerable_questions.append(q)

        return (single_span_questions, multi_span_questions,
                unanswerable_questions, abstractive_questions)
