import random
from typing import List, Dict

from loguru import logger
from overrides import overrides

from stresstest.generate import StoryGenerator
from stresstest.classes import Event, QuestionTypes, ReasoningTypes, Question, Choices, YouIdiotException
from stresstest.util import fmt_dict


class ModifierGenerator(StoryGenerator):
    def __init__(self, config, first_modification=0, fill_with_modification=None, modify_event_types=None,
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
        super().__init__(config)
        self.first_modification = first_modification
        self.fill_with_modifications = fill_with_modification
        self.modify_event_type = modify_event_types or ['goal']
        self.modification_distance = modification_distance
        self.total_modifiable_actions = 2
        self.current_modifiable_actions = 0
        if not total_modifiable_actions > modification_distance and fill_with_modification:
            raise YouIdiotException("Can't have less modifiable actions than modification distance tho!")
        self.in_modification_distance = False

    MODIFIER = "modifier"

    def _determine_if_modify(self):
        actual_event = self.current_event.sentence_nr == self.first_modification + self.modification_distance
        first_modify = self.current_event.sentence_nr == self.first_modification
        before_first_modify = self.current_event.sentence_nr < self.first_modification
        in_modification_distance = self.in_modification_distance
        after_modification_distance = self.current_event.sentence_nr > self.first_modification + self.modification_distance

        logger.debug(fmt_dict(locals()))

        if actual_event or before_first_modify or after_modification_distance:
            return False
        if first_modify or in_modification_distance:
            return True
        raise NotImplementedError()

    @overrides
    def set_actor(self):
        actual_event = self.current_event.sentence_nr == self.first_modification + self.modification_distance
        logger.debug(fmt_dict(locals()))

        if actual_event:
            modified_actors = [sent.actor for sent in self.sentences if
                               sent.event_type in self.modify_event_type and self.MODIFIER in sent.features]
            self.current_event.actor = Choices([p for p in self.world['players'] if p not in modified_actors]).random()
        else:
            super().set_actor()

    @overrides
    def set_action(self):
        actual_event = self.current_event.sentence_nr == self.first_modification + self.modification_distance
        first_modify = self.current_event.sentence_nr == self.first_modification
        before_first_modify = self.current_event.sentence_nr < self.first_modification
        fill_with_modification = random.choice((True, False)) \
            if self.fill_with_modifications is None \
            else self.fill_with_modifications
        logger.debug(fmt_dict(locals()))
        if actual_event or first_modify:
            if first_modify:
                self.in_modification_distance = True
            elif actual_event:
                self.in_modification_distance = False
            self.current_event.event_type = self.EVENT_TYPES.keep_only(*self.modify_event_type).random()
            self.current_modifiable_actions += 1

        elif self.in_modification_distance or before_first_modify:
            if self.in_modification_distance and fill_with_modification:
                self.current_event.event_type = self.EVENT_TYPES.keep_only(*self.modify_event_type).random()
                self.current_modifiable_actions += 1
            else:
                self.current_event.event_type = (self.EVENT_TYPES - self.modify_event_type).random()
        else:
            if self.current_modifiable_actions < self.total_modifiable_actions:
                # if sentences left <= desired total modifiable actions - current modifiable actions
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
    def set_anything_else(self):
        # TODO: for now only for goal
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
                    answer=" ".join((event.actor['first'], event.actor['last'])),
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
                        if attribute == 'coactor':
                            q.answer = " ".join(
                                (event.attributes['coactor'].first, event.attributes['coactor'].last))
                        else:

                            q.answer = event.attributes[attribute]
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
                q.answer = [" ".join((s.actor['first'], s.actor['last'])) for s in story if
                            s.event_type == event_type and self.MODIFIER not in s.features]
                multi_span_questions.append(q)
            elif events == 1:
                q.reasoning = ReasoningTypes.Retrieval
                q.answer = next(" ".join((s.actor['first'], s.actor['last'])) for s in story if
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
                    if next(e.sentence_nr for e in story if condition(e)) == next(
                            e.sentence_nr for e in story if base_condition(e)):
                        q.question_data['easier'] = True
                    single_span_questions.append(q)
                elif events < 1:
                    q.answer = None
                    unanswerable_questions.append(q)

        return (single_span_questions, multi_span_questions,
                unanswerable_questions, abstractive_questions)
