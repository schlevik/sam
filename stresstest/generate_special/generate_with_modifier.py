import random
from typing import List, Dict

from loguru import logger

from stresstest.generate import StoryGenerator
from stresstest.classes import Event, QuestionTypes, ReasoningTypes, Question


class ModifierGenerator(StoryGenerator):
    MODIFIER = "modifier"

    def set_anything_else(self):
        # TODO: for now only for goal
        if self.sentence.event_type == 'goal':
            # TODO: maybe more control
            logger.debug("Setting anything else...")
            if random.choice([True, False]):
                logger.debug("Modified!")
                self.sentence.features.append(self.MODIFIER)
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
        # per-sentence action questions
        for event_type in self.EVENT_TYPES:
            events = (event for event in story if
                      event.event_type == event_type and self.MODIFIER not in event.features)
            for ith, event in enumerate(events):
                q = Question(
                    type=QuestionTypes.DIRECT,
                    target="actor",
                    evidence=[event.sentence_nr],
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
                        reasoning=ReasoningTypes.Retrieval if ith == 0 else ReasoningTypes.OrderingEasy,
                        question_data={"n": ith + 1},

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
            events = sum(s.event_type == event_type and self.MODIFIER not in s.features for s in story)
            q.evidence = [s.sentence_nr for s in story if
                          s.event_type == event_type and self.MODIFIER not in s.features]
            if events > 1:
                q.reasoning = ReasoningTypes.MultiRetrieval
                q.answer = [" ".join((s.actor['first'], s.actor['last'])) for s in story if
                            s.event_type == event_type and self.MODIFIER not in s.features]
                multi_span_questions.append(q)
            elif events == 1:
                q.reasoning = ReasoningTypes.Retrieval
                q.answer = next(
                    " ".join((s.actor['first'], s.actor['last'])) for s in story if
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
