from copy import deepcopy
from typing import Tuple, List

from stresstest.classes import Config, Question
from stresstest.generate import StoryGenerator
from stresstest.generate_special.generate_with_modifier import ModifierGenerator
from stresstest.realize import Realizer
from tests.resources.templates import sentences, dollar, at, percent, bang, templates


def only(sents, n, action='test'):
    sents = deepcopy(sents)
    sents[action] = [sents[action][n]]
    return sents


def get_questions(generator, realizer, events, visits, story) \
        -> Tuple[List[Question], List[Question], List[Question], List[Question]]:
    (single_span_questions, multi_span_questions, unanswerable_questions,
     abstractive_questions) = generator.generate_questions(events, visits)

    for q in single_span_questions + multi_span_questions + unanswerable_questions + abstractive_questions:
        realizer.realise_question(q, story)

    return single_span_questions, multi_span_questions, unanswerable_questions, abstractive_questions


def print_out(story, *questions):
    print("===STORY===:")
    print('\n'.join(story))
    print()
    print("===QUESTIONS===:")
    print(len(questions))
    for q in (q for qs in questions for q in qs):
        if q.realized:
            print(q.realized, q.answer if q.answer else "Unanswerable")


def env_for_modifier(path='stresstest/resources/team-names.json', g_class=ModifierGenerator, do_print=True,
                     do_realise=True):
    from stresstest.resources.templates_modifier import sentences, dollar, at, percent, bang, question_templates
    realizer = Realizer(sentences=sentences, dollar=dollar, at=at, percent=percent, bang=bang,
                        question_templates=question_templates, unique_sentences=False)

    return interactive_env(path, g_class, realizer, do_print, do_realise)


def interactive_env(path='stresstest/resources/team-names.json', g_class=StoryGenerator, realizer=None, do_print=True,
                    do_realise=True):
    cfg = Config(path)
    cfg.pprint()
    generator = g_class(cfg)
    events = generator.generate_story()
    if not do_realise:
        return generator, cfg, events, None, None, None
    realizer = realizer or Realizer()
    story, visits = realizer.realise_story(events, generator.world)
    ssq, maq, uaq, abq = get_questions(generator, realizer, events, visits, story)
    all_questions = (ssq, maq, uaq, abq)
    if do_print:
        print_out(story, ssq, maq, uaq, abq)
    return generator, cfg, events, realizer, story, all_questions


class TestRealizer(Realizer):
    def __init__(self, sentences=sentences, dollar=dollar, at=at, percent=percent, bang=bang, templates=templates,
                 *args, **kwargs):
        super().__init__(sentences=sentences, dollar=dollar, at=at, percent=percent, bang=bang,
                         question_templates=templates, *args, **kwargs)
