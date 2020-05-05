from copy import deepcopy

from stresstest.classes import Config
from stresstest.generate import StoryGenerator
from stresstest.realize import Realizer
from tests.resources.templates import sentences, dollar, at, percent, bang, templates


def only(sents, n, action='test'):
    sents = deepcopy(sents)
    sents[action] = [sents[action][n]]
    return sents


def interactive_env(path='stresstest/resources/config.json', g_class=StoryGenerator, realizer=None, do_print=True):
    c = Config(path)
    c.pprint()
    g = g_class(c)
    t = realizer or Realizer()
    ss = g.generate_story()
    story, visits = t.realise_story(ss, g.world)
    (single_span_questions, multi_span_questions, unanswerable_questions, abstractive_questions) = g.generate_questions(
        ss, visits)
    print(len(unanswerable_questions))
    realised_ssqs = []
    realised_msqs = []
    realised_uaqs = []
    realised_aqs = []
    if do_print:
        print("===STORY===:")
    for q in single_span_questions:
        # try:
        rq = t.realise_question(q)
        if rq:
            realised_ssqs.append(rq)
        # except KeyError as e:
        pass
        # print(f"error with {q}")
    for q in multi_span_questions:
        # try:
        rq = t.realise_question(q)
        if rq:
            realised_msqs.append(rq)
        # except KeyError as e:
        pass
        # print(f"error with {q}")
    for q in unanswerable_questions:
        # try:
        rq = t.realise_question(q)
        if rq:
            realised_uaqs.append(rq)
        # except KeyError as e:
        pass
        # print(f"error with {q}")
    qs = [
        (single_span_questions, realised_ssqs),
        (multi_span_questions, realised_msqs),
        (unanswerable_questions, realised_uaqs),
        (abstractive_questions, realised_aqs)
    ]
    if do_print:
        print('\n'.join(story))
        print()
        print("===QUESTIONS===:")
        for q, a in realised_ssqs + realised_msqs + realised_uaqs + realised_aqs:
            print(q, a if a else "Unanswerable")
    return g, c, t, story, qs, ss


class TestRealizer(Realizer):
    def __init__(self, sentences=sentences, dollar=dollar, at=at, percent=percent, bang=bang, templates=templates,
                 *args, **kwargs):
        super().__init__(sentences=sentences, dollar=dollar, at=at, percent=percent, bang=bang,
                         question_templates=templates, *args, **kwargs)
