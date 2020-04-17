

from stresstest.classes import Config
from stresstest.generate import StoryGenerator
from stresstest.realize import Realizer


def interactive_env(path='stresstest/resources/config.json', g_class=StoryGenerator):
    c = Config(path)
    c.pprint()
    g = g_class(c)
    t = Realizer()
    ss = g.generate_story()
    story, visits = t.realise_story(ss, g.world)
    (single_span_questions, multi_span_questions, unanswerable_questions, abstractive_questions) = g.generate_questions(
        ss, visits)
    realised_ssqs = []
    realised_msqs = []
    realised_uaqs = []
    realised_aqs = []
    print("===STORY===:")
    for q in single_span_questions:
        try:
            rq = t.realise_question(q)
            realised_ssqs.append(rq)
        except KeyError as e:
            pass
            # print(f"error with {q}")
    for q in multi_span_questions:
        try:
            rq = t.realise_question(q)
            realised_msqs.append(rq)
        except KeyError as e:
            pass
            # print(f"error with {q}")
    for q in unanswerable_questions:
        try:
            rq = t.realise_question(q)
            realised_msqs.append(rq)
        except KeyError as e:
            pass
            # print(f"error with {q}")

    qs = [
        (single_span_questions, realised_ssqs),
        (multi_span_questions, realised_msqs),
        (unanswerable_questions, realised_uaqs),
        (abstractive_questions, realised_aqs)
    ]
    print(story)
    print()
    print("===QUESTIONS===:")
    for q, a in realised_ssqs + realised_msqs + realised_uaqs + realised_aqs:
        print(q, a if a else "Unanswerable")
    return g, c, t, story, qs, ss
