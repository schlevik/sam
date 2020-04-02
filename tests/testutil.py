from stresstest.passage.generate import StoryGenerator
from stresstest.realization.templates import Templates
from stresstest.realization.templates_question import QuestionTemplates


def interactive_env(path='stresstest/resources/config.conf', g=None):
    from stresstest.classes import Config
    c = Config(path, [])
    g = g or StoryGenerator(c)
    t = Templates()
    ss = g.generate_story()
    qt = QuestionTemplates()
    story, visits = t.realise_story(ss, g.world)
    (single_span_questions, multi_span_questions,
     unanswerable_questions, abstractive_questions) = g.generate_questions(ss, visits)
    realised_ssqs = []
    realised_msqs = []
    realised_uaqs = []
    realised_aqs = []
    print(story)
    print()
    print("===QUESTIONS===: ")
    for q in single_span_questions:
        try:
            rq = qt.realise_question(q)
            realised_ssqs.append(rq)
            print(rq[0], rq[1])
        except KeyError as e:
            pass
            #print(f"error with {q}")
    for q in multi_span_questions:
        try:
            rq = qt.realise_question(q)
            realised_msqs.append(rq)
            print(rq[0], rq[1])
        except KeyError as e:
            pass
            #print(f"error with {q}")
    for q in unanswerable_questions:
        try:
            rq = qt.realise_question(q)
            realised_msqs.append(rq)
            print(rq[0], "Unanswerable")
        except KeyError as e:
            pass
            #print(f"error with {q}")

    qs = [
        (single_span_questions, realised_ssqs),
        (multi_span_questions, realised_msqs),
        (unanswerable_questions, realised_uaqs),
        (abstractive_questions, realised_aqs)
    ]
    return g, c, t, story, qs
