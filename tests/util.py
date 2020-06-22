from copy import deepcopy
from typing import Tuple, List

from stresstest.classes import Config, Question, Bundle
from stresstest.football import bundle
from stresstest.football.generate_with_modifier import FootballModifierGenerator
from stresstest.realize import Realizer


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


def env_for_modifier(path='stresstest/football/resources/team-names.json', g_class=FootballModifierGenerator,
                     do_print=True,
                     do_realise=True):
    from stresstest.football.resources.templates_modifier import sentences, dollar, at, percent, bang, \
        question_templates
    realizer = Realizer(sentences=sentences, dollar=dollar, at=at, percent=percent, bang=bang,
                        question_templates=question_templates, unique_sentences=False)

    return interactive_env(path, g_class, realizer, do_print, do_realise)


def interactive_env_football(do_print=True, do_realise=True):
    return interactive_env(bundle, modifier=False, do_print=do_print, do_realise=do_realise)


def interactive_env_football_modifier(do_print=True, do_realise=True, first_modification=0, fill_with_modification=None,
                                      modify_event_types=None,
                                      modification_distance=1, total_modifiable_actions=2):
    generator_kwargs = dict(
        first_modification=first_modification,
        fill_with_modification=fill_with_modification,
        modify_event_types=modify_event_types,
        modification_distance=modification_distance,
        total_modifiable_actions=total_modifiable_actions
    )
    return interactive_env(bundle, modifier=True, do_print=do_print, do_realise=do_realise,
                           generator_kwargs=generator_kwargs)


def interactive_env(bundle: Bundle, path='stresstest/football/resources/team-names.json', modifier=False, do_print=True,
                    do_realise=True, generator_kwargs=None):
    cfg = Config(path)
    cfg.pprint()
    if modifier:
        g_class = bundle.generator_modifier
        templates = bundle.templates_modifier
    else:
        g_class = bundle.generator
        templates = bundle.templates
    generator_kwargs = generator_kwargs or {}
    generator = g_class(cfg, **generator_kwargs)
    events = generator.generate_story()

    if not do_realise:
        return generator, cfg, events, None, None, None

    realizer = Realizer(**templates)
    story, visits = realizer.realise_story(events, generator.world)
    ssq, maq, uaq, abq = get_questions(generator, realizer, events, visits, story)
    all_questions = (ssq, maq, uaq, abq)
    if do_print:
        print_out(story, ssq, maq, uaq, abq)
    return generator, cfg, events, realizer, story, all_questions
