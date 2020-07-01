from copy import deepcopy
from typing import Tuple, List

from stresstest.classes import Config, Question, Bundle
from stresstest.football import bundle
from stresstest.realize import Realizer
from stresstest.util import highlight


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


def print_out(story, *questions, highlights=None):
    print("===STORY===:")
    color_map = {h: "red" for name in highlights for h in name.split(" ")}
    print(color_map)
    for s in story:
        print(highlight(s, colors=color_map))
    print()
    print("===QUESTIONS===:")
    print(len(questions))
    for q in (q for qs in questions for q in qs):
        if q.realized:
            print(q.realized, q.answer if q.answer else "Unanswerable")


def interactive_env_football(do_print=True, do_realise=True, **kwargs):
    generator_kwargs = dict(
    )
    generator_kwargs.update(kwargs)
    return interactive_env(bundle, modifier=False, do_print=do_print, do_realise=do_realise)


def interactive_env_football_modifier(cfg=None, do_print=True, do_realise=True, first_modification=0,
                                      fill_with_modification=None,
                                      modify_event_types=None,
                                      modification_distance=1, total_modifiable_actions=2, **kwargs):
    generator_kwargs = dict(
        first_modification=first_modification,
        fill_with_modification=fill_with_modification,
        modify_event_types=modify_event_types,
        modification_distance=modification_distance,
        total_modifiable_actions=total_modifiable_actions,
    )
    generator_kwargs.update(kwargs)
    return interactive_env(bundle=bundle, cfg=cfg, modifier=True, do_print=do_print, do_realise=do_realise,
                           generator_kwargs=generator_kwargs)


def interactive_env(bundle: Bundle, cfg=None, modifier=False, do_print=True,
                    do_realise=True, generator_kwargs=None):
    if not cfg:
        cfg = Config({})
    elif isinstance(cfg, str):
        cfg = Config(cfg)
    elif isinstance(cfg, dict):
        cfg = Config(cfg)
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
    if generator.unique_actors:
        print('unique actors?')
        actors = [e.actor for e in events]
        print(actors)
        assert len(set(actors)) == generator.world.num_sentences
        print('yes')
        if getattr(generator, 'unique_coactors', False):
            print('unique coactors?')
            coactors = [e.attributes['coactor'] for e in events]
            print(coactors)
            assert len(set(coactors)) == generator.world.num_sentences
            assert len(set(actors + coactors)) == 2 * generator.world.num_sentences
            print('yes')
    if not do_realise:
        return generator, cfg, events, None, None, None

    realizer = Realizer(**templates)
    story, visits = realizer.realise_story(events, generator.world)
    ssq, maq, uaq, abq = get_questions(generator, realizer, events, visits, story)
    all_questions = (ssq, maq, uaq, abq)

    if do_print:
        actors = [" ".join([e.actor.first, e.actor.last]) for e in events]
        coactors = [" ".join([e.attributes['coactor'].first, e.attributes['coactor'].last]) for e in events]
        print('Actors', actors)
        print('Coactors', coactors)

        print_out(story, ssq, maq, uaq, abq, highlights=actors + coactors)

    return generator, cfg, events, realizer, story, all_questions
