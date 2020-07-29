from copy import deepcopy
from typing import Tuple, List

from stresstest.classes import Config, Question, Bundle
from stresstest.football import bundle
from stresstest.generate_utils import generate_and_realise
from stresstest.realize import Realizer
from stresstest.reasoning import retrieval, retrieval_reverse
from stresstest.reasoning.argselect import argmin, argmax
from stresstest.reasoning.bridge import bridge_reverse, bridge
from stresstest.reasoning.comparison import comparison_reverse, comparison
from stresstest.reasoning.retrieval_two import retrieval_two_reverse, retrieval_two
from stresstest.print_utils import highlight


def only(sents_or_bundle, n, action='test'):
    sents_or_bundle = deepcopy(sents_or_bundle)
    if isinstance(sents_or_bundle, Bundle):
        # sents_or_bundle.templates['sentences'][action] = [sents_or_bundle.templates['sentences'][action][n]]
        sents_or_bundle.templates_modifier['sentences'][action] = [
            sents_or_bundle.templates_modifier['sentences'][action][n]]
    elif isinstance(sents_or_bundle, dict):
        sents_or_bundle['sentences'][action] = [sents_or_bundle['sentences'][action][n]]
    else:
        sents_or_bundle[action] = [sents_or_bundle[action][n]]
    return sents_or_bundle


def get_questions(generator, realizer, events, visits, story) \
        -> Tuple[List[Question], List[Question], List[Question], List[Question]]:
    (single_span_questions, multi_span_questions, unanswerable_questions,
     abstractive_questions) = generator.generate_questions(events, visits)

    for q in single_span_questions + multi_span_questions + unanswerable_questions + abstractive_questions:
        realizer.realise_question(q, story)

    return single_span_questions, multi_span_questions, unanswerable_questions, abstractive_questions


def print_out(story, *questions, highlights=None):
    print("===STORY===:")
    color_map = {h: "red" for name in highlights for h in name.split(" ")} if highlights else {}
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


def interactive_env_football_modifier(changed_bundle=None, cfg=None, do_print=True, do_realise=True,
                                      first_modification=0,
                                      fill_with_modification=None,
                                      modify_event_types=None,
                                      modifier_type=None,
                                      modification_distance=1, total_modifiable_actions=2, **kwargs):
    generator_kwargs = dict(
        first_modification=first_modification,
        fill_with_modification=fill_with_modification,
        modify_event_types=modify_event_types or ['goal'],
        modification_distance=modification_distance,
        total_modifiable_actions=total_modifiable_actions,
        modifier_type=modifier_type
    )
    changed_bundle = changed_bundle or bundle
    generator_kwargs.update(kwargs)
    return interactive_env(bundle=changed_bundle, cfg=cfg, modifier=True, do_print=do_print, do_realise=do_realise,
                           generator_kwargs=generator_kwargs)


def interactive_env(bundle: Bundle, cfg=None, modifier=False, do_print=True,
                    do_realise=True, generator_kwargs=None):
    if not modifier:
        raise NotImplementedError()
    if not cfg:
        cfg = Config({})
    elif isinstance(cfg, str):
        cfg = Config(cfg)
    elif isinstance(cfg, dict):
        cfg = Config(cfg)
    cfg.pprint()
    # if modifier:
    g_class = bundle.generator_modifier
    templates = bundle.templates_modifier
    # else:
    # g_class = bundle.generator
    # templates = bundle.templates
    generator_kwargs = generator_kwargs or {}
    generator = g_class(cfg, **generator_kwargs)
    events = generator.generate_story()
    if generator.unique_actors:
        actors = [e.actor for e in events]
        assert len(set(actors)) == generator.world.num_sentences
        if getattr(generator, 'unique_coactors', False):
            coactors = [e.attributes['coactor'] for e in events]
            assert len(set(coactors)) == generator.world.num_sentences
            assert len(set(actors + coactors)) == 2 * generator.world.num_sentences
    if not do_realise:
        return generator, cfg, events, None, None, None, None

    realizer = Realizer(**templates, unique_sentences=False)
    story, visits = realizer.realise_story(events, generator.world)
    ssq, maq, uaq, abq = get_questions(generator, realizer, events, visits, story)
    all_questions = (ssq, maq, uaq, abq)

    if do_print:
        actors = [" ".join([e.actor.first, e.actor.last]) for e in events]
        coactors = [" ".join([e.attributes['coactor'].first, e.attributes['coactor'].last]) for e in events]
        print('Actors', actors)
        print('Coactors', coactors)

        print_out(story, ssq, maq, uaq, abq, highlights=actors + coactors)
    print(realizer.context.chosen_templates)
    return generator, cfg, events, realizer, story, all_questions, visits


def showcase(given_bundle=None, n=0):
    test_bundle = only(given_bundle, n, 'goal') if given_bundle else only(bundle, n, 'goal')
    templates = test_bundle.templates_modifier
    generator, cfg, events, realizer, story, all_questions, visits = interactive_env_football_modifier(
        test_bundle, cfg={"world.num_sentences": 2}, do_print=False, do_realise=False
    )
    # realizer = Realizer(**templates, unique_sentences=False)
    # story, visits = realizer.realise_story(events, generator.world)
    # ssq, maq, uaq, abq = get_questions(generator, realizer, events, visits, story)
    # print_out(story, ssq)
    for f in ['VP-neg-impl', 'RB', 'MD', 'VP-pol-rev', 'VB-neg-impl', 'VB-pol-rev']:
        print(f"==== {f} ====")
        events[0].features = [f]
        generator.modifier_type = f
        realizer = Realizer(**templates, unique_sentences=False)
        story, visits = realizer.realise_story(events, generator.world)
        # ssq, maq, uaq, abq = get_questions(generator, realizer, events, visits, story)
        print(story[0])
        # print_out(story, [])


def interactive_env_aligned(bundle=bundle, modify_event_type='goal', modifier_type='RB', max_sents=5, max_modifier=3,
                            config=None, per_modify_distance_per_reasoning=1, reasonings=None, num_workers=1):
    config = config or {"world.num_sentences": max_sents, 'unique_actors': True, 'world.num_players': 4 * max_sents}
    reasonings = reasonings or [retrieval, retrieval_two, retrieval_reverse, retrieval_two_reverse, bridge,
                                bridge_reverse, argmax, argmin, comparison, comparison_reverse]
    reasoning_map = {
        r: per_modify_distance_per_reasoning for r in reasonings
    }
    return generate_and_realise(bundle, config, modify_event_type, modifier_type,
                                reasoning_map, max_modifiers=max_modifier, num_workers=num_workers)
