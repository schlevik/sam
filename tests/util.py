from copy import deepcopy
from typing import Tuple, List

from stresstest.classes import Config, Question, Bundle
from stresstest.football import bundle
from stresstest.realize import Realizer
from stresstest.util import highlight


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
                    do_realise=True, generator_kwargs=None, unique_sentences=True):
    if not cfg:
        cfg = Config({})
    elif isinstance(cfg, str):
        cfg = Config(cfg)
    elif isinstance(cfg, dict):
        cfg = Config(cfg)
    cfg.pprint()
    print(modifier, "is modifier?")
    if modifier:
        g_class = bundle.generator_modifier
        templates = bundle.templates_modifier
    else:
        g_class = bundle.generator
        # templates = bundle.templates
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
        return generator, cfg, events, None, None, None, None

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
