import random
from typing import Dict

import stresstest.football.resources.modifier.sentences
from stresstest.classes import Context, Event
from stresstest.football.classes import Player


def _is_first_event_of_its_type_for_team(ctx: Context):
    event_type = ctx.sent.event_type
    team = ctx.sent['actor']['team']
    return next(sent.sentence_nr for sent in ctx.sentences
                if sent.event_type == event_type and sent.actor['team'] == team) == ctx.sent_nr


def bridge_short(qdata: Dict[str, Event]):
    bridge_event = qdata['bridge-event']
    actor: Player = bridge_event.actor
    coactor: Player = bridge_event.attributes['coactor']
    if bridge_event.event_type == 'goal':
        return f"{actor.first} {actor.last} " + random.choice(["'s goal"])
    elif bridge_event.event_type == 'foul':
        if random.choice([True, False]):
            return f"{actor.first} {actor.last} " + random.choice(["'s foul", "fouled"])
        else:
            return random.choice([f"{coactor.first} {coactor.last} was fouled",
                                  f"the foul on {coactor.first} {coactor.last}"])
    else:
        raise NotImplementedError


def bridge_long(qdata: Dict[str, Event]):
    bridge_event = qdata['bridge-event']
    actor: Player = bridge_event.actor
    coactor: Player = bridge_event.attributes['coactor']
    if bridge_event.event_type == 'goal':
        return f"{actor.first} {actor.last} " + random.choice(["has scored"])
    elif bridge_event.event_type == 'foul':
        if random.choice([True, False]):
            return f"{actor.first} {actor.last} " + random.choice(["has commited a foul", "has fouled"])
        else:
            return f"{coactor.first} {coactor.last} was fouled"
    else:
        raise NotImplementedError


def bridge_impl(qdata: Dict[str, Event]):
    bridge_event = qdata['bridge-event']
    actor: Player = bridge_event.actor
    if bridge_event == 'goal':
        if random.choice([True, False]):
            return bridge_long(qdata)
        else:
            return f"{actor.first} {actor.last}"
    else:
        return bridge_long(qdata)


bang = {
    "NEXT": (lambda ctx: "first" if _is_first_event_of_its_type_for_team(ctx) else "next", 2),
    # "PREAMBLE": Preamble,
    "RANDINT": (lambda ctx: random.randint(1, 15)),
    "PRPS": (lambda ctx: "her" if ctx.world.gender == 'female' else "his"),
    "PRP": (lambda ctx: "she" if ctx.world.gender == 'female' else "he"),
    # difference in time from last event
    # TODO: it's a tiny bit buggy because it doesn't register itself in ctx['visited'], ok for span-based though
    "MINDIFF": (lambda ctx: ctx.sent.attributes['time'] -
                            ctx.sentences[ctx.sent_nr - 1].attributes[
                                'time']),
    "OTHERTEAM": (
        lambda ctx: next(t.name for t in ctx.world.teams if t.id != ctx.sent.actor.team.id)),
    'bridge-short': bridge_short,
    'bridge-long': bridge_long,
    'bridge-impl': bridge_impl
}
