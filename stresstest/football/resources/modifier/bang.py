import random

import stresstest.football.resources.modifier.sentences
from stresstest.classes import Context


def _is_first_event_of_its_type_for_team(ctx: Context):
    event_type = ctx.sent.event_type
    team = ctx.sent['actor']['team']
    return next(sent.sentence_nr for sent in ctx.sentences
                if sent.event_type == event_type and sent.actor['team'] == team) == ctx.sent_nr


bang = {
    "NEXT": (lambda ctx: "first" if _is_first_event_of_its_type_for_team(ctx) else "next", 2),
    # "PREAMBLE": Preamble,
    "RANDINT": (lambda ctx: random.randint(1, 15)),
    "PRPS": (lambda ctx: "her" if ctx.world.gender == 'female' else "his"),
    "PRP": (lambda ctx: "she" if ctx.world.gender == 'female' else "he"),
    # difference in time from last event
    # TODO: it's a tiny bit buggy because it doesn't register itself in ctx['visited'], ok for span-based though
    "MINDIFF": (lambda ctx: ctx.sent.attributes['time'] -
                            stresstest.football.resources.modifier.sentences.sentences[ctx.sent_nr - 1].attributes[
                                'time']),
    "OTHERTEAM": (
        lambda ctx: next(t.name for t in ctx.world.teams if t.id != ctx.sent.actor.team.id))

}
