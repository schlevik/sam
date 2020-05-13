import random
import re

from loguru import logger

from stresstest.classes import Context

sentences = {
    "goal": [
        "%CONNECTIVE.VBD> $ACTOR $ACTORTEAM.name-pos-post $VBD.goal a ($JJ.positive) goal .",

        "%CONNECTIVE.VBD> $ACTOR $VBD.goal a ($JJ.positive) goal for $ACTORTEAM.name .",

        "$ACTORTEAM.name-pos-pre player $ACTOR $VP.attention %CONNECTIVE.ADVP , "
        "$VBG.goal a ($JJ.positive) goal $DISTANCE.PP .",

        "$ACTOR 's goal ( , [$RDM.VBG.goal|$RDM.NOVB] , ) arrived $TIME after !PRPS teammate "
        "$COACTOR 's $PASS-TYPE and [$RDM.CC-V.goal|$RDM.S.goal] .",

        "$TIME a $PASS-TYPE [went to|arrived at] ($ACTORTEAM.name-pos-pre) $COACTOR in $POSITION.VERTICAL and "
        "$COREF-PLAYER swept $POSITION.HEIGHT.NN to the $POSITION.BOX for $ACTOR to poke past the $GOALKEEPER .",

        "A $JJ.positive $DISTANCE.JJ strike from $ACTOR [flying|homing] into $POSITION.GOAL past "
        "[the $GOALKEEPER|a helpess $GOALKEEPER] ($RDM.PP.goal) %CONNECTIVE.VP .",

        "$ACTOR , one of $ACTORTEAM.name-pos-pre better performers today, %CONNECTIVE.VP"
        " as !PRP scored $TIME [$REASON.PP.goal| and $RDM.S.goal] .",

        "$ACTOR scored $TIME to %CONNECTIVE.IVP when !PRP $REASON.CC-V.goal (and $REASON.CC-V.goal) "
        "before $VBG.goal the ball $POSITION.PP.GOAL .",

        "%CONNECTIVE.VBD> $ACTOR scored $TIME $VBG.goal the ball $POSITION.PP.GOAL "
        "after !PRP $REASON.CC-V.goal (and $REASON.CC-V.goal) .",

        "%CONNECTIVE.VBD> the ball arrived [on|at] the $POSITION.BOX (at pace) and [$RDM.VBG.goal] , $ACTOR $VBDO.goal "
        "(just) $POSITION.PP.GOAL (to leave the $GOALKEEPER with no chance) .",

        # TODO: extract "expression player's drive squirmed"
        "%CONNECTIVE.VBD> $ACTOR was free on the $POSITION.BOX , and with the defence slow to react, "
        "the $ACTORTEAM.name-pos-pre player 's drive squirmed beyond the $GOALKEEPER .",

        "%CONNECTIVE.VBD> $ACTOR , on the end of it , $VBDO.goal $POSITION.GOAL $RDM.VBG.goal .",

        "$ACTOR $VBD.goal [$ACTORTEAM.name-pos-pre !NEXT $NN.goal|the !NEXT $NN.goal for $ACTORTEAM.name] "
        "to %CONNECTIVE.IVP after $REASON.S.goal .",
    ],
    "foul": [
        "%CONNECTIVE.VBD> $COACTOR ($COACTORTEAM.name-pos-post) had gone down with $INJURY "
        "after a ($JJ.negative) foul by ($ACTORTEAM.name-pos-pre) $ACTOR .",

        "%CONNECTIVE.VBD> ($COACTORTEAM.name-pos-pre) $COACTOR $VBD-PASSIVE.foul ($ADVJ.neg) by "
        "[$ACTOR ($ACTORTEAM.name-pos-post)|($COACTORTEAM.name-pos-pre) $ACTOR] ($TIME) .",

        "%CONNECTIVE.VBD> $ACTOR $VBD.foul ($COACTORTEAM.name-pos-pre) $COACTOR ($TIME) .",

        "%CONNECTIVE.VBD> $ACTOR $VBD.foul $COACTOR ($COACTORTEAM.name-pos-post) [$RDM.PP.foul|and $RDM.S.any] .",

        "$RDM.S.any as $COACTOR was withdrawn $TIME with !PRPS $BODYPART in a brace following a ($JJ.negative) "
        "challenge from $ACTOR .",

        "%CONNECTIVE.VBG> $ACTOR $VBG.foul $COACTOR $RDM.PP.foul .",

        "%CONNECTIVE.VBG> $ACTOR $RDM.VBG.foul , "
        "$VBG.foul $COACTOR near the $POSITION.BOX .",

        "%CONNECTIVE.VBD> $ACTOR $RDM.VBD.foul with a ($JJ.negative) $NN.foul ($RDM.PP.foul) .",

        "%CONNECTIVE.VBG> $COACTOR winning the ball [in $POSITION.HORIZONTAL|on $POSITION.VERTICAL] "
        "(for $ACTORTEAM.name) and drawing a $NN.foul from $ACTOR ."
        # "If (only) #sent.actor $VBD.goal the penalty, "
        # "the score would be @CONDITION.then, otherwise it would "
        # "stay @CONDITION.else, @CONDITION.value"
    ]
}
dollar = {
    # just some random stuff that makes things look pretty
    "RDM": {
        "S": {
            "any": ["the stadium went wild"],
            "goal": ["$RDM.PP.goal it was spectacular", "$RDM.S.any"]
        },
        "CC-V": {
            "goal": ["ended !PRPS dry run of !RANDINT games",
                     "made the fans chant !PRPS name"]
        },
        # TODO: no verb, need to look up how we call it
        "NOVB": ["a contender for the $RDM.AWARD",
                 "a reward for !PRPS hard work",
                 "!PRPS !RANDINT th goal [of the season|for the club]"],
        # inserted as a gerund
        "VBG": {
            "foul": ["disappointing (the crowd) with an $JJ.negative action"],
            "goal": [
                "being $RDM.NOVB",
                "drawing attention from even !PRPS biggest sceptics",
                "following a $JJ.positive juggle"
            ]
        },
        "VBD": {
            "foul": [
                "(had only) just showed !PRPS reckless edge",
                "disappointed"
            ]
        },
        "AWARD": ["highlight of the day",
                  "action of the match"],
        "PP": {
            "foul": ["for a [$JJ.promising|$JJ.attention] free-kick $NN.opportunity "
                     "[for $NONACTORTEAM.name|for !PRPS opponents|]",

                     'for which !PRP was booked'],
            "goal": ["for !PRPS !RANDINT th league goal of the season"]
        },
        # "BEGINNING": ["$ACTORTEAM.name did well to withstand the barrage"]
    },
    "REASON": {
        "S": {
            "goal": ["$COACTOR (inadvertently) $VBD.pass the ball into !PRPS path (following $REASON.NP)"]
        },

        "NP": ["a run on $POSITION.VERTICAL", 'a (decisive) counter-attack'],

        "PP": {
            "goal": ["after $REASON.NP"]  # TODO: on out wide / on the centre
        },
        "CC-V": {
            "any": ["dribbled !RANDINT metres (on $POSITION.VERTICAL)"],  # TODO: on out wide / on the centre
            "goal": ["ran !RANDINT metres",
                     "intercepted [$NONACTORTEAM.name-pos-pre goalkeeper's goal kick"
                     "| the goal kick of $NONACTORTEAM.name-pos-pre goal keeper]",
                     "$REASON.CC-V.any"]
        }
    },

    "GOALKEEPER": ["goalkeeper", "woman between the posts", "last line of defence"],
    "COREF-PLAYER": ["!PRP", "the player"],
    "POSITION": {
        "VERTICAL": ["the flank", "out wide", "the centre"],
        "HORIZONTAL": ['the middle field', 'their own half', 'the attacking third'],
        "BOX": ["near post", "far post", "penalty spot", "6-yard-area", "edge of the area"],
        "GOAL": [
            "the ($POSITION.HEIGHT.JJ) ($POSITION.LR) corner", "the middle of the goal", "the back of the net",
            "between the posts"
        ],
        "HEIGHT": {
            "JJ": ["lower", "upper"],
            "NN": ["low", "high"]
        },
        "LR": ["left", "right"],
        "PP": {
            "GOAL": ["in $POSITION.GOAL", "under the bar", "off the [post|bar] and in $POSITION.GOAL"],
        }
    },
    "INJURY": ["a (potential) ($BODYPART) injury"],
    "BODYPART": ["knee", "head", "shoulder", "arm", "hip", "leg", "ankle"],
    # NP description of assist
    "PASS-TYPE": ["($JJ.positive) pass", "($JJ.accurate) cross",
                  "($JJ.risky) through ball", "soft clearance", "stray ball"],
    # NP distance description
    "DISTANCE": {
        "PP": ["from #sent.attributes.distance metres (away)"],
        "JJ": ["#sent.attributes.distance metres"],
    },

    # NP time description
    "TIME": ["in minute #sent.attributes.time",
             "on the #sent.attributes.time th minute"],
    # ACTOR ACCESSOR
    "ACTOR": ["#sent.actor.first #sent.actor.last"],
    # COACTOR ACCESSOR
    "COACTOR": [
        "#sent.attributes.coactor.first #sent.attributes.coactor.last"],
    "ACTORTEAM": {
        "name": ['#sent.actor.team.name'],
        "name-pos-pre": ["$ACTORTEAM.name 's"],
        "name-pos-post":
            ["of $ACTORTEAM.name", ", a player of $ACTORTEAM.name ,"]
    },
    "NONACTORTEAM": {
        "name": ["!OTHERTEAM"],
        "name-pos-pre": ["$NONACTORTEAM.name 's"],
        "name-pos-post":
            ["of $NONACTORTEAM.name", ", a player of $NONACTORTEAM.name ,"]
    },
    "VP": {
        "attention": ['put an exclamation mark', 'became the talking point of the day', 'attracted (lots of) attention']
    },

    # variable resolution for the team of the action's coactor
    "COACTORTEAM": {
        # accessor of the team name
        "name": ['#sent.attributes.coactor.team.name'],
        # possessive accessor before the actual co-actor
        "name-pos-pre": ["$COACTORTEAM.name 's"],
        # possessive accessor after the co-actor
        "name-pos-post":
            ["of $COACTORTEAM.name", "playing for $COACTORTEAM.name"]
    },
    # adjectives
    "JJ": {
        "positive":
            ["spectacular", "wonderful", "amazing", "stunning", "searing", "mesmerising"],
        "promising": ["promising", "auspicious", "promisingly looking", "auspiciously looking"],
        "accurate": ["accurate", "pin-point", ],
        "risky": ["bold", "daring", "risky"],
        "attention": ["remarkable", "interesting"],
        "negative": ["bad", "harsh", "unnecessary"],
    },
    # nouns/ noun phrases
    "NN": {
        "opportunity": ["opportunity", "chance"],
        "foul": ['foul (play)'],
        "goal": ["goal"]
    },

    # adverbials
    "ADVJ": {
        # negative
        'neg': ["harshly"]
    },

    ### VERBS
    "VBD": {
        "foul": ["fouled", "felled", "scythed down"],
        "goal": ["scored", "curled in", "put (in)", "hammered in", "drilled in", "slotted in"],
        "nogoal": ["missed", "shot wide"],
        "pass": ["played", "prodded", "passed"]
    },
    "VBDO": {
        "goal": ["curled the ball", "put the ball", "hammered the ball"],
        "nogoal": ["missed", "shot wide"]
    },
    "VBD-PASSIVE": {
        "foul": ["was $VBD.foul", "was sent to the ground", "was scythed down"],
    },
    "VBG": {
        "goal": ["scoring", "hammering in", "curling in", "slotting in", 'drilling in', 'putting in'],
        "foul": ["fouling", "felling", 'scything down', 'upending']
    },
    "VBGO": {
        "goal": ["scoring", "hammering the ball in", "curling the ball in", "slotting the ball in"]
    },
    "CONJ": {
        "contrastive": ["but", "however"]
    }
}


def _current_sentence(ctx):
    return ctx['sent']


def _last_sentence(ctx):
    return ctx['sentences'][ctx['sent_nr'] - 1]


at = {
    "MODIFIER-GOAL": {
        "true": [
            "almost",
            "nearly"
        ],
        "false": [
            "spectacularly"
        ]
    },
    "CONDITION": {
        "if": lambda: True,
        "true": [
            "and $PRONOUN $VBD.goal"
        ],
        "false": [
            "$CONJ.contrastive $PRONOUN $VBD.nogoal"
        ]
    }
}


def _is_contrastive_or_matchstart(ctx):
    if ctx['sent_nr'] == 0:
        return 'matchstart'
    # self last sentence in contrast to current sentence
    last_sentence = _last_sentence(ctx)
    current_sentence = _current_sentence(ctx)
    if last_sentence.event_type == current_sentence.event_type:
        if last_sentence.actor['team'] == current_sentence.actor['team']:
            return 'supportive'
        else:
            return 'contrastive'
    else:
        return "neutral"


# percent looks like if/then/else thing
percent = {
    "CONNECTIVE": {
        "VP": {
            "condition": _is_contrastive_or_matchstart,
            "matchstart": [
                "kicked the match off",
                "started the match",
                "set the tone for the match",
                "opened the action"
            ],
            "neutral": [
                "advanced the action"
            ],
            "contrastive": [
                "constituted a counter strike"
            ],

            "supportive": [
                "added more insult to the injury",
                "continued where $ACTORTEAM.name left off",
            ]
        },
        "IVP": {
            "condition": _is_contrastive_or_matchstart,
            "matchstart": [
                "kick the match off",
                "start the match",
                "set the tone for the match",
                "open the action"
            ],
            "neutral": [
                "advance the action"
            ],
            "contrastive": [
                "constitute a counter strike"
            ],

            "supportive": [
                "add more insult to the injury",
                "continue where [$ACTORTEAM.name|they] left off",
            ]
        },
        "ADVP": {
            "condition": _is_contrastive_or_matchstart,
            "matchstart": ['early in the game', 'as early as $TIME'],
            "supportive": ['to add on'],
            "neutral": ['later on', 'thereafter'],
            "contrastive": ['[decisively quickly|quickly|promptly] answering', 'with a [decisive|quick|prompt] answer']
        },
        "VBD>": {
            "condition": _is_contrastive_or_matchstart,
            "matchstart": [
                "The match started as",
                "After the kickoff",
                "The tone was set with the game just #sent.attributes.time minutes old, when",
                "The first $JJ.attention thing after the kick-off was, when"],
            "neutral": [
                "Then",
                "!MINDIFF minutes after that",
                "$RDM.S.any as"
            ],
            "contrastive": [
                "However",
                "$ACTORTEAM.name answered with a precise move , as",
                "[But|The|But the] $ACTORTEAM.name retaliated as",
                "$ACTORTEAM.name , however, came back when",
            ],

            "supportive": [
                "[To add|adding] insult to $NONACTORTEAM.name-pos-pre injury",
                "Further",
                "The onslaught (by $ACTORTEAM.name) continued, as "
            ]
        },
        "VBG>": {
            "condition": _is_contrastive_or_matchstart,
            "matchstart": ["The match started with"],
            "supportive": [
                "Further pressure (on the attack) [led to|resulted in]"
            ],
            "neutral": [
                "Things proceeded with",
                "$RDM.S.any seeing"
            ],
            'contrastive': ["Things changed ( , however , ) with"]
        }
    },
}


# percent = {
#     # preamble decider for a VBD-led VP
#     "PREAMBLE-VBD": {
#         # at the beginning of the sentence
#         "begin": {
#             "condition": lambda ctx: ctx['sent_nr'] == 0,
#             # put some starting comments
#             True: ["$BEGIN.VBD.matchstart"],
#             # whether to use a contrast or not
#             False: ["%CONTRAST-VBD"]
#         }
#     },
#
#     ,
#
#     "CONTRAST-VBD": {
#         "condition": _is_contrastive,
#         "contrastive": ["$BEGIN.VBD.contrastive"],
#         "supportive": ["$BEGIN.VBD.supportive"],
#         "neutral": ["$BEGIN.VBD.neutral"]
#     }
#
# }


# assuming we only have one action
def _vbx(template, event_type):
    """
    Gives the mode of the action verb (e.g. VBD, VBG, etc)
    """
    logger.debug(event_type)
    logger.debug(template)
    # select action verb. if no action verb, then select any first appearing verb
    vbx = next((x for x in template if x.startswith("$V") and f".{event_type}" in x), None)
    if not vbx:
        logger.debug("Template has no action verb!")
        try:
            vbx = next(x for x in template if ("VB") in x and f".{event_type}" in x)
        except StopIteration:
            first_vbg = next((i for i, x in enumerate(template) if x.endswith('ing')), -1)
            first_vbd = next((i for i, x in enumerate(template) if x.endswith('ed')), -1)
            if first_vbg > 0 and first_vbg > first_vbd:
                return "VBG"
            elif first_vbd > 0 and first_vbd > first_vbg:
                return "VBD"
            else:
                # TODO: hmmm
                return "VBD"
    logger.debug(f"VBX:{vbx}")
    vbx = re.findall(r'(VB.).*\.', vbx)[0]
    logger.debug(f"VBX:{vbx}")
    assert vbx.startswith('VB')
    return vbx


_possible_verb_forms = ("VBG", "VBD")
possible_contrastive = ["contrastive", 'supportive', 'neutral']




def _is_first_event_of_its_type(ctx):
    event_type = ctx['sent'].event_type
    return next(sent.sentence_nr for sent in ctx['sentences'] if sent.event_type == event_type) == ctx['sent_nr']


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
    "MINDIFF": (lambda ctx: ctx.sent.attributes['time'] - ctx.sentences[ctx.sent_nr - 1].attributes['time']),
    "OTHERTEAM": (
        lambda ctx: next(t.name for t in ctx.world.teams if t.id != ctx.sent.actor.team.id))

}
question_templates = {
    'direct': {
        "actor": {
            "goal": [
                "Who scored the #n th goal",
                "Who shot the #n th goal",
                "Who was the #n th goal scorer"
            ],
            "foul": [
                "Who committed the #n th foul",
                "Who fouled for the #n th time",
                "The #n th foul was committed by whom"
            ]
        },
        "distance": {
            "goal": [
                "From how far away was the #n th goal scored",
                "From how far away was the #n th goal shot",
                "The #n th goal was scored from how far (away)",
                "They scored the #n th goal from how far (away)",
            ]
        },
        "time": {
            "goal": [
                "The #n th goal was scored when",
                "When was the #n th goal scored",
                "The #n th goal was scored in what minute",
                "In what minute was the #n th goal scored",
                "When did they score the #n th goal"
            ]
        },
        "coactor": {
            "goal": [
                "Who assisted the #n th goal",
                "After whose pass was the #n th goal scored",
                "Who helped score the #n th goal",
            ],
            "foul": [
                "Who was fouled for the #n th time",
                "Who was fouled #n th"
            ]
        }
    },
    "overall": {
        "actor": {
            "goal": [
                "Who scored",
                "Who scored a goal",
                "Who shot a goal"
            ],
            "foul": [
                "Who committed a foul",
                "Who fouled"
            ]

        },
        "coactor": {
            "foul": [
                "Who was fouled",
                "They fouled whom"
            ]
        },
        "distance": {
            "goal": [
                "From how far away were goals scored",
                "From how far away were goals shot",
                "The goals were shot from how far (away)",
                "They scored the goals from how far (away)"
            ],
        }
    }
}
