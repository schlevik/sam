import random

sentences = {
    "goal": [
        "%PREAMBLE-VBD.begin $ACTOR $ACTORTEAM.name-pos-post $VBD.goal a ($JJ.positive) goal",

        "%PREAMBLE-VBD.begin $ACTOR $VBD.goal a ($JJ.positive) goal for $ACTORTEAM.name",

        "$ACTORTEAM.name-pos-pre player $ACTOR put an exclamation mark, $VBG.goal a ($JJ.positive) goal $DISTANCE.PP",

        "$ACTOR 's goal ( , [$RDM.VBG|RDM.NOVB] , ) arrived $TIME after !PRPS teammate $COACTOR 's $PASS-TYPE "
        "and [$RDM.CC-V.goal|$RDM.S.goal]",

        "$TIME a $PASS-TYPE fell to ($ACTORTEAM.name-pos-pre) $COACTOR in $POSITION.VERTICAL and $COREF-PLAYER "
        "swept $POSITION.HEIGHT to the $POSITION.BOX for $ACTOR to poke past the $GOALKEEPER",

        "A $JJ.positive $DISTANCE.JJ strike from $ACTOR [flying|homing] into $POSITION.GOAL past "
        "[the $GOALKEEPER|a helpess $GOALKEEPER] ($RDM.PP.goal) %PREAMBLE-VP.end",

        "$ACTOR , one of $ACTORTEAM.name-pos-pre better performers today, "
        "scored $TIME [$REASON.PP.goal| and $RDM.S.goal].",

        "$ACTOR scored $TIME when !PRP $REASON.CC-V.goal (and $REASON.CC-V.goal) "
        "before $VBG.goal the ball $POSITION.PP.GOAL .",

        "$ACTOR scored $TIME $VBG.goal the ball $POSITION.PP.GOAL "
        "after !PRP $REASON.CC-V.goal (and $REASON.CC-V.goal) ",

        "%PREAMBLE-VBD.begin the ball arrived [on|at] the $POSITION.BOX (at pace) and [$RDM.VBG] , $ACTOR $VBDO.goal "
        "(just) $POSITION.PP.GOAL (to leave the $GOALKEEPER with no chance)"
        # "%PREAMBLE-VBD.begin"
    ],
    "foul": [
        "%PREAMBLE-VBD.begin $COACTOR ($COACTORTEAM.name-pos-post) had gone down with $INJURY",
        "%PREAMBLE-VBD.begin ($COACTORTEAM.name-pos-pre) $COACTOR $VBD-PASSIVE.foul ($ADVJ.neg) by [$ACTOR ($ACTORTEAM.name-pos-post)|($COACTORTEAM.name-pos-pre) $ACTOR] ($TIME)",
        "%PREAMBLE-VBD.begin $ACTOR $VBD.foul ($COACTORTEAM.name-pos-pre) $COACTOR ($TIME)",
        "%PREAMBLE-VBD.begin $ACTOR $VBD.foul $COACTOR ($COACTORTEAM.name-pos-post)",
        "$RDM.S as $COACTOR was withdrawn $TIME with !PRPS $BODYPART in a brace following a ($JJ.negative) challenge from $ACTOR"
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
                 "a reward for !PRP hard work"],
        # inserted as a gerund
        "VBG": ["being !PRPS !RANDINT th of the season",
                "drawing attention from even !PRPS biggest sceptics",
                "following a $JJ.positive juggle"],
        "AWARD": ["highlight of the day",
                  "action of the match"],
        "PP": {
            "goal": ["for !PRPS !RANDINT th league goal of the season"]
        },
        # "BEGINNING": ["$ACTORTEAM.name did well to withstand the barrage"]
    },
    "REASON": {
        "PP": {
            "goal": ["after a run on $POSITION.VERTICAL"]  # TODO: on out wide / on the centre
        },
        "CC-V": {
            "any": ["dribbled !RANDINT metres (on $POSITION.VERTICAL)"],  # TODO: on out wide / on the centre
            "goal": ["ran !RANDINT metres",
                     "intercepted [$NONACTORTEAM.name-pos-pre goalkeeper's goal kick"
                     "| the goal kick of $NONACTORTEAM.name-pos-pre goal keeper]"]
        }
    },
    "GOALKEEPER": ["goalkeeper", "woman between the posts", "last line of defence"],
    "COREF-PLAYER": ["!PRP", "the player"],
    "POSITION": {
        "VERTICAL": ["the flank", "out wide", "the centre"],
        "BOX": ["near post", "far post", "penalty spot", "6-yard-area", "edge of the area"],
        "GOAL": [
            "the ($POSITION.HEIGHT) ($POSITION.LR) corner", "the middle of the goal", "the back of the net",
            "between the posts"
        ],
        "HEIGHT": ["lower", "upper"],
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
        "PP": ["from #sent.attributes.distance meters (away)"],
        "JJ": ["#sent.attributes.distance meters"],
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
    # stuff to put in the beginning of the sentence
    "BEGIN": {
        # VBD clause is following
        'VBD': {
            "matchstart": [
                "The match started as",
                "After the kickoff",
                "The tone was set with the game just #sent.attributes.time minutes old, when",
                "The first $JJ.attention thing after the kick-off was, when"],
            "neutral": [
                "Then",
                "!MINDIFF minutes after that"
            ],
            "contrastive": [
                "However",
                "[But|The] $ACTORTEAM.name retaliated as",
                "$ACTORTEAM.name , however, came back when",
            ],

            "supportive": [
                "To add insult to $NONACTORTEAM.name-pos-pre injury",
                "Further",
                "The onslaught (by $ACTORTEAM.name) continued, as "
            ]
        },
        'VBG': {
            "matchstart": ["The match started with"]}
    },
    "END": {
        # VBD clause is following
        'VP': {
            "matchstart": [
                "kicked the match off",
                "started the match",
                "set the tone for the match",
                "opened the action"],
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
        }
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
        "accurate": ["accurate", "pin-point", ],
        "risky": ["bold", "daring", "risky"],
        "attention": ["remarkable", "interesting"],
        "negative": ["bad", "harsh", "unnecessary"],
    },
    # adverbials
    "ADVJ": {
        # negative
        'neg': ["harshly"]
    },

    ### VERBS
    "VBD": {
        "foul": ["fouled", "felled"],
        "goal": ["scored", "curled in", "put (in)", "hammered"],
        "nogoal": ["missed", "shot wide"]
    },
    "VBDO": {
        "goal": ["curled the ball", "put the ball", "hammered the ball"],
        "nogoal": ["missed", "shot wide"]
    },
    "VBD-PASSIVE": {
        "foul": ["was $VBD.foul", "was sent to the ground"],
    },
    "VBG": {
        "goal": ["scoring", "hammering in", "curling in", "slotting in"]
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


def _is_contrastive(ctx):
    # self last sentence in contrast to current sentence
    last_sentence = _last_sentence(ctx)
    current_sentence = _current_sentence(ctx)
    if last_sentence.action == current_sentence.action:
        if last_sentence.actor['team'] == current_sentence.actor['team']:
            return 'supportive'
        else:
            return 'contrastive'
    else:
        return "neutral"


# percent looks like if/then/else thing
percent = {
    # preamble decider for a VBD-led VP
    "PREAMBLE-VBD": {
        # at the beginning of the sentence
        "begin": {
            "condition": lambda ctx: ctx['sent_nr'] == 0,
            # put some starting comments
            True: ["$BEGIN.VBD.matchstart"],
            # whether to use a contrast or not
            False: ["%CONTRAST-VBD"]
        }
    },
    "PREAMBLE-VP": {
        "end": {
            "condition": lambda ctx: ctx['sent_nr'] == 0,
            True: ["$END.VP.matchstart"],
            # whether to use a contrast or not
            False: ["%CONTRAST-VP"]
        }
    },
    "CONTRAST-VP": {
        "condition": _is_contrastive,
        "contrastive": ["$END.VP.contrastive"],
        "supportive": ["$END.VP.supportive"],
        "neutral": ["$END.VP.neutral"]
    },

    "CONTRAST-VBD": {
        "condition": _is_contrastive,
        "contrastive": ["$BEGIN.VBD.contrastive"],
        "supportive": ["$BEGIN.VBD.supportive"],
        "neutral": ["$BEGIN.VBD.neutral"]
    }

}

bang = {
    "RANDINT": (lambda ctx: random.randint(1, 15)),
    "PRPS": (lambda ctx: "her" if ctx['world']['gender'] == 'female' else "his"),
    "PRP": (lambda ctx: "she" if ctx['world']['gender'] == 'female' else "he"),
    # difference in time from last event
    # TODO: it's a tiny bit buggy because it doesn't register itself in ctx['visited'], ok for span-based though
    "MINDIFF": (lambda ctx: ctx['sent'].attributes['time'] - ctx['sent'].attributes['time']),
    "OTHERTEAM": (
        lambda ctx: next(t['name'] for t in ctx['world']['teams'] if t['id'] != ctx['sent'].actor['team']['id']))

}
templates = {
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
                "The goals was show from how far (away)",
                "They scored the goals from how far (away)"
            ],
        }
    }
}
