import random

from stresstest.classes import S

sentences = {
    "goal": S([
        "%PREAMBLE-VBD.begin $ACTOR $ACTORTEAM.name-pos-post $VBD.goal a ($JJ.positive) goal",
        "%PREAMBLE-VBD.begin $ACTOR $VBD.goal a ($JJ.positive) goal for $ACTORTEAM.name",
        "$ACTORTEAM.name-pos-pre player $ACTOR put an exclamation mark, $VBG.goal a ($JJ.positive) goal $DISTANCE.PP",
        "$ACTOR 's goal ($RDM.VBG) arrived $TIME after !PRPS teammate $COACTOR 's $PASS-TYPE and [$RDM.CC-V.goal|$RDM.S.goal]",
        "$TIME a $PASS-TYPE fell to ($ACTORTEAM.name-pos-pre) $COACTOR in $POSITION.VERTICAL and $COREF-PLAYER swept $POSITION.HEIGHT to the $POSITION.BOX for $ACTOR to poke past the $GOALKEEPER",
        "A $JJ.positive $DISTANCE.JJ strike from $ACTOR [flying|homing] into $POSITION.GOAL past [the $GOALKEEPER|a helpess $GOALKEEPER] ($RDM.PP.goal) %PREAMBLE-VP.end",
        "$ACTOR , one of $ACTORTEAM.name-pos-pre better performers today, scored $TIME [$REASON.PP.goal| and $RDM.S.goal].",
        "$ACTOR scored $TIME when !PRPS $REASON.CC-V.goal (and $REASON.CC-V.goal) before slotting in at $POSITION.GOAL ."
        # "%PREAMBLE-VBD.begin"
    ]),
    "foul": S([
        "%PREAMBLE-VBD.begin $COACTOR ($COACTORTEAM.name-pos-post) had gone down with $INJURY",
        "%PREAMBLE-VBD.begin ($COACTORTEAM.name-pos-pre) $COACTOR $VBD-PASSIVE.foul ($ADVJ.neg) by [$ACTOR ($ACTORTEAM.name-pos-post)|($COACTORTEAM.name-pos-pre) $ACTOR] ($TIME)",
        "%PREAMBLE-VBD.begin $ACTOR $VBD.foul ($COACTORTEAM.name-pos-pre) $COACTOR ($TIME)",
        "%PREAMBLE-VBD.begin $ACTOR $VBD.foul $COACTOR ($COACTORTEAM.name-pos-post)",
        "$RDM.S as $COACTOR was withdrawn $TIME with !PRPS $BODYPART in a brace following a ($JJ.negative) challenge from $ACTOR"
        # "If (only) #sent.actor $VBD.goal the penalty, "
        # "the score would be @CONDITION.then, otherwise it would "
        # "stay @CONDITION.else, @CONDITION.value"
    ])
}
dollar = {
    # just some random stuff that makes things look pretty
    "RDM": {
        "S": {
            "any": S(["the stadium went wild"]),
            "goal": S(["$RDM.PP.goal it was spectacular", "$RDM.S.any"])
        },
        "CC-V": {
            "goal": S(["ended !PRPS dry run of !RANDINT games",
                       "made the fans chant !PRPS name"])
        },
        # inserted as a gerund
        "VBG": S([", being !PRPS !RANDINT th of the season ,",
                  ", a contender for the $RDM.AWARD ,",
                  ", a reward for !PRP hard work , "
                  ", drawing attention from even !PRPS biggest sceptics ,"]),
        "AWARD": S(["highlight of the day",
                    "action of the match"]),
        "PP": {
            "goal": S(["for !PRPS !RANDINT th league goal of the season"])
        },
        # "BEGINNING": S(["$ACTORTEAM.name did well to withstand the barrage"])
    },
    "REASON": {
        "PP": {
            "goal": S(["after a run on $POSITION.VERTICAL"])
        },
        "CC-V": {
            "goal": S(["ran !RANDINT metres",
                       "intercepted $NONACTORTEAM.name $GOALKEEPER goal kick"])
        }
    },
    "GOALKEEPER": S(["goalkeeper", "woman between the posts", "last line of defence"]),
    "COREF-PLAYER": S(["!PRP", "the player"]),
    "POSITION": {
        "VERTICAL": S(["the flank", "out wide", "the center"]),
        "BOX": S(["near post", "far post", "penalty spot", "6-yard-area"]),
        "GOAL": S(["the ($POSITION.HEIGHT) ($POSITION.LR) corner", "the middle of the goal"]),
        "HEIGHT": S(["low", "high"]),
        "LR": S(["left", "right"])
    },
    "INJURY": S(["a (potential) ($BODYPART) injury"]),
    "BODYPART": S(["knee", "head", "shoulder", "arm", "hip", "leg", "ankle"]),
    # NP description of assist
    "PASS-TYPE": S(["($JJ.positive) pass", "($JJ.accurate) cross",
                    "($JJ.risky) through ball", "soft clearance", "stray ball"]),
    # NP distance description
    "DISTANCE": {
        "PP": S(["from #sent.attributes.distance meters (away)"]),
        "JJ": S(["#sent.attributes.distance meters"]),
    },

    # NP time description
    "TIME": S(["in minute #sent.attributes.time",
               "on the #sent.attributes.time th minute"]),
    # ACTOR ACCESSOR
    "ACTOR": S(["#sent.actor.first #sent.actor.last"]),
    # COACTOR ACCESSOR
    "COACTOR": S([
        "#sent.attributes.coactor.first #sent.attributes.coactor.last"]),
    "ACTORTEAM": {
        "name": S(['#sent.actor.team.name']),
        "name-pos-pre": S(["$ACTORTEAM.name 's"]),
        "name-pos-post": S(
            ["of $ACTORTEAM.name", ", a player of $ACTORTEAM.name ,"])
    },
    "NONACTORTEAM": {
        "name": S(["!OTHERTEAM"]),
        "name-pos-pre": S(["$NONACTORTEAM.name 's"]),
        "name-pos-post": S(
            ["of $NONACTORTEAM.name", ", a player of $NONACTORTEAM.name ,"])
    },
    # stuff to put in the beginning of the sentence
    "BEGIN": {
        # VBD clause is following
        'VBD': {
            "matchstart": S([
                "The match started as",
                "After the kickoff",
                "The tone was set with the game just #sent.attributes.time minutes old, when",
                "The first $JJ.attention thing after the kick-off was, when"]),
            "neutral": S([
                "Then",
                "!MINDIFF minutes after that"
            ]),
            "contrastive": S([
                "However",
                "[But|The] $ACTORTEAM.name retaliated as",
                "$ACTORTEAM.name , however, came back when",
            ]),

            "supportive": S([
                "To add insult to $NONACTORTEAM.name-pos-pre injury",
                "Further",
                "The onslaught (by $ACTORTEAM.name) continued, as "
            ])
        },
        'VBG': {
            "matchstart": S(["The match started with"])}
    },
    "END": {
        # VBD clause is following
        'VP': {
            "matchstart": S([
                "kicked the match off",
                "started the match",
                "set the tone for the match",
                "opened the action"]),
            "neutral": S([
                "advanced the action"
            ]),
            "contrastive": S([
                "constituted a counter strike"
            ]),

            "supportive": S([
                "added more insult to the injury",
                "continued where $ACTORTEAM.name left off",
            ])
        }
    },
    # variable resolution for the team of the action's coactor
    "COACTORTEAM": {
        # accessor of the team name
        "name": S(['#sent.attributes.coactor.team.name']),
        # possessive accessor before the actual co-actor
        "name-pos-pre": S(["$COACTORTEAM.name 's"]),
        # possessive accessor after the co-actor
        "name-pos-post": S(
            ["of $COACTORTEAM.name", "playing for $COACTORTEAM.name"])
    },
    # adjectives
    "JJ": {
        "positive": S(
            ["spectacular", "wonderful", "amazing", "stunning", "searing"]),
        "accurate": S(["accurate", "pin-point", ]),
        "risky": S(["bold", "daring", "risky"]),
        "attention": S(["remarkable", "interesting"]),
        "negative": S(["bad", "harsh", "unnecessary"]),
    },
    # adverbials
    "ADVJ": {
        # negative
        'neg': S(["harshly"])
    },

    ### VERBS
    "VBD": {
        "foul": S(["fouled", "felled"]),
        "goal": S(["scored", "curled in", "put (in)"]),
        "nogoal": S(["missed", "shot wide"])
    },
    "VBD-PASSIVE": {
        "foul": S(["was $VBD.foul", "was sent to the ground"]),
    },
    "VBG": {
        "goal": S(["scoring", "hammering in", "curling in"])
    },
    "CONJ": {
        "contrastive": S(["but", "however"])
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
            True: S(["$BEGIN.VBD.matchstart"]),
            # whether to use a contrast or not
            False: S(["%CONTRAST-VBD"])
        }
    },
    "PREAMBLE-VP": {
        "end": {
            "condition": lambda ctx: ctx['sent_nr'] == 0,
            True: S(["$END.VP.matchstart"]),
            # whether to use a contrast or not
            False: S(["%CONTRAST-VP"])
        }
    },
    "CONTRAST-VP": {
        "condition": _is_contrastive,
        "contrastive": S(["$END.VP.contrastive"]),
        "supportive": S(["$END.VP.supportive"]),
        "neutral": S(["$END.VP.neutral"])
    },

    "CONTRAST-VBD": {
        "condition": _is_contrastive,
        "contrastive": S(["$BEGIN.VBD.contrastive"]),
        "supportive": S(["$BEGIN.VBD.supportive"]),
        "neutral": S(["$BEGIN.VBD.neutral"])
    }

}

bang = {
    "RANDINT": (lambda ctx: random.randint(1, 15)),
    "PRPS": (lambda ctx: "her" if ctx['world']['gender'] == 'female' else "his"),
    "PRP": (lambda ctx: "she" if ctx['world']['gender'] == 'female' else "he"),
    # difference in time from last event
    "MINDIFF": (lambda ctx: ctx['sent'].attributes['time'] - ctx['sent'].attributes['time']),
    "OTHERTEAM": (
        lambda ctx: next(t['name'] for t in ctx['world']['teams'] if t['id'] != ctx['sent'].actor['team']['id']))

}
templates = {
    'direct': {
        "actor": {
            "goal": S([
                "Who scored the #n th goal",
                "Who shot the #n th goal",
                "Who was the #n th goal scorer"
            ]),
            "foul": S([
                "Who committed the #n th foul",
                "Who fouled for the #n th time",
                "The #n th foul was committed by whom"
            ])
        },
        "distance": {
            "goal": S([
                "From how far away was the #n th goal scored",
                "From how far away was the #n th goal shot",
                "The #n th goal was scored from how far (away)",
                "They scored the #n th goal from how far (away)",
            ])
        },
        "time": {
            "goal": S([
                "The #n th goal was scored when",
                "When was the #n th goal scored",
                "The #n th goal was scored in what minute",
                "In what minute was the #n th goal scored",
                "When did they score the #n th goal"
            ])
        },
        "coactor": {
            "goal": S([
                "Who assisted the #n th goal",
                "After whose pass was the #n th goal scored",
                "Who helped score the #n th goal",
            ]),
            "foul": S([
                "Who was fouled for the #n th time",
                "Who was fouled #n th"
            ])
        }
    },
    "overall": {
        "actor": {
            "goal": S([
                "Who scored",
                "Who scored a goal",
                "Who shot a goal"
            ]),
            "foul": S([
                "Who committed a foul",
                "Who fouled"
            ])

        },
        "coactor": {
            "foul": S([
                "Who was fouled",
                "They fouled whom"
            ])
        },
        "distance": {
            "goal": S([
                "From how far away were goals scored",
                "From how far away were goals shot",
                "The goals was show from how far (away)",
                "They scored the goals from how far (away)"
            ]),
        }
    }
}