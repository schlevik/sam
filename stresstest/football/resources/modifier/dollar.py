dollar = {
    # just some random stuff that makes things look pretty
    "NP": {

    },
    "VP": {
        "attention": ['put an exclamation mark', 'became the talking point of the day', 'attracted (lots of) attention']
    },


    "MODIFIER": {
        "VBG": ["@MD.VBG.goal @VB-pol-rev.VBG.goal @VP-pol-rev.VBG.goal @VB-neg-impl.VBG.goal @VP-neg-impl.VBG.goal"],
        "VBD": ["@MD.VBD.goal @VB-pol-rev.VBD.goal @VP-pol-rev.VBD.goal @VB-neg-impl.VBD.goal @VP-neg-impl.VBD.goal"],
        "VBI": ["@MD.VBI.goal @VB-pol-rev.VBI.goal @VP-pol-rev.VBI.goal @VB-neg-impl.VBI.goal @VP-neg-impl.VBI.goal"],
        "nonactor": {
            "VBD": [
                "@MD.VBD.goal-nonactor @VB-pol-rev.VBD.goal-nonactor "
                "@VP-pol-rev.VBD.goal-nonactor @VB-neg-impl.VBD.goal-nonactor @VP-neg-impl.VBD.goal-nonactor"],
            "VBG": [
                "@MD.VBG.goal-nonactor @VB-pol-rev.VBG.goal-nonactor @VP-pol-rev.VBG.goal-nonactor "
                "@VB-neg-impl.VBG.goal-nonactor @VP-neg-impl.VBG.goal-nonactor"],
            "VBI": [
                "@MD.VBI.goal-nonactor @VB-pol-rev.VBI.goal-nonactor "
                "@VP-pol-rev.VBI.goal-nonactor @VB-neg-impl.VBI.goal-nonactor @VP-neg-impl.VBI.goal-nonactor"],

        }
    },
    "RDM": {
        "S": {
            "any": ["the stadium went wild"],
            "goal": [
                # "$RDM.PP.goal it was spectacular",
                "$RDM.S.any"
            ]
        },
        "CC-V": {
            "goal": [
                # "ended !PRPS dry run of !RANDINT games",
                # "made the fans chant !PRPS name"
                "made the fans roar"
            ]
        },
        # TODO: no verb, need to look up how we call it
        "NOVB": [""],  # ["a contender for the $RDM.AWARD",
        # "a reward for !PRPS hard work",
        # "!PRPS !RANDINT th goal [of the season|for the club]"],
        # inserted as a gerund
        "VBG": {
            "foul": ["disappointing (the crowd) with a $JJ.negative action"],
            "goal": [
                # "being $RDM.NOVB",
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
                     "[for $NONACTORTEAM.name|for !PRPS opponents|]"
                     'for which !PRP was booked'],
            "goal": ["for !PRPS !RANDINT th league goal of the season"]
        },
        # "BEGINNING": ["$ACTORTEAM.name did well to withstand the barrage"]
    },
    "REASON": {
        "S": {
            "goal": ["$COACTOR (inadvertently) $VBD.pass the ball into !PRPS path (following $REASON.NP)"]
        },

        "NP": ["a run $POSITION.VERTICAL.PP", 'a (decisive) counter-attack'],

        "PP": {
            "goal": ["after $REASON.NP"]
        },
        "CC-V": {
            "any": ["[ran|dribbled] !RANDINT metres ($POSITION.VERTICAL.PP)",
                    ],  # TODO: on out wide / on the centre
            "goal": [
                "intercepted [$NONACTORTEAM.name-pos-pre goalkeeper's goal kick"
                "| the goal kick of $NONACTORTEAM.name-pos-pre goal keeper]",
                "$REASON.CC-V.any"]
        }
    },

    "GOALKEEPER": ["goalkeeper", "woman between the posts", "last line of defence"],
    "COREF-PLAYER": ["!PRP", "the player"],
    "POSITION": {
        "VERTICAL": {
            "PP": ["on the flank", "out wide", "in the centre"]
        },
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
        "opportunity": ["chance", "opportunity", "occasion", "possibility"],
        "attribute": ["grace", "guts", "nerve", "tenacity", "tenaciousness",
                      "boldness", "power", "strength", "courage", "audaciousness", "finesse"],
        "obligation": ["commitment", "task", " responsibility"],
        "foul": ['foul (play)'],
        "goal": ["goal"],
        "shot": ['shot', 'drive', 'hammer', 'strike']
    },

    # adverbials
    "ADVJ": {
        # negative
        'neg': ["harshly"]
    },

    # verbs
    "VBD": {
        "foul": ["fouled", "felled", "scythed down"],
        "shoot": ['shot', "curled", "put", "hammered", "drilled"],
        "score": ["scored", "hit", "slotted in"],
        "nogoal": ["missed", "shot wide"],
        "pass": ["played", "prodded", "passed"]
    },
    # "VBDO": {
    #     "goal": ["curled the ball", "put the ball", "hammered the ball"],
    #     "nogoal": ["missed", "shot wide"]
    # },
    "VBD-PASSIVE": {
        "foul": ["was $VBD.foul", "was sent to the ground", "was scythed down"],
    },
    "VBG": {
        "shoot": ["shooting", "curling", "putting", "hammering", "drilling"],
        "score": ["scoring", 'slotting in', 'hitting'],
        "foul": ["fouling", "felling", 'scything down', 'upending']
    },
    # "VBGO": {
    #     "goal": ["scoring", "hammering the ball in", "curling the ball in", "slotting the ball in"]
    # },
    "CONJ": {
        "contrastive": ["but", "however"]
    },
    "MD": {
        "neg": ["[would|could|did] [not|n't]"]
    },
    "MDG": {
        "neg": ['not being able to', 'being unable to']
    }

}
