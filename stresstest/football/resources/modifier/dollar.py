dollar = {
    "JJ": {
        "position": {
            "height": ["lower", "upper"],
        },
        "distance": ["#sent.attributes.distance metres"],
        "positive":
            ["spectacular", "wonderful", "amazing", "stunning", "searing", "mesmerising"],
        "promising": ["promising", "auspicious", "promisingly looking", "auspiciously looking"],
        "accurate": ["accurate", "pin-point", ],
        "risky": ["bold", "daring", "risky"],
        "attention": ["remarkable", "interesting"],
        "negative": ["bad", "harsh", "unnecessary"],
        "important": ["important", "crucial"],
    },
    "NP": {
        "opportunity": ["chance", "opportunity", "possibility"],
        "obligation": ["commitment", "responsibility"],
        "attribute": ["grace", "guts", "nerve", "tenacity", "tenaciousness",
                      "boldness", "power", "strength", "courage", "audaciousness", "finesse"],

        "award": ["highlight of the day",
                  "action of the match"],

        "foul": ['foul (play)'],

        "shot": ['shot', 'drive', 'hammer', 'strike'],
        "goal": ["goal"],
        "goal-cause": ["a run $PP.position.vertical", 'a (decisive) counter-attack'],
        "kick-type": ['free kick', 'penalty'],
        "goalkeeper": ["goalkeeper", "woman between the posts", "last line of defence"],
        # TODO !person-gendered between the post
        "coref-player": ["!PRP", "the player"],

        "position": {
            "vertical": ["the flank", "out wide", "the centre"],
            "horizontal": ['the middle field', 'their own half', 'the attacking third'],
            "box": ["near post", "far post", "penalty spot", "6-yard-area", "edge of the area"],
            "goal": [
                "the ($JJ.position.height) ($NP.position.lr) corner", "the middle of the goal", "the back of the net",
                "between the posts",
            ],
            "height": ["low", "high"],
            "lr": ["left", "right"],
        },
        "injury": ["a (potential) ($NP.bodypart) injury"],
        "bodypart": ["knee", "head", "shoulder", "arm", "hip", "leg", "ankle"],
        # NP description of assist
        "pass-type": [
            "($JJ.positive) pass", "($JJ.accurate) cross", "($JJ.risky) through ball", "soft clearance", "stray ball"
        ],

        # actor accessor
        "actor": ["#sent.actor.first #sent.actor.last"],
        # coactor accessor
        "coactor": [
            "#sent.attributes.coactor.first #sent.attributes.coactor.last"
        ],

        "team": {
            # We don't need coactor team for the events we have now, because we know
            # which team coactors come from for each event...
            "actor": ['#sent.actor.team.name'],
            "actor-possessive": ["$NP.team.actor 's"],
            "actor-possessive-post": ["a player of $NP.team.actor "],
            "nonactor": ["!OTHERTEAM"],
            "nonactor-possessive": ["$NP.team.nonactor 's"],
            "nonactor-possessive-post": ["a player of $NP.team.nonactor ,"],
        },
    },
    "VP": {
        "VBD": {
            "attention": ['put an exclamation mark', 'became the talking point of the day',
                          'attracted (lots of) attention'],
            "attention-crowd": [
                # "ended !PRPS dry run of !RANDINT games",
                # "made the fans chant !PRPS name"
                "made the fans [roar|scream|gasp]",
                "made the crowd go wild",
                "[caused a stir|generated buzz] in the stand"

            ],
            "foul-elaboration-coref": [
                "(had only) just showed !PRPS reckless edge",
            ],
            "foul-elaboration": [
                "disappointed",
                "was overzealous"
            ],
            "any-cause": ["[ran|dribbled] !RANDINT metres ($PP.position.vertical)"],
            "goal-cause": [
                "intercepted [$NP.team.nonactor goalkeeper's goal kick"
                "| the goal kick of $PP.team.nonactor-possessive goal keeper]",
                "$VP.VBD.any-cause",
                "stole the ball from the defence",
                "won a tackle [$PP.position.vertical|in $NP.position.horizontal|]"],

            "foul": ["fouled", "felled", "scythed down"],
            "foul-passive": ["was $VP.VBD.foul", "was sent to the ground", "was scythed down"],
            "shoot": ["shot",  "put", 
                      "hammered", "drilled","curled",
                      ],
            "score": ["scored", 
                      "slotted in"
                      ],
            "nogoal": ["missed", "shot wide"],
            "pass": ["played", "prodded", "passed"],
            "modifier": [
                "@MD.VBD.goal @VB-pol-rev.VBD.goal @VP-pol-rev.VBD.goal @VB-neg-impl.VBD.goal @VP-neg-impl.VBD.goal"],
            "modifier-nonactor": ["@MD.VBD.goal-nonactor @VB-pol-rev.VBD.goal-nonactor "
                                  "@VP-pol-rev.VBD.goal-nonactor @VB-neg-impl.VBD.goal-nonactor @VP-neg-impl.VBD.goal-nonactor"],
        },
        "VBG": {
            "shoot": ["shooting", "curling", "putting", "hammering", "drilling"],
            "score": ["scoring", 'slotting in', 'hitting'],
            "foul": ["fouling", "felling", 'scything down', 'upending'],
            "foul-effect": ["disappointing [the crowd|everyone|] with a $JJ.negative action"],
            "goal-effect": [
                # "being $RDM.NOVB",
                "drawing attention from everyone around",
                "following a $JJ.positive juggle",
            ],
            "actor-possessive-post": ["playing for $NP.team.actor ,"],
            "nonactor-possessive-post": ["playing for $NP.team.nonactor ,"],
            'modifier': [
                "@MD.VBG.goal @VB-pol-rev.VBG.goal @VP-pol-rev.VBG.goal @VB-neg-impl.VBG.goal @VP-neg-impl.VBG.goal"],
            'modifier-nonactor': [
                "@MD.VBG.goal-nonactor @VB-pol-rev.VBG.goal-nonactor @VP-pol-rev.VBG.goal-nonactor "
                "@VB-neg-impl.VBG.goal-nonactor @VP-neg-impl.VBG.goal-nonactor"
            ]
        },
        "VBI": {
            "modifier": [
                "@MD.VBI.goal @VB-pol-rev.VBI.goal @VP-pol-rev.VBI.goal @VB-neg-impl.VBI.goal @VP-neg-impl.VBI.goal"],
            "modifier-nonactor": [
                "@MD.VBI.goal-nonactor @VB-pol-rev.VBI.goal-nonactor "
                "@VP-pol-rev.VBI.goal-nonactor @VB-neg-impl.VBI.goal-nonactor @VP-neg-impl.VBI.goal-nonactor"
            ],
        }
    },
    "PP": {
        
        "goal-cause": ["after $NP.goal-cause"],
        "goal-cause-coactor": ["after (!PRPS teammate) $NP.coactor 's $NP.pass-type"],
        'goal-effect': ["for !PRPS !RANDINT th league goal of the season"],
        "foul-effect": ["for a [$JJ.promising|$JJ.attention] free-kick $NP.opportunity ",
                        "(for $NP.team.nonactor)"
                        ],
        "foul-cause-coref": ["for a [$JJ.promising|$JJ.attention] free-kick $NP.opportunity for !PRPS opponents"],
        "foul-effect-coref": ["for which !PRP was booked", "for which she saw a [red|yellow] card"],
        "position": {
            "vertical": ["on the flank", "out wide", "in the centre"],
            "goal": ["in $NP.position.goal", "under the bar",
                     "off the [post|bar] and [in|just into] $NP.position.goal"],
        },
        "distance": ["from #sent.attributes.distance metres (away)"],
        "time": ["in minute #sent.attributes.time",
                 "in the #sent.attributes.time th minute"],
        "team": {
            "actor-possessive": ["of $NP.team.actor"],
            "nonactor-possessive": ["of $NP.team.nonactor", ]
        }
    },
    "S": {
        "attention-crowd": ["the stadium went wild"],
        "goal-cause-coactor": ["$NP.coactor (inadvertently) $VP.VBD.pass the ball into !PRPS path (following $NP.goal-cause)"],
    },

    # adverbials
    "RB": {
        # negative
        'neg': ["harshly"]
    },

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
