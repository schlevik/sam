question_templates = {
    'direct': {
        "actor": {
            "retrieval": {
                "goal": [
                    "Who scored the #n th goal",
                    "The #n th goal was scored by whom",
                    "Who shot the #n th goal",
                    "Who was the #n th goal scorer",
                    "The #n th goal was shot by whom"
                ],
                "foul": [
                    "Who committed the #n th foul",
                    "Who fouled for the #n th time",
                    "The #n th foul was committed by whom",
                    "Who fouled for the #n th time"
                ]
            },
            "retrieval-reverse": {
                "goal": [
                    "Who scored the #n th to last goal",
                    "Who shot the #n th to last goal",
                    "Who was the #n th to last goal scorer",
                    "The #n th to last was shot by whom"
                ],
                "foul": [
                    "Who committed the #n th to last foul",
                    "Who fouled for the #n th to last time",
                    "The #n th to last foul was committed by whom"
                ]
            },
            "bridge": {
                "goal": [
                    "Who scored (a goal) just after !bridge-impl",
                    "Who scored (a goal) just after !bridge-short",
                    "Who shot a goal right after !bridge-short",
                    "Right after !bridge-short , who scored",
                    "Right after !bridge-short , who shot a goal",
                    "After !bridge-short , who was the (next) goal scorer",
                    "After !bridge-short , who scored next",
                    "Who scored (a goal) just after !bridge-impl",
                    "Who was the goal scorer after !bridge-short",
                    "Who was the goal scorer after !bridge-impl"
                ],
            },
            "bridge-reverse": {
                "goal": [
                    "Who scored (a goal) (just) before !bridge-impl",
                    "Who scored (a goal) (just) before !bridge-short",
                    "Who shot a goal (right) before !bridge-short",
                    "Before !bridge-short , who was the goal scorer",
                    "Before !bridge-short , who scored",
                    "Who scored (a goal) (just) before !bridge-impl",
                    "Who was the goal scorer before !bridge-short",
                    "Who was the goal scorer before !bridge-impl"
                ],
            }
        },
        "distance": {
            "retrieval": {
                "goal": [
                    "From how far away was the #n th goal scored",
                    "From how far away did they score the #n th goal",
                    "From how far away was the #n th goal shot",
                    "From how far away did they shoot the #n th goal",
                    "The #n th goal was scored from how far (away)",
                    "They scored the #n th goal from how far (away)",
                ]
            },
            "retrieval-reverse": {
                "goal": [
                    "From how far away was the #n th to last goal scored",
                    "From how far away did they score the #n th goal",
                    "From how far away was the #n th to last goal shot",
                    "From how far away did they shoot the #n th to last goal",
                    "The #n th to last goal was scored from how far (away)",
                    "They scored the #n th to last goal from how far (away)",
                ]
            },
            "bridge": {
                "goal": [
                    "The goal after !bridge-short was scored from how far (away)",
                    "From what distance did they score after !bridge-short",
                    "From how far (away) did they score after !bridge-long",
                    "After !bridge-short , from how far away was the next goal scored",
                    "From how far (away) was the goal after !bridge-short scored",
                    "After !bridge-long , from how far away was the next goal scored",
                    "From what distance did they score after !bridge-long",
                    "From how far (away) was the goal after !bridge-long shot",
                    "They scored (the goal) after !bridge-short from how far (away)",
                ],
            },
            "bridge-reverse": {
                "goal": [
                    "From what distance did they score before !bridge-short",
                    "From what distance did they score before !bridge-long",
                    "From how far (away) did they score before !bridge-long",
                    "From how far (away) was the goal before !bridge-short scored",
                    "From how far (away) was the goal before !bridge-long shot",
                    "After !bridge-short , from how far away was the next goal scored",
                    "The goal before !bridge-short was scored from how far (away)",
                    "After !bridge-long , from how far away was the next goal scored",
                    "They scored (the goal) before !bridge-short from how far (away)",
                ],
            }
        },
        "time": {
            "retrieval": {
                "goal": [
                    "The #n th goal was scored when",
                    "The #n th goal was scored in what minute",
                    "When was the #n th goal scored",
                    "In what minute was the #n th goal scored",
                    "They scored the #n th goal when"
                    "When did they score the #n th goal"
                ]
            },
            "retrieval-reverse": {
                "goal": [
                    "When did they score the #n th to last goal"
                    "They scored the #n th to last goal when"
                    "In what minute was the #n th to last goal scored",
                    "The #n th to last goal was scored when",
                    "The #n th to last goal was scored in what minute",
                    "When was the #n th to last goal scored",
                ]
            },
            "bridge": {
                "goal": [
                    "When did they score the goal after !bridge-short",
                    "After !bridge-short , when was the next goal scored",
                    "After !bridge-long , when was the next goal scored",
                    "When did they shoot the goal after !bridge-short",
                    "When was the goal after !bridge-short scored",
                    "When did they score after !bridge-long",
                    "The goal after !bridge-short was scored when",
                    "When did they shoot the goal after !bridge-long",
                ],
            },
            "bridge-reverse": {
                "goal": [
                    "When did they score the goal before !bridge-short",
                    "When did they shoot the goal before !bridge-short",
                    "The goal before !bridge-short was scored when",
                    "When did they score before !bridge-long",
                    "When did they shoot the goal before !bridge-long",
                    "When was the goal before !bridge-short scored",
                    "Before !bridge-long , when was the next goal scored",
                    "Before !bridge-short , when was the next goal scored",
                ],
            }
        },
        "coactor": {
            "retrieval": {
                "goal": [
                    "Whose pass led to the #n th goal",
                    "Who got the assist for the #n th goal",
                    "Who helped score the #n th goal",
                    "Who assisted the #n th goal",
                    "After whose pass was the #n th goal scored",
                    "The #n th goal was scored after whose pass",
                    "Whose pass yielded the #n th goal",
                    
                ],
                "foul": [
                    "Who was fouled for the #n th time",
                    "Who was fouled #n th"
                ]
            },
            "retrieval-reverse": {
                "goal": [
                    "Whose pass yielded the #n th to last goal",
                    "Who assisted the #n th to last goal",
                    "After whose pass was the #n th goal to last scored",
                    "The #n th to last goal was scored after whose pass",
                    "Who helped score the #n th to last goal",
                    "Who got the assist for the #n th to last goal",
                    "Whose pass led to the #n th to last goal",
                ],
                "foul": [
                    "Who was fouled for the #n th to last time",
                    "Who was fouled #n th to last th"
                ]
            },
            "bridge": {
                "goal": [
                    "Who assisted (the goal) just after !bridge-short",
                    "Whose pass led to a goal (just) after !bridge-long",
                    "After !bridge-short , who helped score (the next goal)",
                    "Who assisted (the goal) just after !bridge-long",
                    "Who got the assist for the goal after !bridge-short",
                    "Who got the assist for the goal after !bridge-long",
                    "Whose pass led to a goal (just) after !bridge-short",
                    "Who helped score the goal after !bridge-short",
                    "After !bridge-long , who helped score (the next goal)",
                    "Who helped score the goal after !bridge-long",
                ],
            },
            "bridge-reverse": {
                "goal": [
                    "Whose pass led to a goal (just) before !bridge-long",
                    "Who assisted (the goal) just before !bridge-long",
                    "Whose pass led to a goal (just) before !bridge-short",
                    "Who assisted (the goal) just before !bridge-short",
                    "Who got the assist for the goal before !bridge-long",
                    "Who got the assist for the goal before !bridge-short",
                    "Who helped score the goal before !bridge-long",
                    "Before !bridge-long , who helped score (the previous goal)",
                    "Before !bridge-short , who helped score (the previous goal)",
                    "Who helped score the goal before !bridge-short"
                ],
            }
        }
    },
    "overall": {
        "actor": {
            'argmax-distance': {
                'goal': [
                    "Who shot the farthest goal",
                    "Who was the goal scorer with the farthest goal",
                    "The farthest goal came from whom",
                    "Who scored the farthest goal",
                    "Who scored from farthest away",
                    "The farthest goal was scored by whom",

                ]
            },
            'argmin-distance': {
                'goal': [
                    "Who shot the closest goal",
                    "who scored the shortest goal",
                    "who scored from the closest distance",
                    "Who was the goal scorer with the closest goal",
                    "The closest goal was shot by whom",
                    "The closest goal was scored by whom",
                ]
            },

            'comparison-time': {
                'goal': [
                    "Did !compare-alternatives score later in the match",
                    "Who scored later , !compare-alternatives",
                    "!compare-alternatives was the player who scored later",
                    "Did !compare-alternatives score later in the game",
                    "Did !compare-alternatives score later",
                    "Out of those two, who scored the last goal, !compare-alternatives",

                ]
            },
            'comparison-reverse-time': {
                'goal': [
                    "Who scored earlier , !compare-alternatives",
                    "Who scored earlier in the match , !compare-alternatives",
                    "Who scored earlier in the game , !compare-alternatives",
                    "Did !compare-alternatives score earlier",
                    "Out of those two, who scored the first goal, !compare-alternatives",
                    "!compare-alternatives was the one who scored earlier",

                ]
            },
            'comparison-distance': {
                'goal': [
                    "Who scored the farther goal , !compare-alternatives",
                    "Did !compare-alternatives score the farther goal",
                    "Who scored from farther away , !compare-alternatives",
                    "!compare-alternatives was the one who scored from farther away",
                    "In the game , who scored from farther away !compare-alternatives",
                    "In the game , who shot the farther goal !compare-alternatives"

                ]
            },
            'comparison-reverse-distance': {
                'goal': [
                    "Who scored the closer goal , !compare-alternatives",
                    "Did !compare-alternatives score the closer goal",
                    "Who scored from less far away , !compare-alternatives",
                    "Who scored the closer goal , !compare-alternatives",
                    "!compare-alternatives was the one who scored the closer goal",
                    "!compare-alternatives scored the closer goal"

                ]
            },
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
                                #"Whose pass led to the #n th goal",
                    #"Who got the assist for the #n th goal",
                    #"Who helped score the #n th goal",
                    #"Who assisted the #n th goal",
                    #"After whose pass was the #n th goal scored",
                    #"The #n th goal was scored after whose pass",
                    #"Whose pass yielded the #n th goal",
        "coactor": {
            'argmin-distance': {
                'goal': [
                    "The closest goal was achieved after whose pass",
                    "Who helped in scoring the closest goal",
                    "The closest goal was assisted by whom",
                    "Whose pass led to the closest goal",
                    "Who assisted the closest goal"
                    "Who helped to score the closest goal",
                    "Who helped to score the goal from the smallest distance",
                ]
            },
            'comparison-distance': {
                'goal': [
                    "Who helped to score the farther goal , !compare-alternatives",
                    "Who assisted the farther goal , !compare-alternatives",
                    "Did !compare-alternatives assist the farther goal",
                    "!compare-alternatives was the one who assisted the farther goal",
                    "!compare-alternatives was the one who helped score the farther goal",
                    "After whose pass was the farthest goal scored, !compare-alternatives",

                ]
            },
            'argmax-distance': {
                'goal': [
                    "The farthest goal was achieved after whose pass",
                    "Whose pass led to the farthest goal",
                    "The farthest goal was assisted by whom",
                    "Who assisted the farthest goal"
                    "Who helped to score the goal from the biggest distance",
                    "Who helped in scoring the farthest goal",
                    "Who helped to score the farthest goal",
                ]
            },
            "foul": [
                "Who was fouled",
                "They fouled whom"
            ]
        },
        "distance": {
            'argmax-distance': {
                'goal': [
                    "What is the biggest distance a goal was scored from",
                    "From how far away was the longest goal scored",
                    "The farthest goal was scored from how far",
                    "What was the farthest goal (of the game)",
                    "They scored the farthest goal from how far (away)",
                ]
            },
            'argmin-distance': {
                'goal': [
                    "The closest goal was scored from how far (away)",
                    "What was the closest goal (of the game)",
                    "What is the smallest distance a goal was scored from",
                    "They scored the closest goal from how far (away)",
                    "From how far away was the shortest goal scored",
                ]
            },
            "goal": [
                "From how far away were goals scored",
                "From how far away were goals shot",
                "The goals were shot from how far (away)",
                "They scored the goals from how far (away)"
            ],
        },
        "time": {
            'argmax-distance': {
                'goal': [
                    "The goal from farthest away was scored when",
                    "They scored the farthest goal when",
                    "In what minute was the farthest goal scored",
                    "When was the farthest goal scored",
                    "When did they score the farthest goal",
                    "At what time was the farthest goal scored",
                ]
            },
            'argmin-distance': {
                'goal': [
                    "At what time was the closest goal scored",
                    "They scored the closest goal when",
                    "When did they score the closest goal",
                    "The goal from the smallest distance was scored when"
                    "When was the closest goal scored",
                    "In what minute was the closest goal scored",
                ]
            },
        }
    }
}
