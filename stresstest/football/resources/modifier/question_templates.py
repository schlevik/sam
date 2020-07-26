question_templates = {
    'direct': {
        "actor": {
            "retrieval": {
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
            "retrieval-reverse": {
                "goal": [
                    "Who scored the #n th to last goal",
                    "Who shot the #n th to last goal",
                    "Who was the #n th to last goal scorer"
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
                    "From how far away was the #n th goal shot",
                    "The #n th goal was scored from how far (away)",
                    "They scored the #n th goal from how far (away)",
                ]
            },
            "retrieval-reverse": {
                "goal": [
                    "From how far away was the #n th to last goal scored",
                    "From how far away was the #n th to last goal shot",
                    "The #n th to last goal was scored from how far (away)",
                    "They scored the #n th to last goal from how far (away)",
                ]
            },
            "bridge": {
                "goal": [
                    "The goal after !bridge-short was scored from how far (away)",
                    "After !bridge-long , from how far away was the next goal scored",
                    "After !bridge-short , from how far away was the next goal scored",
                    "From how far (away) was the goal after !bridge-short scored"
                ],
            },
            "bridge-reverse": {
                "goal": [
                    "The goal before !bridge-short was scored from how far (away)",
                    "Before !bridge-long , from how far away was the next goal scored",
                    "Before !bridge-short , from how far away was the next goal scored",
                    "From how far (away) was the goal before !bridge-short scored"
                ],
            }
        },
        "time": {
            "retrieval": {
                "goal": [
                    "The #n th goal was scored when",
                    "When was the #n th goal scored",
                    "The #n th goal was scored in what minute",
                    "In what minute was the #n th goal scored",
                    "When did they score the #n th goal"
                ]
            },
            "retrieval-reverse": {
                "goal": [
                    "The #n th to last goal was scored when",
                    "When was the #n th to last goal scored",
                    "The #n th to last goal was scored in what minute",
                    "In what minute was the #n th to last goal scored",
                    "When did they score the #n th to last goal"
                ]
            },
            "bridge": {
                "goal": [
                    "The goal after !bridge-short was scored when",
                    "After !bridge-long , when was the next goal scored",
                    "After !bridge-short , when was the next goal scored",
                    "When was the goal after !bridge-short scored",
                ],
            },
            "bridge-reverse": {
                "goal": [
                    "The goal before !bridge-short was scored when",
                    "Before !bridge-long , when was the next goal scored",
                    "Before !bridge-short , when was the next goal scored",
                    "When was the goal before !bridge-short scored",
                ],
            }
        },
        "coactor": {
            "retrieval": {
                "goal": [
                    "Who assisted the #n th goal",
                    "After whose pass was the #n th goal scored",
                    "Who helped score the #n th goal",
                ],
                "foul": [
                    "Who was fouled for the #n th time",
                    "Who was fouled #n th"
                ]
            },
            "retrieval-reverse": {
                "goal": [
                    "Who assisted the #n th to last goal",
                    "After whose pass was the #n th to last scored",
                    "Who helped score the #n th to last goal",
                ],
                "foul": [
                    "Who was fouled for the #n th to last time",
                    "Who was fouled #n th to last th"
                ]
            },
            "bridge": {
                "goal": [
                    "Who assisted (the goal) just after !bridge-short",
                    "Who assisted (the goal) just after !bridge-long",
                    "Whose pass led to a goal (just) after !bridge-short",
                    "Whose pass led to a goal (just) after !bridge-long",
                    "After !bridge-short , who scored (the next goal)",
                    "After !bridge-long , who scored (the next goal)",
                    "Who helped score the goal after !bridge-long",
                    "Who helped score the goal after !bridge-short"
                ],
            },
            "bridge-reverse": {
                "goal": [
                    "Who assisted (the goal) just before !bridge-short",
                    "Who assisted (the goal) just before !bridge-long",
                    "Whose pass led to a goal (just) before !bridge-short",
                    "Whose pass led to a goal (just) before !bridge-long",
                    "After !bridge-short , who before (the next goal)",
                    "After !bridge-long , who before (the next goal)",
                    "Who helped score the goal before !bridge-long",
                    "Who helped score the goal before !bridge-short"
                ],
            }
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
