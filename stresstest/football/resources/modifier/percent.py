from stresstest.classes import Context, Event


def _current_sentence(ctx: Context) -> Event:
    return ctx.sent


def _last_sentence(ctx: Context) -> Event:
    return ctx.sentences[ctx.sent_nr - 1]


def _is_contrastive_or_matchstart(ctx: Context):
    if ctx.sent_nr == 0:
        return 'matchstart'
    # self last sentence in contrast to current sentence
    last_sentence = _last_sentence(ctx)
    current_sentence = _current_sentence(ctx)
    if last_sentence.event_type == current_sentence.event_type:
        if last_sentence.actor.team == current_sentence.actor.team:
            return 'supportive'
        else:
            return 'contrastive'
    else:
        return "neutral"


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
                "continued where [$NP.team.actor|they] left off",
            ]
        },
        "IVP": {
            "condition": _is_contrastive_or_matchstart,
            "matchstart": [
                "kick the match off",
                "start off the match",
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
                "continue where [$NP.team.actor|they] left off",
            ]
        },
        "ADVP": {
            "condition": _is_contrastive_or_matchstart,
            "matchstart": ['early in the game', 'early on'],
            "supportive": ['to add on'],
            "neutral": ['later on', 'thereafter'],
            "contrastive": ['[decisively quickly|quickly|promptly] answering', 'with a [decisive|quick|prompt] answer']
        },
        "VBD>": {
            "condition": _is_contrastive_or_matchstart,
            "matchstart": [
                "The match started as",
                "After the kickoff",
                "The tone was set with the game still being young , when",
                "The first $JJ.attention thing after the kick-off was , when"],
            "neutral": [
                "Then",
                "!MINDIFF minutes after that",
                "$S.attention-crowd as"
            ],
            "contrastive": [
                "However",
                "$NP.team.actor answered with a precise move , as",
                "[But|The|But the] $NP.team.actor retaliated as",
                "$NP.team.actor , however, came back when",
            ],

            "supportive": [
                "[To add|adding] insult to $NP.team.actor-possessive injury",
                "Further",
                "The onslaught (by $NP.team.actor) continued, as "
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
                "$S.attention-crowd seeing"
            ],
            'contrastive': ["Things changed ( , however , ) with"]
        }
    },
}
