def _current_sentence(ctx):
    return ctx['sent']


def _last_sentence(ctx):
    return ctx['sentences'][ctx['sent_nr'] - 1]


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
                "start (off) the match",
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