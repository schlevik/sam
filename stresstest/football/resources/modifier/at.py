at = {
    "RB": {
        "goal": ['almost', 'nearly', 'all but']  # TODO: more
    },
    "MD": {
        "VBD": {
            "goal": ["$MD.neg"],
            "goal-nonactor": ["[did|would] [not|n't]"]
        },
        "VBG": {
            "goal": ["$MDG.neg"],
            "goal-nonactor": ["not"]
        },
        "VBI": {
            "goal": ["not"],
            "goal-nonactor": ['not']
        }
    },
    # http://web.stanford.edu/group/csli_lnr/Lexical_Resources/simple-implicatives/simple-implicatives.prn
    # Verbs of Implicit Negation and their Complements in the History of English, Iyeiri, 2010
    "VB-neg-impl": {  # simple negation + (--)
        "VBD": {
            "goal": ["$MD.neg [manage|get|happen] to",
                     "$MD.neg succeed in",
                     "was [not|n't] [permitted|allowed] to"],
            "goal-nonactor": [
                "[did|would] [not|n't] [get|happen] to",
                'was [blocked|[prevented] from'
            ]
        },
        "VBI": {
            "goal": ["not [manage|get|happen] to",
                     "not succeed in",
                     "not be [permitted|allowed] to"],
            "goal-nonactor": [
                "not [get|happen] to",
                'not be [blocked|[prevented] from'
            ]
        },
        "VBG": {
            "goal": ["not [managing|getting|happening] to",  # + infinitive
                     # "$MDG.neg [manage|get|happen] to",
                     "not succeeding in",
                     # "$MDG.neg succeed in",  # + gerund
                     "[not being|having not been] [permitted|allowed] to", ],  # passive
            "goal-nonactor": [
                "[not] [getting|happening] to",
                'being [blocked|[prevented] from'
            ]
        },
        # 'VBD-passive': {
        #     "goal": []  # negation + (--)
        #
        # },
        # 'VBG-passive': {
        #     "goal": [
        #         # + inf
        #     ]
        # }
    },
    "VB-pol-rev": {  # simple (+-)
        "VBD": {
            "goal": ['[failed|refused] to',
                     "[was|has been] [refrained|refused|prohibited|prevented|hindered] from",  # + gerund
                     "[was|has been] [prohibited|disallowed] to"],  # + inf | TODO  # (+-)
            "goal-nonactor": ["failed to",
                              "[was|has been] [prevented|hindered] from"]
        },
        "VBG": {
            "goal": ['[failing|refusing] to',
                     "[being|having been] [refrained|refused|prohibited|prevented] from",  # + gerund
                     "[being|having been] prohibited to",  # + inf
                     ],
            "goal-nonactor": ["failing to",
                              "[being|having been] [prevented|hindered] from"]
        },
        "VBI": {
            "goal": ['[fail|refuse] to',
                     "be [refrained|refused|prohibited|prevented|hindered] from",  # + gerund
                     "be [prohibited|disallowed] to"],  # + inf | TODO  # (+-)
            "goal-nonactor": ["fail to",
                              "be [prevented|hindered] from"]
        },
        # 'VBD-passive': {
        #   "goal": [

        #  ]
        # },
        # 'VBG-passive': {
        #     "goal": [
        #
        #     ]
        # }
    },
    # https://web.stanford.edu/group/csli_lnr/Lexical_Resources/phrasal-implicatives/ImplicativeTemplates.pdf
    # Verbs of Implicit Negation and their Complements in the History of English, Iyeiri, 2010
    # compound negation + (--)
    "VP-neg-impl": {
        # not HAVE/USE OPPORTUNITY-OCCASION/ABILITY-ATTRIBUTE
        "VBD": {
            'goal': [
                "[would|did] [n't|not] [get|have] the [$NN.opportunity|$NN.attribute] to",
                "$MD.neg find the [$NN.opportunity|$NN.attribute] to",
                "$MD.neg [use|exploit] the $NN.opportunity to",
                # not MEET OBLIGATION
                "$MD.neg [meet|fulfill] the $NN.obligation to"
            ],
            'goal-nonactor': [
                "[would|did] [n't|not] [get|have] the [chance|possibility] to"
            ]
        },
        "VBG": {
            # not HAVE/USE OPPORTUNITY-OCCASION/ABILITY-ATTRIBUTE
            'goal': [
                "not [geting|having|finding] the [$NN.opportunity|$NN.attribute] to",
                "not [using|exploiting] the $NN.opportunity to",
                # not MEET OBLIGATION
                "not [meeting|fulfilling] the $NN.obligation to"
            ],
            'goal-nonactor': [
                "not [getting|having] the [chance|possibility] to"
            ]
        },
        "VBI": {
            'goal': [
                "not [get|have] the [$NN.opportunity|$NN.attribute] to",
                "not find the [$NN.opportunity|$NN.attribute] to",
                "not [use|exploit] the $NN.opportunity to",
                # not MEET OBLIGATION
                "not [meet|fulfill] the $NN.obligation to"
            ],
            'goal-nonactor': [
                "not [get|have] the [chance|possibility] to"
            ]
        }

    },
    # compound (+-)
    "VP-pol-rev": {
        "VBD": {
            "goal": [
                # LACK/FAIL/WASTE OPPORTUNITY-OCCASION
                '[missed|lost|wasted|gave up|threw away|squandered|neglected] the $NN.opportunity to',
                # LACK ATTRIBUTE/ABILITY
                'lacked the $NN.attribute to',
                'lost the nerve to',  # TODO: let native check
                # FAIL OBLIGATION
                'neglected the $NN.obligation to',  # TODO: let native check
                # USE ASSET: empty, i don't see how this fits
                'was denied the $NN.opportunity to'  # passive
            ],
            "goal-nonactor": [
                'was denied the $NN.opportunity to',
                'missed the [chance|possibility] to'
            ]
        },
        "VBG": {
            "goal": [
                # LACK/FAIL/WASTE OPPORTUNITY-OCCASION
                '[missing|losing|wasting|giving up|throwing away|squandering|neglecting] the $NN.opportunity to',
                # LACK ATTRIBUTE/ABILITY
                'lacking the $NN.attribute to',
                'losing the nerve to',  # TODO: let native check
                # FAIL OBLIGATION
                'neglecting the $NN.obligation to',  # TODO: let native check

                # PASSIVE VOICE
                '[being|having been] denied the $NN.opportunity to'
                # USE ASSET: empty, i don't see how this fits
                # he buy a cake
                # he lacked the strength to buy a cake => not he buy a cake
                # he used the money to buy a cake => ?not he buy a cake
            ],
            "goal-nonactor": [
                'being denied the $NN.opportunity to',
                'missing the [chance|possibility] to'
            ]
        },
        "VBI": {
            "goal": [
                # LACK/FAIL/WASTE OPPORTUNITY-OCCASION
                '[miss|lose|waste|give up|throw away|squander|neglect] the $NN.opportunity to',
                # LACK ATTRIBUTE/ABILITY
                'lack the $NN.attribute to',
                'lose the nerve to',  # TODO: let native check
                # FAIL OBLIGATION
                'neglect the $NN.obligation to',  # TODO: let native check

                # PASSIVE VOICE
                'be denied the $NN.opportunity to'
                # USE ASSET: empty, i don't see how this fits
                # he buy a cake
                # he lacked the strength to buy a cake => not he buy a cake
                # he used the money to buy a cake => ?not he buy a cake
            ],
            "goal-nonactor": [
                'be denied the $NN.opportunity to',
                'miss the [chance|possibility] to'
            ]
        }
    }

}
