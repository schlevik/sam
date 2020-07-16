at = {
    "RB": {
        "goal": ['almost', 'nearly', 'all but']  # TODO: more
    },
    "MD": {
        "VBD": {
            "goal": ["$MD.neg"]
        },
        "VBG": {
            "goal": ["$MDG.neg"]
        },
        "VBI": {
            "goal": ["not"]
        }
    },
    # http://web.stanford.edu/group/csli_lnr/Lexical_Resources/simple-implicatives/simple-implicatives.prn
    # Verbs of Implicit Negation and their Complements in the History of English, Iyeiri, 2010
    "VB-neg-impl": {  # simple negation + (--)
        "VBD": {
            "goal": ["$MD.neg [manage|get|happen] to",
                     "$MD.neg succeed in",
                     "was [not|n't] [permitted|allowed] to"],
            "score": [

            ]
        },
        "VBG": {
            "goal": ["not [managing|getting|happening] to",  # + infinitive
                     #"$MDG.neg [manage|get|happen] to",
                     "not succeeding in",
                     #"$MDG.neg succeed in",  # + gerund
                     "[not being|having not been] [permitted|allowed] to", ]  # passive
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
                     "[was|has been] [prohibited|disallowed] to"]  # + inf | TODO  # (+-)
        },
        "VBG": {
            "goal": ['[failing|refusing] to',
                     "[being|having been] [refrained|refused|prohibited|prevented] from",  # + gerund
                     "[being|having been] prohibited to"  # + inf
                     ]
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

            ]
        },
        "VBG": {
            # not HAVE/USE OPPORTUNITY-OCCASION/ABILITY-ATTRIBUTE
            'goal': [
                "not [geting|having|finding] the [$NN.opportunity|$NN.attribute] to",
                "not [using|exploiting] the $NN.opportunity to",
                # not MEET OBLIGATION
                "not [meeting|fulfilling] the $NN.obligation to"
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
            ]
        }
    }

}