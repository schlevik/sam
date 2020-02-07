import random
from copy import deepcopy
from typing import List, Union

from pyhocon import ConfigTree, ConfigFactory
from quicklog import Loggable

from stresstest.classes import dotdict
from stresstest.passage.generate import Sentence


class S(List[str]):
    def __init__(self, iterable: List[str]):
        super().__init__([template.split() for template in iterable])

    def random(self):
        return deepcopy(random.choice(self))


# StringTree = DotDict[str, Union[S, DotDict[str, DotDict]]]


class Templates(Loggable):
    def _create_templates(self):
        self.sentences = S([
            "%CONTRAST $ACTOR of $ACTORTEAM $VBP.goal a ($JJ) goal",
            "$ACTORTEAM 's player $ACTOR put an exclamation mark, $VBPART.goal a ($JJ) goal from $DISTANCE",
            "$ACTOR was $VBP.foul ($ADVJNEG) by "
            "$COACTOR",

            #            "If (only) #sent.actor $VBP.goal the penalty, "
            #            "the score would be @CONDITION.then, otherwise it would "
            #            "stay @CONDITION.else, @CONDITION.value"
        ])
        self.dollar = {
            "DISTANCE": S(["#sent.attributes.distance meters (away)"]),
            "ACTOR": S(["#sent.actor.first #sent.actor.last"]),
            "COACTOR": S(["#sent.attributes.coactor.first #sent.attributes.coactor.last"]),
            "ACTORTEAM": S(['#sent.actor.team.name']),
            "JJ": S(["spectacular", "wonderful"]),
            "ADVJNEG": S(["harshly"]),
            "VBP": {
                "foul": S(["fouled", "felled"]),
                "goal": S(["scored", "curled in"]),
                "nogoal": S(["missed", "shot wide"])
            },
            "VBPART": {
                "goal": S(["scoring", "hammering in", "curling in"])
            },
            "CONJ": {
                "contrastive": S(["but", "however"])
            }
        }
        self.at = {
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
                    "and $PRONOUN $VBP.goal"
                ],
                "false": [
                    "$CONJ.contrastive $PRONOUN $VBP.nogoal"
                ]
            }
        }
        # percent looks like if/then/else thing
        self.percent = {
            "CONTRAST": {

                True: [
                    "However,",
                    "Things changed, as"
                ],
                False: [
                    "To capitalise, ",
                    "They began running away with it, as "
                ]
            }
        }

    def __init__(self):
        self._create_templates()

    def _access(self, word, target=None):
        n = target or self.context
        for k in word.split("."):
            try:
                n = n[k]
            except KeyError:
                n = getattr(n, k)
                if not n:
                    raise NotImplementedError()
        return n

    def realise_sentence(self, sentence: Sentence, world):
        # TODO: track action
        self.context = dotdict()
        self.context['world'] = world
        c = self.context
        # for now let's just assume all sentences cover all roles (eventually)
        c.sent = sentence
        template: List[str] = self.sentences.random()
        realised = []
        c.stack = template
        c.stack.reverse()
        while c.stack:
            self.logger.debug(c.stack)
            c.word = c.stack.pop()
            self.logger.debug(c.word)
            # for now only word
            if c.word.startswith("(") and c.word.endswith(")"):
                if random.choice([True, False]):
                    c.stack.append(c.word[1:-1])

                # if c.word.startswith("@"):
                ...
                # if/then/else
                # if c.word.startswith("%"):
                ...
            # context sensitive
            elif c.word.startswith("#"):
                try:
                    new_words = str(self._access(c.word[1:])).split()
                except KeyError:
                    self.logger.debug(f"{c}")
                    raise NotImplementedError(
                        f"{c.word[1:]} is not in context!")

                new_words.reverse()
                c.stack.extend(new_words)
            # recursive template evaluation
            elif c.word.startswith("$"):
                new_words = self._access(c.word[1:],
                                         target=self.dollar).random()
                new_words.reverse()
                c.stack.extend(new_words)
            else:
                realised.append(c.word)
        return " ".join(realised) + "."
