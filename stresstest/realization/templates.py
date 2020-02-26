import random
from collections import defaultdict
from copy import deepcopy
from typing import List, Union, Tuple, Callable

from quicklog import Loggable

from stresstest.passage.generate import Sentence


class S(List[str]):
    def __init__(self, iterable: List[str]):
        super().__init__([template.split() for template in iterable])

    def random(self) -> Tuple[str, int]:
        choice = random.randint(0, len(self) - 1)
        return deepcopy(self[choice]), choice


class YouIdiotException(Exception):
    ...


class Templates(Loggable):
    def _create_templates(self):
        self.sentences = {
            "goal": S([
                "%PREAMBLE-VBD.begin $ACTOR ($ACTORTEAM.name-pos-post) $VBD.goal a ($JJ.positive) goal",
                "$ACTORTEAM.name-pos-pre player $ACTOR put an exclamation mark, $VBG.goal a ($JJ.positive) goal $DISTANCE.PP",
                "$ACTOR 's goal ($RANDOMSTUFF.VBG) arrived $TIME after !PRP teammates $COACTOR 's $PASS-TYPE and $RANDOMSTUFF.CC",
                "On $TIME a $PASS-TYPE fell to $COACTOR $POSITION.VERTICAL and $COREF-PLAYER swept (low) to the $POSITION.BOX for $ACTOR to poke past $GOALKEEPER",
                "A $JJ.positive $DISTANCE.JJ strike from $ACTOR flying into the corner past a helpless $GOALKEEPER $RANDOMSTUFF.PP"
                #            "If (only) #sent.actor $VBD.goal the penalty, "
                #            "the score would be @CONDITION.then, otherwise it would "
                #            "stay @CONDITION.else, @CONDITION.value"
            ]),
            "foul": S([
                "%PREAMBLE-VBD.begin $ACTOR had gone down with $INJURY",
                "$ACTOR $VBD-PASSIVE.foul ($ADVJ.neg) by $COACTOR",
                "$COACTOR $VBD.foul ($ACTORTEAM.name-pos-pre) $ACTOR",
                "$COACTOR $VBD.foul $ACTOR ($ACTORTEAM.name-pos-post)"
                #            "If (only) #sent.actor $VBD.goal the penalty, "
                #            "the score would be @CONDITION.then, otherwise it would "
                #            "stay @CONDITION.else, @CONDITION.value"
            ])
        }
        self.dollar = {
            # just some random stuff that makes things look pretty
            "RANDOMSTUFF": {
                "CC": S(["ended !PRPS dry run of !RANDINT games",
                         "the stadium went wild over it"]),
                # inserted as a gerund
                "VBG": S([", being !PRPS !RANDINT th of the season,",
                          "a contender for the $RANDOMSTUFF.AWARD",
                          "drawing attention from even !PRPS biggest sceptics"]),
                "AWARD": S(["highlight of the day",
                            "action of the match"]),
                "PP": S(["for !PRPS !RANDOMINT th league goal of the season"])
            },
            "GOALKEEPER": S(["the goalkeeper", "the woman between the posts", "the last line of defence"]),
            "COREF-PLAYER": S(["!PRP", "the player"]),
            "POSITION": {
                "VERTICAL": S(["on the flank", "out wide", "in the center"]),
                "BOX": S(["near post", "far post", "penalty spot", "6-yard-area"]),
                "GOAL": S(["into the ($HIGHT) ($LR) corner", "(straight) into the middle of the goal"]),
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
                        "After !MINDIFF minutes"
                    ])
                },
                'VBG': {
                    "matchstart": S(["The match started with"])}
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
                "attention": S(["remarkable", "interesting"])
            },
            # adverbials
            "ADVJ": {
                # negative
                'neg': S(["harshly"])
            },
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
                    "and $PRONOUN $VBD.goal"
                ],
                "false": [
                    "$CONJ.contrastive $PRONOUN $VBD.nogoal"
                ]
            }
        }
        # percent looks like if/then/else thing
        self.percent = {
            # preamble decider for a VBD-led VP
            "PREAMBLE-VBD": {
                # at the beginning of the sentence
                "begin": {
                    "condition": lambda: self.context['sent_nr'] == 0,
                    # put some starting comments
                    True: S(["$BEGIN.VBD.matchstart"]),
                    # whether to use a contrast or not
                    False: S(["%CONTRAST-VBD"])
                }
            },

            "CONTRAST-VBD": {
                "condition": self._is_contrastive,
                "contrastive": "",
                "supportive": "",
                "neutral": S(["$BEGIN.VBD.neutral"])
            }

        }

        self.bang = {
            "RANDINT": (lambda: random.randint(1, 15)),
            "PRPS": (lambda: "her" if self.context['world'][
                                             'gender'] == 'female' else "his"),
            "PRP": (lambda: "she" if self.context['world'][
                                         'gender'] == 'female' else "he"),
            # difference in time from last event
            "MINDIFF": (lambda: self.context['sent'].attributes['time'] -
                                self.context['sentences'][
                                    self.context['sent_nr'] - 1].attributes[
                                    'time']),
            "NATIONALITY": (lambda: random.choice(["German", "Korean", "Japanese", "French", "Spaniard"])),

        }

    def _is_contrastive(self):
        # self last sentence in contrast to current sentence
        # todo
        return "neutral"

    def __init__(self):
        self._create_templates()
        self.context = None

    def _access_context(self, word: str) -> List[str]:
        n = self.context
        if word.startswith('sent'):
            self.context['visits'][self.context['sent_nr']].append(word)
        for k in word.split("."):
            try:
                n = n[k]
            except KeyError:
                n = getattr(n, k)
                if not n:
                    raise NotImplementedError()
        return str(n).split()

    def _access_percent(self, word) -> dict:
        n = self.percent
        for k in word.split("."):
            try:
                n = n[k]
            except KeyError:
                n = getattr(n, k)
                if not n:
                    raise NotImplementedError
        return n

    def _access_bang(self, word) -> List[str]:
        n = self.bang
        for k in word.split("."):
            try:
                n = n[k]
            except KeyError:
                n = getattr(n, k)
                if not n:
                    raise NotImplementedError
        return str(n()).split()

    def _access_dollar(self, word) -> S:
        n = self.dollar
        for k in word.split("."):
            try:
                n = n[k]
            except KeyError:
                n = getattr(n, k)
                if not n:
                    raise NotImplementedError
            except TypeError:
                print(word)
                raise YouIdiotException()
        return n

    def with_feedback(self, e: Exception):
        if isinstance(e,
                      AttributeError) and "object has no attribute 'random'" in str(
            e):
            msg = f"{self.context['word']} is not a leaf path!"
        elif isinstance(e,
                        TypeError) and "list indices must be integers or slices, not str" in str(
            e):
            msg = ""

        else:
            msg = "And i don't even know what's wrong!"
        self.logger.debug(f"{self.context['sentence_template']}")
        self.logger.debug(f"{self.context['choices']}")
        self.logger.error(msg)
        return YouIdiotException(msg)

    def realise_story(self, sentences: List[Sentence], world):
        self.context = dict()
        self.context['world'] = world
        self.context['sentences'] = sentences
        self.context['visits'] = defaultdict(list)
        realised = []
        for self.context['sent_nr'], self.context['sent'] in enumerate(
                sentences):
            self.logger.debug(self.context['sent'])
            try:
                sent = self.realise_sentence()
            except Exception as e:
                raise self.with_feedback(e)
            realised.append(sent)
        return '\n'.join(realised) + ".", self.context['visits']

    def realise_sentence(self):
        ctx = self.context
        template, template_nr = self.sentences[ctx['sent'].action].random()
        self.context[
            'sentence_template'] = f"{ctx['sent'].action}.{template_nr}"
        self.context['realised'] = []
        self.context['choices'] = []
        self.context['stack'] = template
        self.context['stack'].reverse()
        while self.context['stack']:
            self.logger.debug(self.context['stack'])
            self.context['word'] = self.context['stack'].pop()
            self.logger.debug(self.context['word'])

            word = self.context['word']
            stack = self.context['stack']

            # alternative as in ()
            if word.startswith("(") and word.endswith(")"):
                # 50/50 whether to ignore it
                if random.choice([True, False]):
                    stack.append(word[1:-1])

                # if c.word.startswith("@"):
                ...
                # if/then/else
            elif word.startswith("%"):
                branch = self._access_percent(word[1:])
                self.logger.debug(branch)
                self.logger.debug(200 * "X")
                result = branch['condition']()
                self.logger.debug(result)
                new_words, idx = branch[result].random()
                self.context['choices'].append(f"{word}.{idx}")

                self.logger.debug(new_words)
                new_words = [str(w) for w in new_words[::-1]]
                stack.extend(new_words)
            # context sensitive
            elif word.startswith("#"):
                try:
                    new_words = self._access_context(word[1:])
                except KeyError:
                    raise NotImplementedError(
                        f"{self.context['word'][1:]} is not in context!")
                # new_words = [w for w in new_words[::-1]]
                stack.extend(new_words[::-1])
            # recursive template evaluation
            elif word.startswith("$"):
                new_words, idx = self._access_dollar(word[1:]).random()
                new_words = [str(w) for w in new_words[::-1]]
                self.context['choices'].append(f"{word}.{idx}")
                stack.extend(new_words)
            elif word.startswith("!"):
                # get and execute
                new_words = self._access_bang(word[1:])
                stack.extend(new_words[::-1])

            else:
                ctx['realised'].append(self.context['word'])
        self.logger.debug(ctx['realised'])
        return " ".join(ctx['realised'])
