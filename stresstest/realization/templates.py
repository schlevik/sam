import random
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple
import inspect
from quicklog import Loggable
import re
import textwrap

from stresstest.passage.generate import Sentence

pattern = re.compile(r"([^(\[\]]\S*|\(.+?\)|\[.+?\])\s*")


class S(List[str]):
    def __init__(self, iterable: List[str]):
        super().__init__([pattern.findall(template) for template in iterable])

    def random(self) -> Tuple[str, int]:
        choice = random.randint(0, len(self) - 1)
        return deepcopy(self[choice]), choice


class YouIdiotException(Exception):
    ...


class TemplateLogic(Loggable, ABC):
    bang = {}
    dollar = {}
    sentences = {}
    at = {}
    percent = {}

    @abstractmethod
    def _create_templates(self):
        ...

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
        if isinstance(e, AttributeError) and "object has no attribute 'random'" in str(e):
            msg = f"{self.context['word']} is not a leaf path and template doesn't provide .any"
        elif isinstance(e, TypeError) and "list indices must be integers or slices, not str" in str(e):
            msg = ""
        elif isinstance(e, KeyError) and "dict object has no attribute" in str(e):
            msg = f"{self.context['word']} is not a valid template path!"
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

        # select template and the chosen number (for tracking purposes)
        template, template_nr = self.sentences[ctx['sent'].action].random()

        # set chosen template
        self.context['sentence_template'] = f"{ctx['sent'].action}.{template_nr}"

        # initialise context context
        self.context['realised'] = []
        self.context['choices'] = []
        self.context['stack'] = template
        self.context['stack'].reverse()

        # while there's something on the stack
        while self.context['stack']:
            self.logger.debug(f"Stack content: {self.context['stack']}")
            self.context['word'] = self.context['stack'].pop()
            self.logger.debug(f"Current word: {self.context['word']}")

            word = self.context['word']
            stack = self.context['stack']

            # optional as in ()
            if word.startswith("(") and word.endswith(")"):
                self.logger.debug("...Word is an option ()...")
                # 50/50 whether to ignore it
                if random.choice([True, False]):
                    new_words = word[1:-1].split()[::-1]
                    self.logger.debug(f"... new words: {new_words}")
                    stack.extend(new_words)
                else:
                    random.choice("Discarding!")
                # if c.word.startswith("@"):

            # alternative as in []
            elif word.startswith("[") and word.endswith("]"):
                self.logger.debug("...Word is an alternative []...")
                alternatives = word[1:-1].split("|")
                choice = random.choice(alternatives)
                self.logger.debug(f"...Choosing {choice}")
                stack.extend(choice.split()[::-1])

            # if/then/else
            elif word.startswith("%"):
                self.logger.debug("...Word is a condition %...")
                branch = self._access_percent(word[1:])
                self.logger.debug(f"...Evaluating: {branch['condition'].__name__}\n{textwrap.dedent(inspect.getsource(branch['condition']))}")
                result = branch['condition']()
                self.logger.debug(f"with result: {result}")
                new_words, idx = branch[result].random()
                self.context['choices'].append(f"{word}.{idx}")

                self.logger.debug(f"... new words: {new_words}")
                new_words = [str(w) for w in new_words[::-1]]
                stack.extend(new_words)

            # accessing context
            elif word.startswith("#"):
                self.logger.debug("...Word is context #...")
                new_words = self._access_context(word[1:])

                self.logger.debug(f"... new words: {new_words}")
                stack.extend(new_words[::-1])

            # recursive template evaluation
            elif word.startswith("$"):
                self.logger.debug("...Word is template $...")
                try:
                    new_words, idx = self._access_dollar(word[1:]).random()
                except (KeyError, AttributeError) as _:
                    new_words, idx = self._access_dollar(word[1:] + ".any").random()

                new_words = new_words[::-1]
                self.context['choices'].append(f"{word}.{idx}")
                self.logger.debug(f"... new words: {new_words}")
                stack.extend(new_words)

            elif word.startswith("!"):

                self.logger.debug("...Word is a function !...")
                # get and execute
                new_words = self._access_bang(word[1:])
                self.logger.debug(f"... new words: {new_words}")
                stack.extend(new_words[::-1])

            else:
                ctx['realised'].append(self.context['word'])
            self.logger.debug("...done!")

        self.logger.debug(f"Final sentence is: {ctx['realised']}")

        return " ".join(ctx['realised'])


class Templates(TemplateLogic):
    def _create_templates(self):
        self.sentences = {
            "goal": S([
                # "%PREAMBLE-VBD.begin $ACTOR $ACTORTEAM.name-pos-post $VBD.goal a ($JJ.positive) goal",
                # "%PREAMBLE-VBD.begin $ACTOR $VBD.goal a ($JJ.positive) goal for $ACTORTEAM.name",
                # "$ACTORTEAM.name-pos-pre player $ACTOR put an exclamation mark, $VBG.goal a ($JJ.positive) goal $DISTANCE.PP",
                # "$ACTOR 's goal ($RDM.VBG) arrived $TIME after !PRPS teammate $COACTOR 's $PASS-TYPE and [$RDM.CC-V.goal|$RDM.S.goal]",
                # "$TIME a $PASS-TYPE fell to ($ACTORTEAM.name-pos-pre) $COACTOR in $POSITION.VERTICAL and $COREF-PLAYER swept $POSITION.HEIGHT to the $POSITION.BOX for $ACTOR to poke past the $GOALKEEPER",
                # "A $JJ.positive $DISTANCE.JJ strike from $ACTOR [flying|homing] into $POSITION.GOAL past [the $GOALKEEPER|a helpess $GOALKEEPER] ($RDM.PP.goal) %PREAMBLE-VP.end",
                # "$ACTOR , one of $ACTORTEAM.name-pos-pre better performers today, scored $TIME [$REASON|$RDM.S.goal].",
                # "$ACTOR scored $TIME when !PRPS $REASON.CC-V.goal (and $REASON.CC-V.goal) before slotting in at $POSITION.GOAL ."
                "%PREAMBLE-VBD.begin"
            ]),
            "foul": S([
                "%PREAMBLE-VBD.begin $COACTOR ($COACTORTEAM.name-pos-post) had gone down with $INJURY",
                "%PREAMBLE-VBD.begin ($COACTORTEAM.name-pos-pre) $COACTOR $VBD-PASSIVE.foul ($ADVJ.neg) by [$ACTOR ($ACTORTEAM.name-pos-post)|($COACTORTEAM.name-pos-pre) $ACTOR] ($TIME)",
                "%PREAMBLE-VBD.begin $ACTOR $VBD.foul ($COACTORTEAM.name-pos-pre) $COACTOR ($TIME)",
                "%PREAMBLE-VBD.begin $ACTOR $VBD.foul $COACTOR ($COACTORTEAM.name-pos-post)",
                "$RDM.S as $COACTOR was withdrawn $TIME with !PRPS $BODYPART in a brace following a ($JJ.negative) challenge from $ACTOR"
                # "If (only) #sent.actor $VBD.goal the penalty, "
                # "the score would be @CONDITION.then, otherwise it would "
                # "stay @CONDITION.else, @CONDITION.value"
            ])
        }
        self.dollar = {
            # just some random stuff that makes things look pretty
            "RDM": {
                "S": {
                    "any": S(["the stadium went wild"]),
                    "goal": S(["$RDM.PP.goal it was spectacular", "$RDM.S.any"])
                },
                "CC-V": {
                    "goal": S(["ended !PRPS dry run of !RANDINT games",
                               "made the fans chant !PRPS name"])
                },
                # inserted as a gerund
                "VBG": S([", being !PRPS !RANDINT th of the season ,",
                          ", a contender for the $RDM.AWARD ,",
                          ", a reward for !PRP hard work , "
                          ", drawing attention from even !PRPS biggest sceptics ,"]),
                "AWARD": S(["highlight of the day",
                            "action of the match"]),
                "PP": {
                    "goal": S(["for !PRPS !RANDINT th league goal of the season"])
                },
                # "BEGINNING": S(["$ACTORTEAM.name did well to withstand the barrage"])
            },
            "REASON": {
                "PP": {
                    "goal": S(["after a run on $POSITION.VERTICAL"])
                },
                "CC-V": {
                    "goal": S(["ran !RANDINT metres",
                               "intercepted $NONACTORTEAM.name $GOALKEEPER goal kick"])
                }
            },
            "GOALKEEPER": S(["goalkeeper", "woman between the posts", "last line of defence"]),
            "COREF-PLAYER": S(["!PRP", "the player"]),
            "POSITION": {
                "VERTICAL": S(["the flank", "out wide", "the center"]),
                "BOX": S(["near post", "far post", "penalty spot", "6-yard-area"]),
                "GOAL": S(["the ($POSITION.HEIGHT) ($POSITION.LR) corner", "the middle of the goal"]),
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
            "NONACTORTEAM": {
                "name": S(["!OTHERTEAM"]),
                "name-pos-pre": S(["$NONACTORTEAM.name 's"]),
                "name-pos-post": S(
                    ["of $NONACTORTEAM.name", ", a player of $NONACTORTEAM.name ,"])
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
                        "!MINDIFF minutes after that"
                    ]),
                    "contrastive": S([
                        "However",
                        "[But|The] $ACTORTEAM.name retaliated as",
                        "$ACTORTEAM.name , however, came back when",
                    ]),

                    "supportive": S([
                        "To add insult to $NONACTORTEAM.name-pos-pre injury",
                        "Further",
                        "The onslaught (by $ACTORTEAM.name) continued, as "
                    ])
                },
                'VBG': {
                    "matchstart": S(["The match started with"])}
            },
            "END": {
                # VBD clause is following
                'VP': {
                    "matchstart": S([
                        "kicked the match off",
                        "started the match",
                        "set the tone for the match",
                        "opened the action"]),
                    "neutral": S([
                        "advanced the action"
                    ]),
                    "contrastive": S([
                        "constituted a counter strike"
                    ]),

                    "supportive": S([
                        "added more insult to the injury",
                        "continued where $ACTORTEAM.name left off",
                    ])
                }
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
                "attention": S(["remarkable", "interesting"]),
                "negative": S(["bad", "harsh", "unnecessary"]),
            },
            # adverbials
            "ADVJ": {
                # negative
                'neg': S(["harshly"])
            },

            ### VERBS
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
            "PREAMBLE-VP": {
                "end": {
                    "condition": lambda: self.context['sent_nr'] == 0,
                    True: S(["$END.VP.matchstart"]),
                    # whether to use a contrast or not
                    False: S(["%CONTRAST-VP"])
                }
            },
            "CONTRAST-VP": {
                "condition": self._is_contrastive,
                "contrastive": S(["$END.VP.contrastive"]),
                "supportive": S(["$END.VP.supportive"]),
                "neutral": S(["$END.VP.neutral"])
            },

            "CONTRAST-VBD": {
                "condition": self._is_contrastive,
                "contrastive": S(["$BEGIN.VBD.contrastive"]),
                "supportive": S(["$BEGIN.VBD.supportive"]),
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
            "MINDIFF": (lambda: self._current_sentence().attributes['time'] - self._last_sentence().attributes['time']),
            "OTHERTEAM": (lambda: next(t['name'] for t in self.context['world']['teams'] if t['id'] != self._current_sentence().actor['team']['id']))

        }

    def _current_sentence(self):
        return self.context['sent']

    def _last_sentence(self):
        return self.context['sentences'][self.context['sent_nr'] - 1]

    def _is_contrastive(self):
        # self last sentence in contrast to current sentence
        last_sentence = self._last_sentence()
        current_sentence = self._current_sentence()
        if last_sentence.action == current_sentence.action:
            if last_sentence.actor['team'] == current_sentence.actor['team']:
                return 'supportive'
            else:
                return 'contrastive'
        else:
            return "neutral"
