import random
from collections import defaultdict
from copy import deepcopy
from typing import List, Union, Tuple, Callable

from quicklog import Loggable

from stresstest.passage.generate import Sentence


class S(List[str]):
    def __init__(self, iterable: List[str]):
        super().__init__([template.split() for template in iterable])

    def random(self) -> Tuple[List[str], int]:
        choice = random.randint(0, len(self) - 1)
        return deepcopy(self[choice]), choice


class YouIdiotException(Exception):
    ...


class QuestionTemplates(Loggable):
    def _create_templates(self):
        self.templates = {
            'direct': {
                "actor": {
                    "goal": S([
                        "Who scored the #n th goal",
                        "Who shot the #n th goal",
                        "Who was the #n th goal scorer"
                    ]),
                    "foul": S([
                        "Who was fouled for the #n th time",
                        "Who was fouled #n th"
                    ])
                },
                "distance": {
                    "goal": S([
                        "From how far away was the #n th goal scored",
                        "From how far away was the #n th goal shot",
                        "The #n th goal was scored from how far (away)",
                        "They scored the #n th goal from how far (away)",
                    ])
                },
                "time": {
                    "goal": S([
                        "The #n th goal was scored when",
                        "When was the #n th goal scored",
                        "The #n th goal was scored in what minute",
                        "In what minute was the #n th goal scored",
                        "When did they score the #n th goal"
                    ])
                },
                "coactor": {
                    "goal": S([
                        "Who assisted the #n th goal",
                        "After whose pass was the #n th goal scored",
                        "Who helped score the #n th goal",
                    ]),
                    "foul": S([
                        "Who committed the #n th foul",
                        "Who fouled for the #n th time",
                        "The #n th foul was committed by whom"
                    ])
                }
            },
            "overall": {
                "actor": {
                    "goal": S([
                        "Who scored",
                        "Who scored a goal",
                        "Who shot a goal"
                    ]),
                    "foul": S([
                        "Who was fouled",
                        "They fouled whom"
                    ])
                },
                "distance": {
                    "goal": S([
                        "From how far away were goals scored",
                        "From how far away were goals shot",
                        "The goals was show from how far (away)",
                        "They scored the goals from how far (away)"
                    ]),
                }
            }
        }

    def __init__(self):
        self._create_templates()

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
        return n

    def realise_question(self, q):
        self.logger.debug(q)
        template, template_nr = self.templates[q['type']][q['target']][q['action']].random()
        question_words = []
        template.reverse()
        stack = template
        while stack:
            self.logger.debug(stack)
            word = stack.pop()
            self.logger.debug(word)

            # alternative as in ()
            if word.startswith("(") and word.endswith(")"):
                # 50/50 whether to ignore it
                if random.choice([True, False]):
                    stack.append(word[1:-1])

                # if c.word.startswith("@"):
                ...
                # if/then/else
            # elif word.startswith("%"):
            #     branch = self._access_percent(word[1:])
            #     self.logger.debug(branch)
            #     self.logger.debug(200 * "X")
            #     result = branch['condition']()
            #     self.logger.debug(result)
            #     new_words, idx = branch[result].random()
            #     self.context['choices'].append(f"{word}.{idx}")
            #
            #     self.logger.debug(new_words)
            #     new_words = [str(w) for w in new_words[::-1]]
            #     stack.extend(new_words)
            # context sensitive
            elif word.startswith("#"):
                try:
                    new_word = q[word[1:]]
                except KeyError:
                    raise NotImplementedError(
                        f"{word} is not in question!")
                stack.append(str(new_word))
            # recursive template evaluation
            # elif word.startswith("$"):
            #     new_words, idx = self._access_dollar(word[1:]).random()
            #     new_words = [str(w) for w in new_words[::-1]]
            #     self.context['choices'].append(f"{word}.{idx}")
            #     stack.extend(new_words)
            # elif word.startswith("!"):
            #     # get and execute
            #     new_words = self._access_bang(word[1:])
            #     stack.extend(new_words[::-1])

            else:
                question_words.append(word)
        self.logger.debug(question_words)
        return " ".join(question_words) + "?", q['answer']
