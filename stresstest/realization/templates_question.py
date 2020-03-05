import random
import re
from collections import defaultdict
from copy import deepcopy
from typing import List, Union, Tuple, Callable

from quicklog import Loggable

from stresstest.passage.generate import Sentence

pattern = re.compile(r"([^(\[\]]\S*|\(.+?\)|\[.+?\])\s*")


class S(List[str]):
    def __init__(self, iterable: List[str]):
        super().__init__([pattern.findall(template) for template in iterable])

    def random(self) -> Tuple[List[str], int]:
        choice = random.randint(0, len(self) - 1)
        return deepcopy(self[choice]), choice


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
                        "Who committed the #n th foul",
                        "Who fouled for the #n th time",
                        "The #n th foul was committed by whom"
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
                        "Who was fouled for the #n th time",
                        "Who was fouled #n th"
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
                        "Who committed a foul",
                        "Who fouled"
                    ])

                },
                "coactor": {
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

    def realise_question(self, q):
        self.logger.debug(f"Question: {q}")
        template, template_nr = self.templates[q['type']][q['target']][q['action']].random()
        question_words = []
        template.reverse()
        stack = template
        while stack:
            self.logger.debug(f"Current stack is: {stack}")
            word = stack.pop()
            self.logger.debug(word)

            # alternative as in ()
            if word.startswith("(") and word.endswith(")"):
                self.logger.debug("...Word is an option ()...")
                # 50/50 whether to ignore it
                if random.choice([True, False]):
                    new_words = word[1:-1].split()[::-1]
                    self.logger.debug(f"... new words: {new_words}")
                    stack.extend(new_words)
                else:
                    random.choice("Discarding!")

            # context access
            elif word.startswith("#"):
                try:
                    new_word = q[word[1:]]
                except KeyError:
                    raise NotImplementedError(
                        f"{word} is not in question!")
                stack.append(str(new_word))

            else:
                question_words.append(word)
        self.logger.debug(question_words)
        return " ".join(question_words) + "?", q['answer']
