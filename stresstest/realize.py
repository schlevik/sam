import inspect
import random
import textwrap
from collections import defaultdict
from typing import List

from loguru import logger

from stresstest.classes import S, YouIdiotException
from stresstest.generate import Sentence
from stresstest.resources.templates import percent, sentences, at, dollar, bang, templates as question_templates


class Realizer:
    # bang: dict  # what is bang?
    # dollar: dict  # what is dollar?
    # sentences: dict  # what is sentences?
    # at: dict  # what is at?
    # percent: dict  # what is percent?
    # question_templates: dict  # well that's sort of obvious

    def __init__(self, sentences=sentences, bang=bang, dollar=dollar, at=at, percent=percent,
                 question_templates=question_templates):
        self.context = None
        # TODO: move S([]) here
        self.bang = bang
        self.dollar = dollar
        self.sentences = sentences
        self.at = at
        self.percent = percent
        self.question_templates = question_templates

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
        return str(n(self.context)).split()

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
        logger.debug(f"{self.context['sentence_template']}")
        logger.debug(f"{self.context['choices']}")
        logger.error(msg)
        return YouIdiotException(msg)

    def realise_story(self, sentences: List[Sentence], world):
        self.context = dict()
        self.context['world'] = world
        self.context['sentences'] = sentences
        self.context['visits'] = defaultdict(list)
        realised = []
        for self.context['sent_nr'], self.context['sent'] in enumerate(sentences):
            logger.debug(self.context['sent'])
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
            logger.debug(f"Stack content: {self.context['stack']}")
            self.context['word'] = self.context['stack'].pop()
            logger.debug(f"Current word: {self.context['word']}")

            word = self.context['word']
            stack = self.context['stack']

            # optional as in ()
            if word.startswith("(") and word.endswith(")"):
                logger.debug("...Word is an option ()...")
                # 50/50 whether to ignore it
                if random.choice([True, False]):
                    new_words = word[1:-1].split()[::-1]
                    logger.debug(f"... new words: {new_words}")
                    stack.extend(new_words)
                else:
                    random.choice("Discarding!")
                # if c.word.startswith("@"):

            # alternative as in []
            elif word.startswith("[") and word.endswith("]"):
                logger.debug("...Word is an alternative []...")
                alternatives = word[1:-1].split("|")
                choice = random.choice(alternatives)
                logger.debug(f"...Choosing {choice}")
                stack.extend(choice.split()[::-1])

            # if/then/else
            elif word.startswith("%"):
                logger.debug("...Word is a condition %...")
                branch = self._access_percent(word[1:])
                logger.debug(
                    f"...Evaluating: {branch['condition'].__name__}\n{textwrap.dedent(inspect.getsource(branch['condition']))}")
                result = branch['condition'](ctx)
                logger.debug(f"with result: {result}")
                new_words, idx = branch[result].random()
                self.context['choices'].append(f"{word}.{idx}")

                logger.debug(f"... new words: {new_words}")
                new_words = [str(w) for w in new_words[::-1]]
                stack.extend(new_words)

            # accessing context
            elif word.startswith("#"):
                logger.debug("...Word is context #...")
                new_words = self._access_context(word[1:])

                logger.debug(f"... new words: {new_words}")
                stack.extend(new_words[::-1])

            # recursive template evaluation
            elif word.startswith("$"):
                logger.debug("...Word is template $...")
                try:
                    new_words, idx = self._access_dollar(word[1:]).random()
                except (KeyError, AttributeError) as _:
                    new_words, idx = self._access_dollar(word[1:] + ".any").random()

                new_words = new_words[::-1]
                self.context['choices'].append(f"{word}.{idx}")
                logger.debug(f"... new words: {new_words}")
                stack.extend(new_words)

            elif word.startswith("!"):

                logger.debug("...Word is a function !...")
                # get and execute
                new_words = self._access_bang(word[1:])
                logger.debug(f"... new words: {new_words}")
                stack.extend(new_words[::-1])

            else:
                ctx['realised'].append(self.context['word'])
            logger.debug("...done!")

        logger.debug(f"Final sentence is: {ctx['realised']}")

        return " ".join(ctx['realised'])

    def realise_question(self, q):
        logger.debug(f"Question: {q}")
        template, template_nr = self.question_templates[q['type']][q['target']][q['action']].random()
        question_words = []
        template.reverse()
        stack = template
        while stack:
            logger.debug(f"Current stack is: {stack}")
            word = stack.pop()
            logger.debug(word)

            # alternative as in ()
            if word.startswith("(") and word.endswith(")"):
                logger.debug("...Word is an option ()...")
                # 50/50 whether to ignore it
                if random.choice([True, False]):
                    new_words = word[1:-1].split()[::-1]
                    logger.debug(f"... new words: {new_words}")
                    stack.extend(new_words)
                else:
                    random.choice("Discarding!")

            # context access
            elif word.startswith("#"):
                try:
                    new_word = q[word[1:]]
                except KeyError:
                    raise NotImplementedError(f"{word} is not in question!")
                stack.append(str(new_word))

            else:
                question_words.append(word)
        logger.debug(question_words)
        return " ".join(question_words) + "?", q['answer']
