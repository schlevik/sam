import inspect
import random
import textwrap
from collections import defaultdict
from itertools import count
from typing import List, Callable, Optional, Dict, Tuple, Union

from loguru import logger

from stresstest.classes import S, YouIdiotException, F
from stresstest.generate import Sentence
from stresstest.resources.templates import percent, sentences, at, dollar, bang, templates as question_templates


def process_templates(templates, allow_conditions=False) -> dict:
    processed = {}
    for k, v in templates.items():
        if k == 'condition':
            if not allow_conditions:
                raise ValueError("Conditions not allowed!")
            assert isinstance(v, Callable)
        elif isinstance(v, dict):
            v = process_templates(v, allow_conditions)
        elif isinstance(v, list):
            v = S(v)
        elif isinstance(v, type) and issubclass(v, F):
            logger.debug(f"templates[{k}] = {v} is instance of F")
            v = v()
            logger.debug(v.options)
            logger.debug(v.number)
        elif isinstance(v, Callable):
            v = F.make(v)
        elif isinstance(v, Tuple):
            v = F.make(*v)
        else:
            raise ValueError(f"Template values can only be lists or dicts! (was: {v}: {type(v)})")
        processed[k] = v
    return processed


class Realizer:

    def __init__(self, sentences=sentences, bang=bang, dollar=dollar, at=at, percent=percent,
                 question_templates=question_templates, validate=True, unique_sentences=True):
        self.unique_sentences = unique_sentences
        logger.debug("Creating new Realizer...")
        self.context = None
        self.bang = process_templates(bang)
        self.dollar = process_templates(dollar)
        self.sentences = process_templates(sentences)
        self.at = at
        self.percent = process_templates(percent, True)
        self.question_templates = process_templates(question_templates)
        if validate:
            logger.debug("Validating templates...")
            self.validate(self.dollar, 'dollar')
            self.validate(self.sentences, 'sentences')
            self.validate(self.percent, 'percent')
            self.validate(self.question_templates, 'question_templates')
            logger.debug("Validation done, looks ok.")

    def get_first_invalid_key(self, sentence: S) -> Optional[str]:
        words = sentence[:]
        while words:
            word = words.pop()
            process_function = self.decide_process_function(word)
            new_words = []
            if process_function == self.process_option:
                new_words = word[1:-1].split(" ")
            elif process_function == self.process_alternative:
                alternatives = word[1:-1].split("|")
                logger.debug(alternatives)
                new_words = [word for alternative in alternatives for word in alternative.split(" ")]
                logger.debug(new_words)
            elif process_function == self.process_condition:
                try:
                    self._access_percent(word[1:])
                except Exception as e:
                    logger.error(str(e))
                    return word
            elif process_function == self.process_function:
                try:
                    self._access_bang(word[1:])
                except Exception as e:
                    logger.error(str(e))
                    return word
            elif process_function == self.process_template:
                try:
                    x = self._access_dollar(word[1:])
                    assert isinstance(x, list) or "any" in x
                except Exception as e:
                    logger.error(str(e))
                    return word
            words.extend(new_words)
        return None

    def _estimate_words(self, sentence: S):
        # todo: w/o replacement
        combinations = 1
        logger.debug(f"Estimating size of '{sentence}'.")
        for w in sentence:
            logger.debug(f"w is {w}")
            process_function = self.decide_process_function(w)
            if process_function == self.process_template:
                logger.debug(self.visited_keys_estimate)
                combinations *= self._estimate_sentences(self._access_dollar(w[1:]))

            elif process_function == self.process_option:
                combinations *= 1 + self._estimate_words(w[1:].split(" "))

            elif process_function == self.process_alternative:
                combinations *= self._estimate_sentences([sent.split(" ") for sent in w[1:-1].split("|")])

            elif process_function == self.process_condition:
                combinations *= sum(
                    self._estimate_sentences(v) for k, v in self._access_percent(w[1:]).items() if
                    k != "condition"
                )
            elif process_function == self.process_function:
                f = self._access_bang(w[1:])
                if f.options:
                    logger.debug(f"Calculating with options: {f.options}")
                    combinations *= sum(self._estimate_words(o) for o in S(f.options))
                else:
                    logger.debug(f"Calculating with number: {f.number}")
                    assert f.number > 0
                    combinations *= f.number

            elif not process_function:
                ...
            else:
                raise NotImplementedError()
        logger.debug(f"Size of '{sentence}' is {combinations}.")
        return combinations

    def _estimate_sentences(self, sentences: List[S]):
        combined = 0
        for sentence in sentences:
            combined += self._estimate_words(sentence)

        return combined

    def estimate_size(self, sentences: List[S]) -> int:
        combined = 0
        for sentence in sentences:
            self.visited_keys_estimate = defaultdict(count)
            combined += self._estimate_words(sentence)

        return combined

    def validate(self, template: dict, path=''):
        for k, v in template.items():
            if isinstance(v, Callable):
                pass  # you're off the hook
            elif isinstance(v, list):
                for i, t in enumerate(v):
                    invalid = self.get_first_invalid_key(t)
                    if invalid:
                        raise ValueError(f"path '{path}.{k}', sentence '{i}' contains access "
                                         f"key >>>{invalid}<<< which is invalid!")
            elif isinstance(v, dict):
                self.validate(v, path=f"{path}.{k}")

    def _access_context(self, word: str, record_visit=True) -> List[str]:
        n = self.context
        if word.startswith('sent') and record_visit:
            self.context['visits'][self.context['sent_nr']].append(word)
        for k in word.split("."):
            try:
                n = n[k]
            except KeyError:
                n = getattr(n, k)
                if not n:
                    raise NotImplementedError()
        return str(n).split()

    def _access_percent(self, word) -> Dict[str, Union[Callable, S]]:
        n = self.percent
        for k in word.split("."):
            try:
                n = n[k]
            except KeyError:
                n = getattr(n, k)
                if not n:
                    raise NotImplementedError()
        return n

    def _access_bang(self, word) -> F:
        n = self.bang
        for k in word.split("."):
            try:
                n = n[k]
            except KeyError:
                n = getattr(n, k)
                if not n:
                    raise NotImplementedError()
        return n

    def _access_dollar(self, word) -> S:
        n = self.dollar
        for k in word.split("."):
            try:
                n = n[k]
            except KeyError:
                n = getattr(n, k)
                if not n:
                    raise NotImplementedError()
        return n

    def with_feedback(self, e: Exception):
        if isinstance(e, AttributeError) and "object has no attribute 'random'" in str(e):
            msg = f"{self.context['word']} is not a leaf path and template doesn't provide .any"
        elif isinstance(e, TypeError) and "list indices must be integers or slices, not str" in str(e):
            msg = ""
        elif isinstance(e, KeyError) and "dict object has no attribute" in str(e):
            msg = f"{self.context['word']} is not a valid template path!"
        elif isinstance(e, YouIdiotException):
            msg = str(e)
        else:
            msg = "And i don't even know what's wrong!"
        logger.debug(f"{self.context['chosen_templates'][-1]}")
        logger.debug(f"{self.context['choices']}")
        logger.error(msg)
        return YouIdiotException(msg)

    def decide_process_function(self, word) -> Optional[Callable[[str], List[str]]]:
        if word.startswith("(") and word.endswith(")"):
            r = self.process_option
        elif word.startswith("[") and word.endswith("]"):
            r = self.process_alternative
        elif word.startswith("%"):
            r = self.process_condition
        elif word.startswith("#"):
            r = self.process_context
        elif word.startswith("$"):
            r = self.process_template
        elif word.startswith("!"):
            r = self.process_function
        else:
            return None
        logger.debug(f"Deciding to process with {r.__name__}")
        return r

    def process_option(self, word):
        # word = self.context['word']
        # stack = self.context['stack']
        logger.debug("...Word is an option ()...")
        # 50/50 whether to ignore it
        if random.choice([True, False]):
            new_words = word[1:-1].split()
            logger.debug(f"... new words: {new_words}")
            return new_words
            # stack.extend(new_words)
        else:
            random.choice("Discarding!")
            return []

    def process_alternative(self, word):
        # word = self.context['word']
        # stack = self.context['stack']
        logger.debug("...Word is an alternative []...")
        alternatives = word[1:-1].split("|")
        choice = random.choice(alternatives)
        logger.debug(f"...Choosing {choice}")
        # stack.extend(choice.split()[::-1])
        return choice.split()

    def process_condition(self, word):
        # word = self.context['word']
        # stack = self.context['stack']
        logger.debug("...Word is a condition %...")
        branch = self._access_percent(word[1:])
        logger.debug(
            f"...Evaluating: {branch['condition'].__name__}\n{textwrap.dedent(inspect.getsource(branch['condition']))}")
        result = branch['condition'](self.context)
        logger.debug(f"with result: {result}")
        new_words, idx = branch[result].random()
        self.context['choices'].append(f"{word}.{idx}")

        logger.debug(f"... new words: {new_words}")
        new_words = [str(w) for w in new_words]
        # stack.extend(new_words)
        return new_words

    def process_context(self, word):
        logger.debug("...Word is context #...")
        new_words = self._access_context(word[1:])

        logger.debug(f"... new words: {new_words}")
        # self.context['stack'].extend(new_words[::-1])
        return new_words

    def process_template(self, word):
        logger.debug("...Word is template $...")
        logger.debug(f"Choices so far: {self.context['choices']}")
        exclude = [int(x.rsplit(".", 1)[-1]) for x in self.context['choices'] if x.startswith(word + '.')]
        logger.debug(f"Excluding choices: {exclude}")
        try:
            new_words, idx = self._access_dollar(word[1:]).random(exclude=exclude)
        except (KeyError, AttributeError) as _:
            logger.debug("Trying any...")
            new_words, idx = self._access_dollar(word[1:] + ".any").random(exclude=exclude)
        except IndexError as _:
            raise YouIdiotException(f"Template {self.context['chosen_templates'][-1]} uses '{word}' "
                                    f"more often than there are unique alternatives!")
        new_words = new_words
        self.context['choices'].append(f"{word}.{idx}")
        logger.debug(f"... new words: {new_words}")
        # self.context['stack'].extend(new_words)
        return new_words

    def process_function(self, word):
        logger.debug("...Word is a function !...")
        # get and execute
        func = self._access_bang(word[1:])
        new_words = str(func(self.context)).split()
        logger.debug(f"... new words: {new_words}")
        # self.context['stack'].extend(new_words[::-1])
        return new_words

    def realise_story(self, sentences: List[Sentence], world) -> Tuple[List[str], Dict]:
        self.context = dict()
        self.context['world'] = world
        self.context['sentences'] = sentences
        self.context['chosen_templates'] = list()
        self.context['visits'] = defaultdict(list)
        self.context['realizer'] = self
        realised = []
        for self.context['sent_nr'], self.context['sent'] in enumerate(sentences):
            logger.debug(self.context['sent'])
            # try:
            # except Exception as e:
            #    raise self.with_feedback(e)
            sent = self.realise_sentence()
            realised.append(sent)
        return realised, self.context['visits']

    def realise_sentence(self):
        ctx = self.context
        logger.debug(f"===PROCESSING NEW SENTENCE: #{ctx['sent_nr']}, action = {ctx['sent'].action}===")

        # select template and the chosen number (for tracking purposes)
        logger.debug(f"Use unique sentences?: {self.unique_sentences}")
        if self.unique_sentences:
            exclude = [int(x.rsplit(".", 1)[-1]) for x in self.context['chosen_templates'] if
                       x.startswith(ctx['sent'].action + '.')]
            logger.debug(f'Choices to exclude... {exclude}')
            if len(exclude) == len(self.sentences[ctx['sent'].action]):
                raise YouIdiotException(f"Your story has more '{ctx['sent'].action}' actions (> {len(exclude)}) "
                                        f"than you have choices for ({len(self.sentences[ctx['sent'].action])})!")
        else:
            exclude = []
        template, template_nr = self.sentences[ctx['sent'].action].random(exclude=exclude)

        # set chosen template
        self.context['chosen_templates'].append(f"{ctx['sent'].action}.{template_nr}")

        # initialise context
        self.context['realised'] = []
        self.context['choices'] = []
        self.context['stack'] = template
        self.context['stack'].reverse()

        # while there's something on the stack
        while self.context['stack']:

            # take next word from stack
            self.context['word'] = self.context['stack'].pop()
            word = self.context['word']

            logger.debug(f"State: {' '.join(ctx['realised'])} ++ {word} ++ {' '.join(self.context['stack'][::-1])}")

            # decide what to process with
            process_function = self.decide_process_function(word)

            # apply if not a plain word
            if process_function:
                new_words = process_function(word)
                ctx['stack'].extend(new_words[::-1])
            # if plain word, just append
            else:
                ctx['realised'].append(word)
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
                new_words = self.process_option(word)
                stack.extend(new_words[::-1])
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
