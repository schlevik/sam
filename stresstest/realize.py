import inspect
import random
import textwrap
from collections import defaultdict
from typing import List, Callable, Optional, Dict, Tuple, Union

from loguru import logger

from stresstest.classes import S, YouIdiotException, F, Event, Context, Question
from stresstest.resources.templates import percent, sentences, at, dollar, bang, \
    question_templates as question_templates


class Accessor:
    def __init__(self, context: Context = None, sentences=sentences, percent=percent, at=at, dollar=dollar, bang=bang,
                 question_templates=question_templates):
        self.context = context or Context()
        self.bang = prepare_templates(bang)
        self.dollar = prepare_templates(dollar)
        self.sentences = prepare_templates(sentences)
        self.at = prepare_templates(at)
        self.percent = prepare_templates(percent, True)
        self.question_templates = prepare_templates(question_templates)

    def _access(self, word: str, target):
        n = target
        for k in word.split("."):
            try:
                n = n[k]
            except KeyError:
                n = getattr(n, k)
                if not n:
                    raise NotImplementedError()
        return n

    def access_context(self, word: str, record_visit=True) -> List[str]:
        if word.startswith('sent') and record_visit:
            self.context.visits[self.context.sent_nr].append(word)
        return str(self._access(word, self.context)).split()

    def access_percent(self, word) -> Dict[str, Union[Callable, S]]:
        return self._access(word, self.percent)

    def access_bang(self, word) -> F:
        return self._access(word, self.bang)

    def access_dollar(self, word) -> S:
        return self._access(word, self.dollar)

    def access_at(self, word) -> S:
        return self._access(word, self.at)


class Processor:
    def __init__(self, accessor: Accessor):
        self.accessor = accessor
        self.context = self.accessor.context

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
        elif word.startswith("@"):
            r = self.process_feature
        else:
            return None
        logger.debug(f"Deciding to process with {r.__name__}")
        return r

    def process_feature(self, word):
        logger.debug("...Word is a feature @...")
        if word[1:].startswith("MODIFIER"):
            if "modifier" in self.context.sent.features:
                new_words, idx = self.accessor.access_at(word[1:]).random()
            else:
                new_words = []
            return new_words
        else:
            raise NotImplementedError(f"Don't know how to process {word.split('.', 1)[0]} type of feature!")

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
        branch = self.accessor.access_percent(word[1:])
        logger.debug(
            f"...Evaluating: {branch['condition'].__name__}\n{textwrap.dedent(inspect.getsource(branch['condition']))}")
        result = branch['condition'](self.context)
        logger.debug(f"with result: {result}")
        new_words, idx = branch[result].random()
        self.context.choices.append(f"{word}.{idx}")

        logger.debug(f"... new words: {new_words}")
        new_words = [str(w) for w in new_words]
        # stack.extend(new_words)
        return new_words

    def process_context(self, word):
        logger.debug("...Word is context #...")
        new_words = self.accessor.access_context(word[1:])

        logger.debug(f"... new words: {new_words}")
        # self.context['stack'].extend(new_words[::-1])
        return new_words

    def process_template(self, word):
        logger.debug("...Word is template $...")
        logger.debug(f"Choices so far: {self.context.choices}")
        exclude = [int(x.rsplit(".", 1)[-1]) for x in self.context.choices if x.startswith(word + '.')]
        logger.debug(f"Excluding choices: {exclude}")
        try:
            new_words, idx = self.accessor.access_dollar(word[1:]).random(exclude=exclude)
        except (KeyError, AttributeError) as _:
            logger.debug("Trying any...")
            new_words, idx = self.accessor.access_dollar(word[1:] + ".any").random(exclude=exclude)
        except IndexError as _:
            raise YouIdiotException(f"Template {self.context.chosen_templates[-1]} uses '{word}' "
                                    f"more often than there are unique alternatives!")
        new_words = new_words
        self.context.choices.append(f"{word}.{idx}")
        logger.debug(f"... new words: {new_words}")
        # self.context['stack'].extend(new_words)
        return new_words

    def process_function(self, word):
        logger.debug("...Word is a function !...")
        # get and execute
        func = self.accessor.access_bang(word[1:])
        new_words = str(func(self.context)).split()
        logger.debug(f"... new words: {new_words}")
        # self.context['stack'].extend(new_words[::-1])
        return new_words


class Validator:
    def __init__(self, processor: Processor):
        self.processor = processor
        self.accessor = self.processor.accessor

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

    def get_first_invalid_key(self, sentence: S) -> Optional[str]:
        words = sentence[:]
        while words:
            word = words.pop()
            process_function = self.processor.decide_process_function(word)
            new_words = []
            if process_function == self.processor.process_option:
                new_words = word[1:-1].split(" ")
            elif process_function == self.processor.process_alternative:
                alternatives = word[1:-1].split("|")
                logger.debug(alternatives)
                new_words = [word for alternative in alternatives for word in alternative.split(" ")]
                logger.debug(new_words)
            elif process_function == self.processor.process_condition:
                try:
                    self.accessor.access_percent(word[1:])
                except Exception as e:
                    logger.error(str(e))
                    return word
            elif process_function == self.processor.process_function:
                try:
                    self.accessor.access_bang(word[1:])
                except Exception as e:
                    logger.error(str(e))
                    return word
            elif process_function == self.processor.process_template:
                try:
                    x = self.accessor.access_dollar(word[1:])
                    assert isinstance(x, list) or "any" in x
                except Exception as e:
                    logger.error(str(e))
                    return word
            words.extend(new_words)
        return None


class SizeEstimator:
    def __init__(self, processor: Processor):
        self.processor = processor
        self.accessor = self.processor.accessor

    def _estimate_words(self, sentence: S, pessimistic=False):
        # todo: w/o replacement
        combinations = 1
        logger.debug(f"Estimating size of '{sentence}'.")
        aggr = min if pessimistic else sum
        for w in sentence:
            logger.debug(f"w is {w}")
            process_function = self.processor.decide_process_function(w)
            if process_function == self.processor.process_template:
                combinations *= self._estimate_sentences(self.accessor.access_dollar(w[1:]), pessimistic)

            elif process_function == self.processor.process_option:
                combinations *= 1 + self._estimate_words(w[1:-1].split(" "), pessimistic)

            elif process_function == self.processor.process_alternative:
                combinations *= self._estimate_sentences([sent.split(" ") for sent in w[1:-1].split("|")], pessimistic)

            elif process_function == self.processor.process_condition:

                combinations *= aggr(
                    self._estimate_sentences(v, pessimistic) for k, v in self.accessor.access_percent(w[1:]).items() if
                    k != "condition"
                )
            elif process_function == self.processor.process_function:
                f = self.accessor.access_bang(w[1:])
                if f.options:
                    logger.debug(f"Calculating with options: {f.options}")
                    result = aggr(self._estimate_words(o, pessimistic) for o in S(f.options))
                    logger.debug(f"Size of '{w}' is {result}.")
                    combinations *= result
                else:
                    if not pessimistic:
                        assert f.number > 0
                        logger.debug(f"Calculating with number: {f.number}")
                        # if pessimistic, assume f.number = 1
                        combinations *= f.number

            elif not process_function or process_function == self.processor.process_context:
                ...
            else:
                logger.debug(f"Unknown process function: {process_function}")
                raise NotImplementedError()

        logger.debug(f"Size of '{sentence}' is {combinations}.")
        return combinations

    def _estimate_sentences(self, sentences: List[S], pessimistic=False):
        combined = 0
        for sentence in sentences:
            combined += self._estimate_words(sentence, pessimistic)

        return combined

    def estimate_size(self, sentences: List[S], pessimistic=False) -> int:
        combined = 0
        for sentence in sentences:
            combined += self._estimate_words(sentence, pessimistic)

        return combined


def prepare_templates(templates, allow_conditions=False) -> dict:
    processed = {}
    for k, v in templates.items():
        if k == 'condition':
            if not allow_conditions:
                raise ValueError("Conditions not allowed!")
            assert isinstance(v, Callable)
        elif isinstance(v, dict):
            v = prepare_templates(v, allow_conditions)
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
    context: Context

    def __init__(self, sentences=sentences, bang=bang, dollar=dollar, at=at, percent=percent,
                 question_templates=question_templates, validate=True, unique_sentences=True):
        self.unique_sentences = unique_sentences
        logger.debug("Creating new Realizer...")
        self.context = None
        self.bang = prepare_templates(bang)
        self.dollar = prepare_templates(dollar)
        self.sentences = prepare_templates(sentences)
        self.at = prepare_templates(at)
        self.percent = prepare_templates(percent, True)
        self.question_templates = prepare_templates(question_templates)
        self.accessor = Accessor(self.context, sentences=sentences, bang=bang, dollar=dollar, at=at, percent=percent,
                                 question_templates=question_templates)
        self.processor = Processor(self.accessor)
        if validate:
            validator = Validator(self.processor)
            logger.debug("Validating templates...")
            validator.validate(self.dollar, 'dollar')
            validator.validate(self.sentences, 'sentences')
            validator.validate(self.percent, 'percent')
            validator.validate(self.question_templates, 'question_templates')
            logger.debug("Validation done, looks ok.")

    def with_feedback(self, e: Exception):
        if isinstance(e, AttributeError) and "object has no attribute 'random'" in str(e):
            msg = f"{self.context.word} is not a leaf path and template doesn't provide .any"
        elif isinstance(e, TypeError) and "list indices must be integers or slices, not str" in str(e):
            msg = ""
        elif isinstance(e, KeyError) and "dict object has no attribute" in str(e):
            msg = f"{self.context.word} is not a valid template path!"
        elif isinstance(e, YouIdiotException):
            msg = str(e)
        else:
            msg = "And i don't even know what's wrong!"
        logger.debug(f"{self.context.chosen_templates[-1]}")
        logger.debug(f"{self.context.choices}")
        logger.error(msg)
        return YouIdiotException(msg)

    def realise_story(self, sentences: List[Event], world) -> Tuple[List[str], Dict]:
        self.context = Context()
        self.processor.context = self.context
        self.accessor.context = self.context
        self.context.world = world
        self.context.sentences = sentences
        self.context.chosen_templates = list()
        self.context.visits = defaultdict(list)
        self.context.realizer = self
        realised = []
        for self.context.sent_nr, self.context.sent in enumerate(sentences):
            logger.debug(self.context.sent)
            # try:
            # except Exception as e:
            #    raise self.with_feedback(e)
            sent = self.realise_sentence()
            realised.append(sent)
        return realised, self.context.visits

    def realise_sentence(self):
        ctx: Context = self.context
        logger.debug(f"===PROCESSING NEW SENTENCE: #{ctx.sent_nr}, event = {ctx.sent.event_type}===")

        # select template and the chosen number (for tracking purposes)
        logger.debug(f"Use unique sentences?: {self.unique_sentences}")
        if self.unique_sentences:
            exclude = [int(x.rsplit(".", 1)[-1]) for x in self.context.chosen_templates if
                       x.startswith(ctx.sent.event_type + '.')]
            logger.debug(f'Choices to exclude... {exclude}')
            if len(exclude) == len(self.sentences[ctx.sent.event_type]):
                raise YouIdiotException(f"Your story has more '{ctx.sent.event_type}' events (> {len(exclude)}) "
                                        f"than you have choices for ({len(self.sentences[ctx.sent.event_type])})!")
        else:
            exclude = []
        template, template_nr = self.sentences[ctx.sent.event_type].random(exclude=exclude)

        # set chosen template
        self.context.chosen_templates.append(f"{ctx.sent.event_type}.{template_nr}")

        # initialise context
        self.context.realized = []
        self.context.choices = []
        self.context.stack = template
        self.context.stack.reverse()

        # while there's something on the stack
        while self.context.stack:

            # take next word from stack
            self.context.word = self.context.stack.pop()
            word = self.context.word

            logger.debug(f"State: {' '.join(ctx.realized)} ++ {word} ++ {' '.join(self.context.stack[::-1])}")

            # decide what to process with
            process_function = self.processor.decide_process_function(word)

            # apply if not a plain word
            if process_function:
                new_words = process_function(word)
                ctx.stack.extend(new_words[::-1])
            # if plain word, just append
            else:
                ctx.realized.append(word)
            logger.debug("...done!")

        logger.debug(f"Final sentence is: {ctx.realized}")

        return " ".join(ctx.realized)

    def _fix_units(self, question: Question, passage: List[str]):
        if question.target == 'distance' and question.answer:
            unit = 'metre'
        elif question.target == 'time' and question.answer:
            unit = 'minute'
        else:
            return question.answer
        logger.debug(f"Target: {question.target}")
        logger.debug(f"Target: {passage}")
        new_answers = []
        for answer in question.answer if isinstance(question.answer, list) else [question.answer]:
            logger.debug(f"Answer: {answer}, Evidence: {question.evidence}")
            candidate_sentences = [passage[i] for i in question.evidence]
            candidate_spans = []
            for candidate_sentence in candidate_sentences:
                tokens = candidate_sentence.split()
                logger.debug(f"Evidence tokens: {tokens}")
                answer_position = next((i for i, token in enumerate(tokens) if str(token) == str(answer)), None)
                logger.debug(f"Answer found in evidence: {answer_position}")
                if answer_position:
                    unit_found = False
                    i = 0
                    while not unit_found:
                        try:
                            if tokens[answer_position + i].startswith(unit):
                                candidate_spans.append(tokens[answer_position:answer_position + i + 1])
                                unit_found = True
                            if tokens[answer_position - i].startswith(unit):
                                candidate_spans.append(tokens[answer_position - i:answer_position + 1])
                                unit_found = True
                            i += 1
                        except IndexError:
                            break
            shortest_answer = sorted(candidate_spans, key=len)[0]
            new_answers.append(shortest_answer)
        new_answers = [' '.join(answer) for answer in new_answers]
        return new_answers if isinstance(question.answer, list) else new_answers[0]

    def realise_question(self, q: Question, passage: List[str]):
        logger.debug(f"Question: {q}")
        try:
            template, template_nr = self.question_templates[q.type][q.target][q.event_type].random()
        except KeyError:
            return None
        question_words = []
        template.reverse()
        stack = template
        while stack:
            logger.debug(f"Current stack is: {stack}")
            word = stack.pop()
            logger.debug(word)

            # alternative as in ()
            if word.startswith("(") and word.endswith(")"):
                new_words = self.processor.process_option(word)
                stack.extend(new_words[::-1])
            # context access
            elif word.startswith("#"):
                try:
                    new_word = q.question_data[word[1:]]
                except KeyError:
                    raise NotImplementedError(f"{word} is not in question data!")
                stack.append(str(new_word))

            else:
                question_words.append(word)
        logger.debug(question_words)
        q.realized = " ".join(question_words) + "?"
        answer = self._fix_units(q, passage)
        q.answer = answer
        return " ".join(question_words) + "?", q.answer
