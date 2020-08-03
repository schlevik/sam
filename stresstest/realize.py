import inspect
import random
import textwrap
from collections import defaultdict
from itertools import zip_longest
from typing import List, Callable, Optional, Dict, Tuple, Union, Any

import nltk
import pattern.text.en as pattern_en
from loguru import logger
from pattern.text.en import tenses

from stresstest.classes import S, YouIdiotException, F, Event, Context, Question, World


class RandomChooser:
    def choose(self, choices, exclude=None, *args, **kwargs):
        if isinstance(choices, S):
            return choices.random(exclude=exclude)
        elif isinstance(choices, Callable):
            return str(choices()).split()
        elif isinstance(choices, List):
            choice = random.choice(range(len(choices)))
            return choices[choice], choice


class DeterminedSentenceTemplateChooser(RandomChooser):
    def choose(self, choices, exclude=None, *args, is_sentence=True, **kwargs):
        ...


class DeterminedChooser(RandomChooser):
    def __init__(self, choices: Optional[List[List[Tuple[str, Any]]]], sentence_choices: List[Tuple[str, int]]):
        self.choices = choices or [[]] * len(sentence_choices)
        self.sentence_choices = sentence_choices
        self.iter = self._iter()
        self.skip = False

    def _iter(self):
        for choices_for_ith_sentence, ith_sentence_template in zip(self.choices, self.sentence_choices):
            yield ith_sentence_template
            for choice in choices_for_ith_sentence:
                yield choice

    def choose(self, choices, exclude=None, *args, **kwargs):
        if self.skip:
            # raise NotImplementedError()
            return super().choose(choices, exclude=exclude, *args, **kwargs)
        name, choice = next(self.iter)
        if isinstance(choices, S):
            # name, idx = next(self.iter)
            logger.debug(f"Choosing {name}.{choice}")
            if choice == '<any>':
                return super().choose(choices, exclude, *args, **kwargs)
            else:
                return choices[choice], choice
        elif isinstance(choices, Callable):
            # name, content = next(self.iter)
            logger.debug(f"For {name} choosing {choice}")
            return choice
        elif isinstance(choices, List):
            logger.debug(f"Choosing {choices}:{choice} ({name})")
            if choice == '<any>':
                super().choose(choices, exclude, *args, **kwargs)
            else:
                return choices[choice], choice


class Accessor:
    def __init__(self, sentences, percent, at, dollar, bang,
                 question_templates, context: Context = None):
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
            except (KeyError, TypeError) as _:
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
    def __init__(self, accessor: Accessor, chooser: RandomChooser):
        self.accessor = accessor
        self.context = self.accessor.context
        self.chooser = chooser
        self._is_recording = True

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
        elif word == "{":
            r = self.stop_recording
        elif word == "}":
            r = self.start_recording
        else:
            return None
        logger.debug(f"Deciding to process with {r.__name__}")
        return r

    def process_feature(self, word):
        logger.debug("...Word is a feature @...")
        logger.debug(self.accessor.at.keys())
        modifier_type = word[1:].split(".", 1)[0]
        logger.debug(f"{modifier_type} in {self.accessor.at.keys()}?: {modifier_type in self.accessor.at}")
        if modifier_type in self.accessor.at:
            if any(s.startswith(f"{modifier_type}") for s in self.context.sent.features):
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
        include, idx = self.chooser.choose([True, False])
        self.record(word, idx)
        if include:
            new_words = word[1:-1].split()
            logger.debug(f"... new words: {new_words}")
            return new_words
            # stack.extend(new_words)
        else:
            random.choice("Discarding!")
            return []

    def process_alternative(self, word):
        logger.debug("...Word is an alternative []...")
        alternatives = word[1:-1].split("|")
        # choice = random.choice(alternatives)
        choice, idx = self.chooser.choose(alternatives)
        logger.debug(f"...Choosing {choice}")
        self.record(word, idx)
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
        # new_words, idx = branch[result].random()
        new_words, idx = self.chooser.choose(branch[result])
        self.record(word, idx)

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
        logger.debug(f"Choices so far: {self.context.choices[self.context.sent_nr]}")

        # to not use the same template twice
        exclude = [
            choice for name, choice in self.context.current_choices if name.startswith(word + '.') or name == word
        ]
        logger.debug(f"Excluding choices: {exclude}")
        try:
            choices = self.accessor.access_dollar(word[1:])
            # new_words, idx = choices.random(exclude=exclude)
            # new_words, idx = self.chooser.choose(choices, exclude)
        except (KeyError, AttributeError) as _:
            logger.debug("Trying any...")
            choices = self.accessor.access_dollar(word[1:] + ".any")

            # new_words, idx = choices.random(exclude=exclude)
        except IndexError as _:
            raise YouIdiotException(f"Template {self.context.chosen_templates[-1]} uses '{word}' "
                                    f"more often than there are unique alternatives!")
        new_words, idx = self.chooser.choose(choices, exclude)
        self.record(word, idx)
        logger.debug(f"... new words: {new_words}")
        # self.context['stack'].extend(new_words)
        return new_words

    def process_function(self, word, args=None):
        args = args or self.context
        logger.debug("...Word is a function !...")
        # get and execute
        func = self.accessor.access_bang(word[1:])
        # new_words = str(func(self.context)).split()
        new_words = self.chooser.choose(lambda: func(args))
        logger.debug(f"... new words: {new_words}")
        self.record(word, new_words)
        # self.context['stack'].extend(new_words[::-1])
        return new_words

    def start_recording(self, *args, **kwargs):
        logger.debug("...Word is a start recording }...")
        self._is_recording = True
        self.chooser.skip = False
        return []

    def stop_recording(self, *args, **kwargs):
        logger.debug("...Word is a stop recording {...")
        self._is_recording = False
        self.chooser.skip = True
        return []

    def record(self, word, idx):
        if self._is_recording:
            logger.debug(f"Recording: ({word},{idx})")
            self.context.current_choices.append((word, idx))
            assert "almost" not in word
        else:
            logger.debug(f"NOT RECORDING: ({word},{idx})")
        logger.debug(f"IS RECORDING?: {self._is_recording}")
        logger.debug(f"RECORDED SO FAR: {self.context.current_choices}")


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
            elif process_function == self.processor.process_feature:
                try:
                    x = self.accessor.access_at(word[1:])
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
            elif process_function == self.processor.process_feature:
                # pessimistic = True because usually the selection is not random
                combinations *= self._estimate_sentences(self.accessor.access_at(w[1:]), pessimistic=True)

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
    context: Optional[Context]

    def __init__(self, sentences, bang, dollar, at, percent,
                 question_templates, validate=True, unique_sentences=True):
        self.unique_sentences = unique_sentences
        logger.debug("Creating new Realizer...")
        self.context = None
        self.bang = prepare_templates(bang)
        self.dollar = prepare_templates(dollar)
        self.sentences = prepare_templates(sentences)
        self.at = prepare_templates(at)
        self.percent = prepare_templates(percent, True)
        self.question_templates = prepare_templates(question_templates)
        self.accessor = Accessor(context=self.context, sentences=sentences, bang=bang, dollar=dollar, at=at,
                                 percent=percent,
                                 question_templates=question_templates)
        self.chooser = RandomChooser()
        self.processor = Processor(self.accessor, self.chooser)
        if validate:
            validator = Validator(self.processor)
            logger.debug("Validating templates...")
            validator.validate(self.dollar, 'dollar')
            validator.validate(self.sentences, 'sentences')
            validator.validate(self.percent, 'percent')
            validator.validate(self.question_templates, 'question_templates')
            logger.debug("Validation done, looks ok.")

    def realise_story(self, events: List[Event], world) -> Tuple[List[str], Dict]:
        self.context = Context()
        self.processor.context = self.context
        self.accessor.context = self.context
        self.context.world = world
        self.context.sentences = events
        self.context.chosen_templates = list()
        self.context.choices = []
        self.context.visits = defaultdict(list)
        self.context.realizer = self

        realised = []
        for self.context.sent_nr, self.context.sent in enumerate(events):
            logger.debug(self.context.sent)
            # try:
            # except Exception as e:
            #    raise self.with_feedback(e)
            sent = self.realise_sentence()
            realised.append(sent)
        return realised, self.context.visits

    def realise_with_sentence_choices(self, events: List[Event], world: World, chosen_templates):
        self.chooser = DeterminedChooser(None, chosen_templates)
        self.processor.chooser = RandomChooser()  # just to be sure
        return self.realise_story(events, world)

    def realise_with_choices(self, events: List[Event], world: World, choices: List[List[Tuple[Any, int]]],
                             chosen_templates):
        self.chooser = DeterminedChooser(choices, chosen_templates)
        self.processor.chooser = self.chooser
        return self.realise_story(events, world)

    def realise_sentence(self):
        ctx: Context = self.context
        logger.debug(f"===PROCESSING NEW SENTENCE: #{ctx.sent_nr}, event = {ctx.sent.event_type}===")

        # select template and the chosen number (for tracking purposes)
        logger.debug(f"Use unique sentences?: {self.unique_sentences}")
        if self.unique_sentences:
            exclude = [idx for event_type, idx in self.context.chosen_templates if
                       event_type.startswith(ctx.sent.event_type + '.') or event_type == ctx.sent.event_type]
            logger.debug(f'Choices to exclude... {exclude}')
            if len(exclude) >= len(self.sentences[ctx.sent.event_type]):
                raise YouIdiotException(f"Your story has more '{ctx.sent.event_type}' events (> {len(exclude)}) "
                                        f"than you have choices for ({len(self.sentences[ctx.sent.event_type])})!")
        else:
            exclude = []
        # template, template_nr = self.sentences[ctx.sent.event_type].random(exclude=exclude)
        template, template_nr = self.chooser.choose(self.sentences[ctx.sent.event_type], exclude=exclude)
        # set chosen template
        self.context.chosen_templates.append((ctx.sent.event_type, template_nr))
        # initialise context
        self.context.realized = []
        self.context.choices.append(list())
        self.context.stack = template[::-1]
        # self.context.stack.reverse()
        logger.debug(f"Template: {template}")
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
        # normalise space
        return " ".join(" ".join(self.post_process(ctx.realized)).split())

    vocals = "aeiou"

    def post_process(self, tokens: List[str]):
        # tokens = sentence.split(" ")
        lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        result = []
        for i, (prev_token, token, next_token) in \
                enumerate(zip_longest([""] + tokens[:-1], tokens, tokens[1:], fillvalue="")):
            prev_prev_token = tokens[i - 2] if i >= 2 else ""
            # capitalise
            if i == 0:
                token = token[0].upper() + token[1:]
            if token == '1' and next_token == 'th':
                try:
                    if tokens[i + 2] == 'to' and tokens[i + 3] == 'last':
                        token = ''
                except IndexError:
                    pass
            if next_token == 'last' and token == 'to' and prev_token == 'th' and prev_prev_token == '1':
                token = ''
            if (token == 'a' or token == 'A') and len(next_token) > 0 and next_token[0] in self.vocals:
                token = token + "n"
            elif token == 'into' and next_token == 'between':
                token = 'in'
            elif token == 'th' and prev_token.endswith('1'):
                token = 'st'
                try:
                    if next_token == 'to' and tokens[i + 2] == 'last':
                        token = ''
                except IndexError:
                    pass
            elif token == 'th' and prev_token.endswith('2'):
                token = 'nd'
            elif token == 'th' and prev_token.endswith('3'):
                token = 'rd'
            elif prev_token == 'to' \
                    and (token.endswith("ed") or token.endswith("ing") or
                         tenses(token) and tenses(token)[0][0] == 'past'):
                token = lemmatizer.lemmatize(token, 'v')
            elif prev_token == 'in' or prev_token == 'from' and (any(t[0] == 'past' for t in tenses(token))):
                # VERY HACKY

                if (prev_prev_token in ['refrained', "refused", "prohibited", "prevented", "hindered"]
                        or (prev_token == "in" and prev_prev_token == 'succeed')):
                    token = lemmatizer.lemmatize(token, 'v')
                    try:
                        token = pattern_en.verbs[token][5]
                    except:
                        if not token.endswith("ing"):
                            token = lemmatizer.lemmatize(token).rsplit("e", 1)[0] + "ing"

            elif prev_token in ('not', "n't") and prev_prev_token in ("could", "would", "did"):
                token = lemmatizer.lemmatize(token, 'v')
            else:
                pass
            result.append(token)
        return result

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
            for idx, candidate_sentence in enumerate(candidate_sentences):
                tokens = candidate_sentence.split()
                logger.debug(f"Evidence tokens: {tokens}")
                answer_positions = [i for i, token in enumerate(tokens) if str(token) == str(answer)]
                for answer_position in answer_positions:
                    logger.debug(f"Answer found in evidence {idx}: {answer_position}")
                    unit_found = False
                    done = False
                    i = 0
                    while not unit_found and not done:
                        if answer_position + i < len(tokens) and tokens[answer_position + i].startswith(unit):
                            candidate_spans.append(tokens[answer_position:answer_position + i + 1])
                            unit_found = True
                            logger.debug(f"Unit found in evidence {idx}: {answer_position + i} ")
                            logger.debug(f"adding: tokens[{answer_position}:{answer_position + i + 1}: "
                                         f"{tokens[answer_position:answer_position + i + 1]}")

                        if answer_position - i > 0 and tokens[answer_position - i].startswith(unit):
                            candidate_spans.append(tokens[answer_position - i:answer_position + 1])
                            unit_found = True
                            logger.debug(f"Unit found in evidence {idx}: {answer_position + i} ")
                            logger.debug(f"adding: tokens[{answer_position - i}:{answer_position + 1}]: "
                                         f"{tokens[answer_position - i:answer_position + 1]}")
                        if answer_position + i > len(tokens) and answer_position - i < 0:
                            done = True
                        i += 1
            shortest_answer = sorted(candidate_spans, key=len)[0]
            logger.debug(f"Shortest answer found: {shortest_answer} in {sorted(candidate_spans, key=len)}")
            new_answers.append(shortest_answer)
        assert new_answers
        new_answers = [' '.join(answer) for answer in new_answers]
        return new_answers if isinstance(question.answer, list) else new_answers[0]

    def realise_question(self, q: Question, passage: List[str], ignore_missing_keys=True):
        self.processor.chooser = RandomChooser()
        logger.debug(f"Question: {q}")
        try:
            # first see if there's a reasoning key
            template, template_nr = self.question_templates[q.type][q.target][q.reasoning][q.event_type].random()
        except KeyError as e:
            try:
                # if not, try without reasoning
                logger.debug(str(e))
                logger.warning(f"{'.'.join([q.type, q.target, q.reasoning, q.event_type])} "
                               'not found, trying without reasoning key....')
                template, template_nr = self.question_templates[q.type][q.target][q.event_type].random()
            except KeyError:
                # if still not: ¯\_(ツ)_/¯
                if ignore_missing_keys:
                    return None
                else:
                    raise YouIdiotException(f"Question templates are missing the key "
                                            f"{'.'.join([q.type, q.target, q.reasoning, q.event_type])}")
        logger.debug(f'Template: {template}')
        question_words = []
        template.reverse()
        stack = template
        while stack:
            logger.debug(f"Current stack is: {stack}")
            word = stack.pop()
            logger.debug(word)

            # option as in ()
            if word.startswith("(") and word.endswith(")"):
                new_words = self.processor.process_option(word)
                stack.extend(new_words[::-1])
            # context access
            elif word.startswith("#"):
                try:
                    new_word = str(q.question_data[word[1:]])
                except KeyError:
                    raise NotImplementedError(f"{word} is not in question data!")
                stack.append(str(new_word))
            elif word.startswith("!"):
                new_words = self.processor.process_function(word, args=q.question_data)
                stack.extend(new_words[::-1])
            else:
                question_words.append(word)
        logger.debug(question_words)
        q.realized = " ".join(" ".join(self.post_process(question_words)).split()) + " ?"
        answer = self._fix_units(q, passage)
        assert answer, f"{q}, {passage}"
        q.answer = answer

        return q.realized, q.answer
