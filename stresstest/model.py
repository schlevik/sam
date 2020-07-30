import collections
import string
from typing import Iterable

from allennlp.predictors import Predictor
from loguru import logger
from overrides import overrides
from transformers import AlbertForQuestionAnswering, AlbertTokenizer
from transformers.data.metrics.squad_metrics import _get_best_indexes

from stresstest.classes import Model, Entry, Choices


def _filter_albert(input, start_logits, end_logits, start_indexes, end_indexes, bad_tokens=(2, 3)):
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
    )
    max_len = len(input)
    prelim_predictions = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if start_index >= max_len:
                continue
            if end_index >= max_len:
                continue
            # if start_index not in feature.token_to_orig_map:
            #     continue
            # if end_index not in feature.token_to_orig_map:
            #     continue
            # if not feature.token_is_max_context.get(start_index, False):
            #     continue
            if input[start_index] in bad_tokens:
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            # if length > max_answer_length:
            #     continue
            prelim_predictions.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logits[start_index],
                    end_logit=end_logits[end_index],
                )
            )

    prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

    return prelim_predictions


class NaQANet(Model):
    @overrides
    def predict(self, question, passage):
        # TODO: sanitize predictions in a way that is useful
        ans = self.predictor.predict(question=question, passage=passage)
        ans_type = ans['answer']['answer_type']
        if ans_type == 'count':
            ans_value = ans['answer']['count']
        else:
            try:
                ans_value = ans['answer']['value']
            except KeyError:
                ans_value = ans['answer']
                ans_value.pop('answer_type')
        return f"{ans_value}"

    @classmethod
    def make(cls, path, gpu=False):
        from allennlp_models.rc.models.naqanet import NumericallyAugmentedQaNet
        cuda_device = 0 if gpu else -1
        predictor = Predictor.from_path(path, cuda_device=cuda_device)
        return NaQANet("NaQAnet", predictor, gpu=gpu)


class BiDAF(Model):
    @classmethod
    def make(cls, path, gpu=False):
        try:
            from allennlp_models.rc.models.bidaf import BidirectionalAttentionFlow
        except:
            import allennlp_models.rc
        cuda_device = 0 if gpu else -1
        return cls("BiDAF", Predictor.from_path(path, cuda_device=cuda_device), gpu=gpu)

    def predict_batch(self, batch: Iterable[Entry]) -> Iterable[str]:
        return [
            p['best_span_str'] for p in
            self.predictor.predict_batch_json({"question": e.question, "passage": e.passage} for e in batch)
        ]


class Albert(Model):
    def __init__(self, name, path: str, gpu=False):
        self.tokenizer = AlbertTokenizer.from_pretrained(path)
        pretrained_albert_model = AlbertForQuestionAnswering.from_pretrained(path)
        super().__init__(name, pretrained_albert_model, gpu)
        if self.gpu:
            self.predictor.cuda()

    def _match(self, passage, result):
        result = result.replace("<pad>", "")
        result = result.replace("<unk>", "")

        result = [c for c in result.lower() if c not in string.whitespace]

        if not result:
            return 0, 0
        j = 0
        start = 0
        in_string = False
        for i, c in enumerate(passage.lower()):
            if c not in string.whitespace:
                if c != result[j]:
                    in_string = False
                    j = 0
                if c == result[j]:
                    j += 1
                    if not in_string:
                        in_string = True
                        start = i
                if j >= len(result):
                    return start, i + 1

    @classmethod
    def make(cls, path, gpu=False) -> Model:
        return cls("AlBERT", path, gpu)

    def post_process(self, question, passage, tokens):
        result = " ".join(t for t in tokens if not (t == "[CLS]" or t == "[SEP]"))
        logger.debug(f"Albert prediction: {result}")
        try:
            result = f'{question} {passage}'[slice(*self._match(f'{question} {passage}', result))]
            logger.debug(f"After matching: {result}")
        except:
            raise ValueError(f"This should not happen! Question: {question} Passage: {passage} prediction: {result}")
        return result

    def move_to_gpu(self, input_dict):
        if self.gpu:
            return {k: v.cuda() for k, v in input_dict}
        else:
            return input_dict

    @overrides
    def predict(self, entry: Entry):
        question = entry.question
        passage = entry.passage
        d = self.tokenizer.encode_plus(question.lower(), passage.lower(), return_tensors='pt')

        s, e = self.predictor(**self.move_to_gpu(d))
        logger.debug(s)
        start_indices = _get_best_indexes(s[0], 5)
        end_indices = _get_best_indexes(e[0], 5)
        logger.debug(f"Start indices: {start_indices}")
        logger.debug(f"End indices: {end_indices}")
        results = _filter_albert(d['input_ids'][0], s[0], e[0], start_indices, end_indices,
                                 bad_tokens=[self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                                             self.tokenizer.cls_token_id, self.tokenizer.sep_token_id])

        best_s, best_e, *_ = results[0]
        tokens = [
            self.tokenizer.decode(int(i))
            for i in d['input_ids'][0][best_s:best_e + 1]
        ]
        logger.debug(f"Start index: {s.argmax()}")
        logger.debug(f"End index: {e.argmax()}")
        logger.debug(f"Tokens: {tokens}")
        return self.post_process(question, passage, tokens)

    @overrides
    def predict_batch(self, batch: Iterable[Entry]) -> Iterable[str]:
        # TODO: so this is shit
        batch = list(batch)
        d = self.tokenizer.batch_encode_plus(((e.question.lower(), e.passage.lower()) for e in batch),
                                             return_tensors='pt', pad_to_max_length=True)

        ss, es = self.predictor(**self.move_to_gpu(d))
        result = []
        # technically, this is shit...
        for j, (start, end, entry) in enumerate(zip(ss, es, batch)):
            tokens = [
                self.tokenizer.decode(int(i))
                for i in d['input_ids'][j][start.argmax():end.argmax() + 1]
            ]
            result.append(self.post_process(entry.question, entry.passage, tokens))
        return result

        # try:
        #     result = f'{question} {passage}'[slice(*self._match(f'{question} {passage}', result))]
        #     logger.debug(f"After matching: {result}")
        # except:
        #     raise ValueError(f"This should not happen! Question: {question} Passage: {passage} prediction: {result}")
        # return result


class RandomBaseline(Model):
    def __init__(self, name):
        super().__init__(name, None, False)

    @classmethod
    def make(cls, path=None, gpu=False):
        return cls("Random")

    def predict(self, entry: Entry) -> str:
        events = entry.qa['events']

        candidates = [str(a) if isinstance(a, int) else f"{a['first']} {a['last']}"
                      for e in events for a in list(e['attributes'].values()) + [e['actor']]]
        choices = Choices([c for c in candidates if c in entry.passage])

        return choices.random()

    def predict_batch(self, batch: Iterable[Entry]) -> Iterable[str]:
        return (self.predict(entry) for entry in batch)


class EducatedBaseline(Model):
    def __init__(self, name):
        super().__init__(name, None, False)

    @classmethod
    def make(cls, path=None, gpu=False):
        return cls("Educated")

    def predict(self, entry: Entry) -> str:
        events = entry.qa['events']
        answer_is_numbers = any(d in entry.answer for d in string.digits)
        if answer_is_numbers:
            candidates = [str(a) for e in events for a in list(e['attributes'].values()) if isinstance(a, int)]
        else:
            candidates = [f"{a['first']} {a['last']}" for e in events for a in
                          list(e['attributes'].values()) + [e['actor']] if isinstance(a, dict)]
        choices = Choices([c for c in candidates if c in entry.passage])
        return choices.random()

    def predict_batch(self, batch: Iterable[Entry]) -> Iterable[str]:
        return (self.predict(entry) for entry in batch)


class InformedBaseline(Model):
    def __init__(self, name):
        super().__init__(name, None, False)

    @classmethod
    def make(cls, path=None, gpu=False):
        return cls("Educated")

    def predict(self, entry: Entry) -> str:
        events = entry.qa['events']
        answer_is_numbers = any(d in entry.answer for d in string.digits)
        question_is_comparison = 'comparison' in entry.qa['reasoning']
        if question_is_comparison:
            candidates = [f"{a['first']} {a['last']}" for e in events for a in
                          list(e['attributes'].values()) + [e['actor']] if isinstance(a, dict)]
            candidates = [c for c in candidates if c in entry.question]
            candidates = list(set(candidates))
            assert all(c in entry.passage for c in candidates)
            assert len(candidates) == 2
        elif answer_is_numbers:
            candidates = [str(a) for e in events for a in list(e['attributes'].values()) if isinstance(a, int)]
        else:
            candidates = [f"{a['first']} {a['last']}" for e in events for a in
                          list(e['attributes'].values()) + [e['actor']] if isinstance(a, dict)]

        choices = Choices([c for c in candidates if c in entry.passage])
        return choices.random()

    def predict_batch(self, batch: Iterable[Entry]) -> Iterable[str]:
        return (self.predict(entry) for entry in batch)
