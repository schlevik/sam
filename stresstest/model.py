import itertools
import string
from typing import Iterable

from allennlp.predictors import Predictor
from loguru import logger
from overrides import overrides
from transformers import AlbertForQuestionAnswering, AlbertTokenizer

from stresstest.classes import Model, Entry


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
        from allennlp_models.rc.qanet.naqanet_model import NumericallyAugmentedQaNet
        cuda_device = 0 if gpu else -1
        predictor = Predictor.from_path(path, 'reading-comprehension', cuda_device=cuda_device)
        return NaQANet("NaQAnet", predictor, gpu=gpu)


class BiDAF(Model):
    @classmethod
    def make(cls, path, gpu=False):
        from allennlp_models.rc.bidaf.bidaf_model import BidirectionalAttentionFlow
        cuda_device = 0 if gpu else -1
        return cls("BiDAF", Predictor.from_path(path, 'reading-comprehension', cuda_device=cuda_device), gpu=gpu)

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
        if not result:
            return 0, 0
        result = [c for c in result.lower() if c not in string.whitespace]
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
    def predict(self, question, passage):
        d = self.tokenizer.encode_plus(question.lower(), passage.lower(), return_tensors='pt')

        s, e = self.predictor(**self.move_to_gpu(d))

        tokens = [
            self.tokenizer.decode(int(i))
            for i in d['input_ids'][0][s.argmax():e.argmax() + 1]
        ]
        return self.post_process(question, passage, tokens)

    @overrides
    def predict_batch(self, batch: Iterable[Entry]) -> Iterable[str]:
        batch = list(batch)
        d = self.tokenizer.batch_encode_plus(((e.question.lower(), e.passage.lower()) for e in batch),
                                             return_tensors='pt', pad_to_max_length=True)
        if self.gpu:
            token_type_ids = d['token_type_ids'].cuda()
            input_ids = d['input_ids'].cuda()
        else:
            input_ids = d['input_ids']
            token_type_ids = d['token_type_ids']

        ss, es = self.predictor(token_type_ids=token_type_ids,
                                input_ids=input_ids)
        result = []
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
