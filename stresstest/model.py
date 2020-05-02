from allennlp.predictors import Predictor
from overrides import overrides
from transformers import AlbertForQuestionAnswering, AlbertTokenizer

from stresstest.classes import Model


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
        # TODO: here, we'll probably need
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


class Albert(Model):
    def __init__(self, name, path: str, gpu=False):
        self.tokenizer = AlbertTokenizer.from_pretrained(path)
        pretrained_albert_model = AlbertForQuestionAnswering.from_pretrained(path)
        super().__init__(name, pretrained_albert_model, gpu)
        if self.gpu:
            self.predictor.cuda()

    @classmethod
    def make(cls, path, gpu=False) -> Model:
        return cls("AlBERT", path, gpu)

    @overrides
    def predict(self, question, passage):
        d = self.tokenizer.encode_plus(question.lower(), passage.lower(),
                                       return_tensors='pt')
        if self.gpu:
            token_type_ids = d['token_type_ids'].cuda()
            input_ids = d['input_ids'].cuda()
        else:
            input_ids = d['input_ids']
            token_type_ids = d['token_type_ids']

        s, e = self.predictor(token_type_ids=token_type_ids,
                              input_ids=input_ids)
        return " ".join(
            [
                self.tokenizer.decode(int(i))
                for i in d['input_ids'][0][s.argmax():e.argmax() + 1]
            ]
        )
