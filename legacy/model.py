from allennlp.pretrained import bidirectional_attention_flow_seo_2017
from allennlp.pretrained import naqanet_dua_2019
from overrides import overrides
from quicklog import Loggable
from transformers import AlbertForQuestionAnswering, AlbertTokenizer


class Model(Loggable):
    """
    Unifying interface to all those stupid things.
    """

    def __init__(self, name, torch_model, gpu=False):
        self.name = name
        self.model = torch_model
        self.gpu = gpu

    def predict(self, question, passage):
        # TODO: something something GPU
        return self.model.predict(question=question, passage=passage)[
            'best_span_str']


class NaQANet(Model):
    @overrides
    def predict(self, question, passage):
        ans = self.model.predict(question=question, passage=passage)
        ans_type = ans['answer']['answer_type']
        if ans_type == 'count':
            ans_value = ans['answer']['count']
        else:
            try:
                ans_value = ans['answer']['value']
            except KeyError:
                ans_value = ans['answer']
                ans_value.pop('answer_type')
        return f"{ans_value} ({ans_type})"


def bidaf() -> Model:
    return Model("BiDAF", bidirectional_attention_flow_seo_2017())


def naqanet() -> Model:
    return NaQANet("NaQAnet", naqanet_dua_2019())


def albert(path, gpu=False) -> Model:
    return Albert("AlBERT", path, gpu)


class Albert(Model):
    def __init__(self, name, path: str, gpu=False):
        self.tokenizer = AlbertTokenizer.from_pretrained(path)
        model = AlbertForQuestionAnswering.from_pretrained(path)
        super().__init__(name, model, gpu)
        if self.gpu:
            self.model.cuda()

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

        s, e = self.model(token_type_ids=token_type_ids,
                          input_ids=input_ids)
        return " ".join(
            [
                self.tokenizer.decode(int(i))
                for i in d['input_ids'][0][s.argmax():e.argmax() + 1]
            ]
        )
