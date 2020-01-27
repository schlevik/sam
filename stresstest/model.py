from typing import Tuple

from allennlp.predictors import Predictor
from allennlp.pretrained import bidirectional_attention_flow_seo_2017
from allennlp.pretrained import naqanet_dua_2019
from transformers import AlbertForQuestionAnswering, AlbertTokenizer


def bidaf() -> Tuple[str, Predictor]:
    return "BiDAF", bidirectional_attention_flow_seo_2017()


def naqanet() -> Tuple[str, Predictor]:
    return "NaQAnet", naqanet_dua_2019()


class Albert:
    def __init__(self, path: str, gpu=False):
        self.tokenizer = AlbertTokenizer.from_pretrained(path)
        self.model = AlbertForQuestionAnswering.from_pretrained(path)
        self.gpu = gpu

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
