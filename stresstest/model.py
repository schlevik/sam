from typing import Tuple

from allennlp.predictors import Predictor
from allennlp.pretrained import bidirectional_attention_flow_seo_2017
from allennlp.pretrained import naqanet_dua_2019


def bidaf() -> Tuple[str, Predictor]:
    return "BiDAF", bidirectional_attention_flow_seo_2017()


def naqanet() -> Tuple[str, Predictor]:
    return "NaQAnet", naqanet_dua_2019()
