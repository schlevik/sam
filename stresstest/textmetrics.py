from abc import abstractmethod, ABC
from typing import Dict, List, Iterable

import spacy
import numpy as np
import editdistance
from joblib import Parallel, delayed
from spacy.tokens import Doc
from tqdm import tqdm


class Distance(ABC):

    @abstractmethod
    def __call__(self, text: Doc, other_text: Doc) -> float:
        ...


class Levenshtein(Distance):
    def __call__(self, text: Doc, other_text: Doc) -> float:
        return editdistance.distance(str(text), str(other_text))


class Jaccard(Distance):

    def __call__(self, text: List[str], other_text: List[str]) -> float:
        tokens = set(str(t) for t in text)
        other_tokens = set(str(t) for t in other_text)
        return (len(tokens.intersection(other_tokens)) /
                len(tokens.union(other_tokens)))


class Embedding(Distance):
    embeddings: Dict[str, np.ndarray]

    def __call__(self, text: Doc, other_text: Doc) -> float:
        pass

    def __init__(self, embeddings: str):
        # self.embeddings = ...
        ...


nlp = None


def _get_spacy():
    global nlp
    if not nlp:
        nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner'])
    return nlp


def test(text: List[str], other_text: List[str]) -> float:
    tokens = set(str(t) for t in text)
    other_tokens = set(str(t) for t in other_text)
    return (len(tokens.intersection(other_tokens)) /
            len(tokens.union(other_tokens)))


def pointwise_average_distance(corpus: Iterable[str],
                               distance: Distance) -> List[float]:
    nlp = _get_spacy()
    docs = (nlp.pipe(corpus))
    docs = [[str(t) for t in text] for text in docs]
    return Parallel(4)(delayed(distance)(text, other_text) for i, text in enumerate(tqdm(docs)) for
                        j, other_text in enumerate(docs) if i != j)
