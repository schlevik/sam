from abc import abstractmethod, ABC
from typing import Dict, Iterable

from scipy.stats import t
import numpy as np
import math

from loguru import logger
from string import digits

from statsmodels.stats.proportion import proportion_confint

from stresstest.classes import Entry


class EvalMetric(ABC):
    @property
    @abstractmethod
    def binary(self):
        pass

    @abstractmethod
    def __call__(self, gold: Iterable[Entry], predictions: Dict[str, Dict[str, str]], *args, **kwargs):
        ...


class EM(EvalMetric):
    binary = True

    def __init__(self, relaxed=False, max_length=3):
        self.relaxed = relaxed
        self.max_length = max_length

    def __call__(self, gold: Iterable[Entry], predictions: Dict[str, Dict[str, str]], **kwargs):
        em = []
        for i, (story_id, story, question_id, question, answer, _) in enumerate(gold, 1):
            logger.debug(f"Passage: {story}")
            answer = answer or ''
            prediction = predictions[story_id][question_id]
            logger.debug(f"Question: {question}")
            logger.debug(f"Answer: {prediction}")
            logger.debug(f"Gold: {answer}")
            if (self.relaxed and len(prediction.split()) <= self.max_length and
                    set(answer).intersection(digits) and
                    "".join(c for c in str(answer) if c in digits) == ''.join(
                        c for c in str(prediction) if c in digits)):
                logger.debug("Correct (relaxed)!")
                em.append(1)
            elif "".join(str(answer).lower().split()) == "".join(str(prediction).lower().split()):
                logger.debug("Correct!")
                em.append(1)
            else:
                logger.debug("Wrong!")
                em.append(0)
        else:
            logger.warning(f"Evaluating on empty gold!")
        logger.debug(f'Mean EM: {sum(em) / len(em)}')
        return em


class F1(EvalMetric):
    binary = False

    def __call__(self, gold: Iterable[Entry], predictions: Dict[str, Dict[str, str]], **kwargs):
        overall_f1 = []
        for i, (story_id, story, question_id, question, answer, _) in enumerate(gold, 1):
            gold_tokens = set(str(answer).lower().split(" "))
            prediction_tokens = set(str(predictions[story_id][question_id]).lower().split(" "))
            logger.debug(f"Question: {question}")

            logger.debug(f"Answer Tokens: {gold_tokens}")
            logger.debug(f"Prediction Tokens: {prediction_tokens}")

            tp = len(gold_tokens.intersection(prediction_tokens))
            fp = len(prediction_tokens) - tp
            fn = len(gold_tokens) - tp
            logger.debug(f"TP: {tp}")
            logger.debug(f"FP: {fp}")
            logger.debug(f"FN: {fn}")

            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0

            logger.debug(f"Precision: {precision}")
            logger.debug(f"Recall: {recall}")

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            logger.debug(f"F1: {f1}")
            overall_f1.append(f1)
        else:
            logger.warning(f"Evaluating on empty gold!")
        return overall_f1


def get_mean_var_ci(sample, alpha=0.025):
    sample = np.array(sample)
    t_ci = t.ppf(1 - alpha, df=len(sample) - 1)
    return sample.mean(), sample.var(), t_ci * sample.std() / math.sqrt(len(sample))


def get_mean_var_ci_bernoulli(sample, alpha=0.05):
    lower, _ = proportion_confint(sum(sample), len(sample), alpha=alpha)
    mean = sum(sample) / len(sample)
    return mean, None, mean - lower
