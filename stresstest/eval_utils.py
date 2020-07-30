import string
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Dict, Iterable

from scipy.stats import t
import numpy as np
import math

from loguru import logger
from string import digits

from statsmodels.stats.proportion import proportion_confint

from stresstest.classes import Entry
from stresstest.print_utils import highlight
from stresstest.util import sample_iter


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

    def __call__(self, gold: Iterable[Entry], predictions: Dict[str, str], **kwargs):
        em = []
        for i, (story_id, story, question_id, question, answer, _) in enumerate(gold, 1):
            logger.debug(f"Passage: {story}")
            answer = answer or ''
            prediction = predictions[question_id]
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
        if em:
            logger.debug(f'Mean EM: {sum(em) / len(em)}')
        else:
            logger.warning(f"Evaluating on empty gold!")
        return em


class F1(EvalMetric):
    binary = False

    def __call__(self, gold: Iterable[Entry], predictions: Dict[str, str], **kwargs):
        overall_f1 = []
        for i, (story_id, story, question_id, question, answer, _) in enumerate(gold, 1):
            gold_tokens = set(str(answer).lower().split(" "))
            prediction_tokens = set(str(predictions[question_id]).lower().split(" "))
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
        if overall_f1:
            logger.debug(f'Mean EM: {sum(overall_f1) / len(overall_f1)}')
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


def evaluate_intervention(aligned_baseline, aligned_intervention, aligned_control,
                          predictions_baseline, predictions_intervention, predictions_control):
    longest_answer = max(len(b.answer.split()) for b in aligned_baseline + aligned_intervention)
    em = EM(relaxed=True, max_length=longest_answer)
    results_baseline = em(aligned_baseline, predictions_baseline)

    # got it correct in first place
    correct_before_intervention = [
        (d, aligned_intervention[i], predictions_baseline[d.qa_id], predictions_intervention[d.qa_id]) for i, d
        in enumerate(aligned_baseline) if results_baseline[i] == 1
    ]

    results_intervention = em(aligned_intervention, predictions_intervention)

    # changed prediction correctly in line with altered semantics
    # model right -model changes prediction- model right
    correct_change_correct = [
        (d, aligned_intervention[i], predictions_baseline[d.qa_id], predictions_intervention[d.qa_id]) for i, d
        in enumerate(aligned_baseline) if results_baseline[i] == 1 and results_intervention[i] == 1
    ]

    # unimpressed: didn't change prediction although semantics dictated change
    # model right -model keeps prediction- model wrong
    correct_keep_wrong = [
        (d, aligned_intervention[i], predictions_baseline[d.qa_id], predictions_intervention[d.qa_id]) for i, d
        in
        enumerate(aligned_baseline) if
        predictions_baseline[d.qa_id] ==
        predictions_intervention[d.qa_id].replace("almost ", "").replace("nearly ", "")
        and results_baseline[i]
    ]

    # confused: changed prediction when semantics dictated change but changed to wrong
    # model right -model changes prediction- model wrong
    correct_change_wrong = [
        (d, aligned_intervention[i], predictions_baseline[d.qa_id], predictions_intervention[d.qa_id]) for i, d
        in
        enumerate(aligned_baseline) if
        predictions_baseline[d.qa_id] !=
        predictions_intervention[d.qa_id].replace("almost ", "").replace("nearly ", "")
        and results_baseline[i] and not results_intervention[i]
    ]
    # model wrong -model changes prediction- model right
    wrong_change_right = [
        (d, aligned_intervention[i], predictions_baseline[d.qa_id], predictions_intervention[d.qa_id]) for i, d
        in
        enumerate(aligned_baseline) if
        predictions_baseline[d.qa_id] !=
        predictions_intervention[d.qa_id].replace("almost ", "").replace("nearly ", "")
        and not results_baseline[i] and results_intervention[i]
    ]
    # model wrong -model keeps prediction- model right
    wrong_keep_right = [
        (d, aligned_intervention[i], predictions_baseline[d.qa_id], predictions_intervention[d.qa_id]) for i, d
        in
        enumerate(aligned_baseline) if
        results_intervention[i] and not results_baseline[i] and
        predictions_baseline[d.qa_id] == predictions_intervention[d.qa_id]
    ]
    if predictions_control:

        results_control = em(aligned_control, predictions_control)
        correct_baseline_control = [
            (d, aligned_control[i], predictions_baseline[d.qa_id], predictions_control[d.qa_id]) for i, d
            in enumerate(aligned_baseline) if results_baseline[i] == 1 and results_control[i] == 1
        ]
        correct_baseline_control_intervention = [
            (d, aligned_intervention[i], aligned_control[i], predictions_baseline[d.qa_id],
             predictions_intervention[d.qa_id], predictions_control[d.qa_id]) for i, d
            in enumerate(aligned_baseline) if (results_baseline[i] == 1 and results_intervention[i] == 1
                                               and results_control[i] == 1)
        ]
    else:
        results_control = []
        correct_baseline_control = []
        correct_baseline_control_intervention = []
    assert len(correct_before_intervention) > len(correct_change_correct)
    assert len(correct_before_intervention) == len(correct_keep_wrong) + len(correct_change_wrong) + len(
        correct_change_correct)
    assert sum(results_intervention) == len(correct_change_correct) + len(wrong_change_right) + len(
        wrong_keep_right), sum(results_intervention)
    overall_results = []
    if results_control:
        for b, i, c in zip(results_baseline, results_intervention, results_control):
            if b and c:
                overall_results.append(1 if b + i == 2 else 0)
    else:
        for b, i in zip(results_baseline, results_intervention):
            if b:
                overall_results.append(1 if b + i == 2 else 0)

    return (overall_results, results_baseline, results_intervention, results_control,
            correct_before_intervention, correct_change_correct, correct_keep_wrong, correct_change_wrong,
            wrong_change_right, wrong_keep_right, correct_baseline_control, correct_baseline_control_intervention
            )


def align(baseline, intervention, control, assert_same=False):
    gold_flat = list(sample_iter(baseline))
    gold_intervention_flat = list(sample_iter(intervention))
    q_m_by_id = {m.qa_id: m for m in gold_intervention_flat}

    if control:
        gold_control_flat = list(sample_iter(control))
        q_c_by_id = {c.qa_id: c for c in gold_control_flat}

    else:
        q_c_by_id = defaultdict(lambda: None)

    aligned = [(b, q_m_by_id[b.qa_id], q_c_by_id[b.qa_id]) for b in gold_flat if b.qa_id in q_m_by_id]
    aligned = [(b, m, c) for b, m, c in aligned if b.answer != m.answer]
    if assert_same:
        assert len(aligned) == len(gold_flat)
    aligned_baseline, aligned_intervention, aligned_control = zip(*aligned)
    for b, i, c in aligned:
        assert b.qa_id == i.qa_id
    return list(aligned_baseline), list(aligned_intervention), list(aligned_control)


def split_and_eval_by_num_modifiers(baseline, intervention, control, pba, pia, pca, n):
    aligned_n_mod = [(b, i, c) for b, i, c in zip(baseline, intervention, control) if
                     i.qa['modification_data']['modification_distance'] == n]
    baseline_n, intervention_n, control_n = zip(*aligned_n_mod)
    overall, results_baseline = evaluate_intervention(baseline_n, intervention_n, control_n, pba, pia, pca)[:2]
    mod_per_passage = \
        sum(d.qa['modification_data']['modification_distance'] if d.qa['modification_data']['fill_with_modification']
            else 1 for _, d, _ in aligned_n_mod) / len(aligned_n_mod)
    return sum(overall) / len(overall), len(baseline_n), mod_per_passage, sum(results_baseline) / len(results_baseline)


def split_and_eval_by_answer_type(baseline, intervention, control, pba, pia, pca):
    at_numbers = [(b, i, c) for b, i, c in zip(baseline, intervention, control) if
                  any(d in i.answer for d in string.digits)]
    at_ne = [(b, i, c) for b, i, c in zip(baseline, intervention, control) if
             not any(d in i.answer for d in string.digits)]
    assert at_ne
    assert at_numbers
    assert len(at_ne) + len(at_numbers) == len(baseline)
    overall_numbers = evaluate_intervention(*zip(*at_numbers), pba, pia, pca)[0]
    overall_ne = evaluate_intervention(*zip(*at_ne), pba, pia, pca)[0]
    return (
        (sum(overall_numbers) / len(overall_numbers), len(at_numbers)),
        (sum(overall_ne) / len(overall_ne), len(at_ne))
    )
