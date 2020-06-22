from scipy.stats import t
import numpy as np
import math

from loguru import logger


def em(gold, predictions):
    em = 0
    # num_questions = count()
    i = None
    for i, (story_id, story, question_id, question, answer) in enumerate(gold, 1):
        logger.debug(f"Passage: {story}")
        answer = answer or ''
        prediction = predictions[story_id][question_id]
        logger.debug(f"Question: {question}")
        logger.debug(f"Answer: {prediction}")
        logger.debug(f"Gold: {answer}")
        if "".join(str(answer).lower().split()) == "".join(str(prediction).lower().split()):
            em += 1
    if i:
        result = em / i
    else:
        logger.warning(f"Evaluating on empty gold!")
        result = 0
    logger.debug(f'Result EM: {result}')
    return result


def f1(gold, predictions):
    overall_f1 = 0
    i = 0
    for i, (story_id, story, question_id, question, answer) in enumerate(gold, 1):
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
        overall_f1 += f1
    if i:
        result = overall_f1 / i
    else:
        logger.warning(f"Evaluating on empty gold!")
        return 0
    return result


def get_mean_var_ci(sample, alpha=0.025):
    sample = np.array(sample)
    t_ci = t.ppf(1 - alpha, df=len(sample) - 1)
    return sample.mean(), sample.var(), t_ci * sample.std() / math.sqrt(len(sample))
