from itertools import count

from loguru import logger

from scipy.stats import t
import numpy as np
import math


def eval_em(gold, predictions):
    em = 0
    # num_questions = count()
    i = None
    for i, (story_id, story, question_id, question, answer) in enumerate(gold):
        # for story in gold:
        # story_id = story['id']

        logger.debug(f"Passage: {story}")
        # for qa in story['qas']:
        # next(num_questions)
        # question_id = qa['id']
        # answer = qa['answer']
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


def get_mean_var_ci(sample, ci=0.025):
    sample = np.array(sample)
    t_ci = t.ppf(1 - ci, df=len(sample) - 1)
    return sample.mean(), sample.var(), t_ci * sample.std() / math.sqrt(len(sample))
