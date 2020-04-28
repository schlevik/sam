from itertools import count

import click
from loguru import logger

from stresstest.util import load_json


@click.command()
@click.option("--predictions", type=str, default='data/predictions.json')
@click.option("--gold", type=str, default='data/stresstest.json')
# @click.option("--output", type=str, default="data/stresstest.json")
@click.option("--metric", type=str, default='em')
def evaluate(predictions, gold, metric):
    gold = load_json(gold)
    predictions = load_json(predictions)
    sample_metrics = []
    for i, (gold_sample, predictions_sample) in enumerate(zip(gold, predictions)):
        click.echo(f"Evaluating Sample #{click.style(str(i), fg='green', bold=True)}")
        em = 0
        num_questions = count()
        for story in gold_sample:
            story_id = story['id']

            logger.debug(f"Passage: {story['passage']}")
            for qa in story['qas']:
                next(num_questions)
                question_id = qa['id']
                answer = qa['answer']
                answer = answer or ''
                prediction = predictions_sample[story_id][question_id]
                logger.debug(f"Question: {qa['question']}")
                logger.debug(f"Answer: {prediction}")
                logger.debug(f"Gold: {answer}")
                if "".join(str(answer).lower().split()) == "".join(str(prediction).lower().split()):
                    em += 1
        sample_metrics.append(em / next(num_questions))
        click.echo(f"Result: {sample_metrics[-1]}")
