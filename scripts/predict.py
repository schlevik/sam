import json
from collections import defaultdict

import click
import quickconf
from loguru import logger

from stresstest.classes import Model
from stresstest.util import load_json, sample_iter


@click.command()
@click.option("--input", type=str, default='data/stresstest.json')
@click.option("--output", type=str, default="data/predictions.json")
@click.option("--model", type=str, default='models/model.tar.gz')
@click.option("--cls", type=str, default=None)
@click.option("--gpu", default='auto', type=bool)
def predict(input, output, model, cls, gpu):
    if gpu == 'auto':
        try:
            from torch.cuda import is_available
            gpu = is_available()
        except:
            gpu = False
    data = load_json(input)
    model_cls: Model = quickconf.load_class(cls, relative_import='stresstest.model', restrict_to=Model)
    model = model_cls.make(model, gpu=gpu)
    predictions = []
    for sample in data:
        sample_predictions = defaultdict(dict)
        for passage_id, passage, question_id, question, _ in sample_iter(sample):
            # maybe something something batch
            logger.debug(f"Passage:{passage}, Question: {question}")
            answer = model.predict(question=question, passage=passage)
            sample_predictions[passage_id][question_id] = str(answer)
        predictions.append(sample_predictions)
    with open(output, "w+") as f:
        json.dump(predictions, f)
