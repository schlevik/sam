import json
from collections import defaultdict

import click
import quickconf
from loguru import logger
from tqdm import tqdm

from stresstest.classes import Model
from stresstest.util import load_json, sample_iter, num_questions


@click.command()
@click.option("--input", type=str, default='data/stresstest.json')
@click.option("--output", type=str, default="data/predictions.json")
@click.option("--model", type=str, default='models/model.tar.gz')
@click.option("--cls", type=str, default=None)
@click.option("--gpu", default=None, type=bool)
def predict(input, output, model, cls, gpu):
    logger.debug(gpu)
    if gpu is None:
        import torch
        try:
            gpu = torch.cuda.is_available()
            if gpu:
                str(torch.rand(1).to(torch.device("cuda")))
        except:
            gpu = False
    sample = load_json(input)
    num_q = num_questions(sample)
    click.echo(f"Evaluating {click.style(model)} on  sample (n={num_q}, |{{C}}|={len(sample)}). "
               f"Running on {click.style('gpu' if gpu else 'cpu', fg='green', bold=True)}.")
    model_cls: Model = quickconf.load_class(cls, relative_import='stresstest.model', restrict_to=Model)
    model = model_cls.make(model, gpu=gpu)

    predictions = defaultdict(dict)
    for passage_id, passage, question_id, question, *_ in tqdm(sample_iter(sample), position=1, total=num_q):
        # TODO: maybe something something batch
        logger.debug(f"Passage:{passage}, Question: {question}")
        answer = model.predict(question=question, passage=passage)
        predictions[passage_id][question_id] = str(answer)

    with open(output, "w+") as f:
        json.dump(predictions, f)
