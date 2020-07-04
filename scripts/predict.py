import json
import os
import sys
from collections import defaultdict
from typing import Iterable, List

import click
import quickconf
from loguru import logger
from tqdm import tqdm

from stresstest.classes import Model, Entry
from stresstest.util import load_json, sample_iter, num_questions, fmt_dict, do_import, batch


def _is_gpu_available():
    import torch
    try:
        gpu = torch.cuda.is_available()
        if gpu:
            str(torch.rand(1).to(torch.device("cuda")))
    except:
        gpu = False
    return gpu


@click.command()
@click.argument("in_files", nargs=-1, type=str)
@click.option('models', "--model", envvar="MODELS", multiple=True, type=click.Path())
@click.option('model_classes', "--cls", envvar="CLASSES", multiple=True, type=str, default=None)
@click.option("--gpu", default=None, type=bool)
@click.option("--batch-size", default=8, type=int)
@click.option("--output-folder", type=str, default="data/predictions")
def predict(in_files, output_folder, models, model_classes, gpu, batch_size):
    # There is a chance i'll need to scrap all of this and do convert to features stuff
    if gpu is None:
        gpu = _is_gpu_available()
    logger.debug(fmt_dict(locals()))
    if not len(models) == len(model_classes):
        click.echo(f"Num models supplied ({len(models)})!= num model classes supplied ({len(model_classes)})!")
        sys.exit(1)

    for cls, weights_path in zip(model_classes, models):
        model_cls: Model = do_import(cls, relative_import='stresstest.model')
        model = model_cls.make(weights_path, gpu=gpu)
        click.echo(f"Evaluating model '{click.style(model_cls.__name__, fg='green', bold=True)}' from weights file: "
                   f"{click.style(weights_path, fg='blue')}.")
        click.echo(f"Running on {click.style('gpu' if gpu else 'cpu', fg='green', bold=True)}.")
        for in_file in in_files:
            sample = load_json(in_file)
            num_q = num_questions(sample)
            click.echo(
                f"Evaluating on sample (n={num_q}, |{{C}}|={len(sample)}): {click.style(in_file, fg='blue')}")

            predictions = defaultdict(dict)
            for sample_batch in batch(tqdm(sample_iter(sample), position=1, total=num_q), batch_size=batch_size):
                sample_batch: List[Entry]
                batch_predictions = model.predict_batch(sample_batch)
                for entry, answer in zip(sample_batch, batch_predictions):
                    logger.debug(f"Passage: {entry.passage}")
                    logger.debug(f"Question: {entry.question}")
                    logger.debug(f"Prediction: {answer}")
                    predictions[entry.id][entry.qa_id] = str(answer)
            output_base = os.path.splitext(os.path.basename(in_file))[0]
            weights_addon = os.path.basename(weights_path)
            weights_addon = weights_addon.replace(".tar", "").replace(".gz", "").replace(".zip", "").replace(".tgz", "")
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            output = os.path.join(output_folder, f"{output_base}-{weights_addon}.json")
            click.echo(f"Saving predictions to {click.style(output, fg='blue')}")
            with open(output, "w+") as f:
                json.dump(predictions, f)
