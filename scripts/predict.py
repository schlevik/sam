import sys
from typing import List

import click
from loguru import logger
from tqdm import tqdm

from scripts.utils import write_json, get_output_predictions_file_name
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
        # TODO: Bidaf should also respect max answer length
        model = model_cls.make(weights_path, gpu=gpu)
        click.echo(f"Evaluating model '{click.style(model_cls.__name__, fg='green', bold=True)}' from weights file: "
                   f"{click.style(weights_path, fg='blue')}.")
        click.echo(f"Running on {click.style('gpu' if gpu else 'cpu', fg='green', bold=True)}.")
        for in_file in in_files:
            sample = load_json(in_file)
            num_q = num_questions(sample)
            click.echo(
                f"Evaluating on sample (n={num_q}, |{{C}}|={len(sample)}): {click.style(in_file, fg='blue')}")

            predictions = dict()
            for sample_batch in batch(tqdm(sample_iter(sample), position=1, total=num_q), batch_size=batch_size):
                sample_batch: List[Entry]
                batch_predictions = model.predict_batch(sample_batch)
                for entry, answer in zip(sample_batch, batch_predictions):
                    logger.debug(f"Passage: {entry.passage}")
                    logger.debug(f"Question: {entry.question}")
                    logger.debug(f"Prediction: {answer}")
                    predictions[entry.qa_id] = str(answer)
            output_file_name = get_output_predictions_file_name(in_file, output_folder, weights_path)
            click.echo(f"Saving predictions to {click.style(output_file_name, fg='blue')}")
            write_json(predictions, output_file_name, pretty=False)
