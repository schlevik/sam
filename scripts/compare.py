import math
from operator import itemgetter
import numpy as np
import click
import quickconf
from typing import Iterable

from loguru import logger

from stresstest.textmetrics import pointwise_average_distance, Distance
from stresstest.util import load_json


@click.command()
@click.option("--input", type=str, default="data/stresstest.json")
@click.option("--flat", default=False, is_flag=True)
@click.option("--attr", type=str, default='passage')
@click.option("--metric", type=str, default='Jaccard')
def compare(input, flat, attr, metric):
    metric_class = quickconf.load_class(metric, restrict_to=Distance, relative_import='stresstest.textmetrics')
    paragraphs = load_json(input)
    distances = []
    getter = itemgetter(attr)
    if flat:
        paragraphs = [paragraphs]
    logger.debug(f"Evaluating {len(paragraphs)} sample(s).")
    for i, sample in enumerate(paragraphs):
        logger.debug(f"Sample #{i} has {len(sample)} paragraphs.")
        corpus: Iterable[str] = (getter(s) for s in sample)
        result = pointwise_average_distance(corpus, metric_class())
        result = np.array(result)
        # printable_result = f'{result.mean()} ± {2 * result.std() / math.sqrt(len(result))}'
        distances.append(result.mean())
    distances = np.array(distances)
    printable_result = f'{distances.mean():.4f} ± {2 * distances.std() / math.sqrt(len(distances)):.4f}'
    click.echo(f"Pointwise average distance under the {click.style(metric, fg='green')} class: "
               f"{click.style(printable_result, fg='green', bold=True)}")


@click.command()
@click.option("--input", type=str, default="data/drop_dataset_dev.json")
@click.option("--output", type=str, default="data/drop_nfl.json")
def process_drop(input, output):
    import ujson as json
    with open(input) as f:
        data = json.load(f)
    drop = [v for k, v in data.items() if "nfl" in k]
    with open(output, "w+") as f:
        json.dump(drop, f)
