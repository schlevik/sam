import math
from operator import itemgetter
import numpy as np
import click
import quickconf
from typing import Iterable
from loguru import logger
import json

from scipy.stats import t

from stresstest.textmetrics import pointwise_average_distance, Distance
from stresstest.util import load_json


@click.command()
@click.option("--input", type=str, default="data/stresstest.json")
@click.option("--output", type=str, default="metrics/compare.json")
@click.option("--flat", default=False, is_flag=True)
@click.option("--attr", type=str, default='passage')
@click.option("--metric", type=str, default='Jaccard')
def compare(input, output, flat, attr, metric):
    # TODO: make metrics appendable
    metric_class = quickconf.load_class(metric, restrict_to=Distance, relative_import='stresstest.textmetrics')
    samples = load_json(input)
    distances = []
    getter = itemgetter(attr)
    if flat:
        samples = [samples]
    logger.debug(f"Evaluating {len(samples)} sample(s).")
    for i, sample in enumerate(samples):
        logger.debug(f"Sample #{i} has {len(sample)} paragraphs.")
        corpus: Iterable[str] = (getter(s) for s in sample)
        result = pointwise_average_distance(corpus, metric_class())
        result = np.array(result)
        # printable_result = f'{result.mean()} ± {2 * result.std() / math.sqrt(len(result))}'
        distances.append(result.mean())
    distances = np.array(distances)
    t_975 = t.ppf(1 - .025, df=len(samples) - 1)
    printable_result = f'{distances.mean():.4f} ± {t_975 * distances.std() / math.sqrt(len(distances)):.4f}'
    click.echo(f"Pointwise average distance under the {click.style(metric, fg='green')} metric: "
               f"{click.style(printable_result, fg='green', bold=True)}")
    result = {
        "num_samples": len(samples),
        metric: {
            'human_readable': printable_result,
            'mean': distances.mean(),
            'variance': distances.var(),
            '±(95ci)': t_975 * distances.std() / math.sqrt(len(distances))
        }
    }
    with open(output, "w+") as f:
        json.dump(result, f, indent=4, separators=(',', ': '))


@click.command()
@click.option("--input", type=str, default="data/drop_dataset_dev.json")
@click.option("--output", type=str, default="data/drop_nfl.json")
def process_drop(input, output):
    with open(input) as f:
        data = json.load(f)
    drop = [v for k, v in data.items() if "nfl" in k]
    with open(output, "w+") as f:
        json.dump(drop, f)
