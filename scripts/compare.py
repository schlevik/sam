import math
from operator import itemgetter
import numpy as np
import click
import quickconf
from typing import Iterable
from loguru import logger
import json

from scipy.stats import t

from scripts.utils import write_json
from stresstest.textmetrics import pointwise_average_distance, Distance
from stresstest.util import load_json


@click.command()
@click.option("--input", type=str, default="data/stresstest.json")
@click.option("--reference", type=str, default="data/drop_nfl.json")
@click.option("--output", type=str, default="metrics/compare.json")
# @click.option("--flat", default=False, is_flag=True)
@click.option("--attr", type=str, default='passage')
@click.option("--metric", type=str, default='Jaccard')
def diversity(input, reference, output, attr, metric):
    # TODO: make metrics appendable
    metric_class = quickconf.load_class(metric, restrict_to=Distance, relative_import='stresstest.textmetrics')
    samples = load_json(input)
    reference = load_json(reference)
    distances = []
    getter = itemgetter(attr)
    # if flat:
    #    samples = [samples]
    logger.debug(f"Evaluating {len(samples)} sample(s).")
    # samples
    for i, sample in enumerate(samples):
        logger.debug(f"Sample #{i} has {len(sample)} paragraphs.")
        corpus: Iterable[str] = (getter(s) for s in sample)
        result = pointwise_average_distance(corpus, metric_class())
        result = np.array(result)
        # printable_result = f'{result.mean()} ± {2 * result.std() / math.sqrt(len(result))}'
        distances.append(result.mean())

    distances = np.array(distances)
    t_975 = t.ppf(1 - .025, df=len(samples) - 1)
    ci95 = t_975 * distances.std() / math.sqrt(len(distances))
    printable_result = f'{distances.mean():.4f} +/- {ci95:.4f}'
    click.echo(f"Pointwise average distance under the {click.style(metric, fg='green')} metric: "
               f"{click.style(printable_result, fg='green', bold=True)}")

    # reference
    corpus_reference: Iterable[str] = (getter(s) for s in reference)
    result_reference = np.array(pointwise_average_distance(corpus_reference, metric_class()))
    printable_result_reference = f'{result_reference.mean():.4f}'

    click.echo(f"Reference pointwise average distance under the {click.style(metric, fg='green')} metric: "
               f"{click.style(printable_result_reference, fg='green', bold=True)}")
    result_reference = result_reference.mean()

    within_ci = abs(distances.mean() - result_reference) < t_975 * distances.std() / math.sqrt(len(distances))
    result = {
        "num_samples": len(samples),
        metric: {
            'ours': {
                'human_readable': printable_result,
                'mean': distances.mean(),
                'variance': distances.var(),
                '95ci': ci95,
            },
            "reference": {
                'human_readable': printable_result_reference,
                'mean': result_reference,
            },
            "difference": {
                "difference": distances.mean() - result_reference,
                "within_ci": bool(within_ci)
            }
        },

    }
    write_json(result, output)


@click.command()
@click.option("--input", type=str, default="data/drop_dataset_dev.json")
@click.option("--output", type=str, default="data/drop_nfl.json")
def process_drop(input, output):
    with open(input) as f:
        data = json.load(f)
    drop = [v for k, v in data.items() if "nfl" in k]
    with open(output, "w+") as f:
        json.dump(drop, f)
