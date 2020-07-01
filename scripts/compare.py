import math
from operator import itemgetter
import numpy as np
import click
import quickconf
from typing import Iterable, Callable, Any, List
from loguru import logger
import json

from scripts.utils import write_json, MetricParam
from stresstest.eval_utils import get_mean_var_ci
from stresstest.textmetrics import pointwise_average_distance
from stresstest.util import load_json


@click.command()
@click.option("--input", type=str, default="data/stresstest.json")
@click.option("--reference", type=str, default="data/drop_nfl.json")
@click.option("--output", type=str, default=None)
@click.option("--attr", type=str, default='passage')
@click.option("--metric", type=MetricParam(), default='Jaccard')
def diversity(input, reference, output, attr, metric: type):
    # TODO: make metrics appendable
    sample = load_json(input)
    reference = load_json(reference)
    getter: Callable[[Any], str] = itemgetter(attr)

    # samples
    corpus: List[str] = [getter(s) for s in sample]
    n = len(corpus)
    logger.debug(f"Evaluating sample with n={n} paragraphs.")
    result = pointwise_average_distance(corpus, metric())
    result = np.array(result)
    mean, var, ci95 = get_mean_var_ci(result, alpha=0.025)
    printable_result = f'{mean:.4f} +/- {ci95:.4f}'
    click.echo(f"Point-wise average distance under the {click.style(str(metric.__name__), fg='green')} metric (n={n}): "
               f"{click.style(printable_result, fg='green', bold=True)}")

    corpus_reference: List[str] = [getter(s) for s in reference]
    result_reference = np.array(pointwise_average_distance(corpus_reference, metric()))
    mean_ref, var_ref, ci95_ref = get_mean_var_ci(result_reference, alpha=0.025)
    printable_result_reference = f'{mean_ref:.4f} +/- {ci95_ref:.4f}'
    click.echo(
        f"Reference point-wise average distance under the {click.style(str(metric.__name__), fg='green')} "
        f"metric(n={len(corpus_reference)}): "
        f"{click.style(printable_result_reference, fg='green', bold=True)}")

    result = {
        str(metric.__name__): {
            'ours': {
                "n": n,
                'human_readable': printable_result,
                'mean': mean,
                'variance': var,
                '95ci': ci95,
            },
            "reference": {
                'human_readable': printable_result_reference,
                'mean': mean_ref,
                'variance': var_ref,
                '95ci': ci95_ref
            },
            "difference": {
                "difference": mean - mean_ref,
                # "within_ci": bool(within_ci)
            }
        },

    }
    if output:
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
