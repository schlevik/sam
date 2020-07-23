import glob
import math
import os
import subprocess
import tempfile
from collections import defaultdict
from operator import itemgetter
import numpy as np
import click

from typing import Dict, List, Any, Callable
from loguru import logger
import json
import pandas as pd
from scipy.stats import t
from tqdm import tqdm

from scripts.utils import write_json
from stresstest.eval_utils import get_mean_var_ci
from stresstest.util import load_json

DEFAULT_INDICES = [
    # NEG: adjacent sentence verb lemma overlap
    'adjacent_overlap_verb_sent', 'adjacent_overlap_verb_sent_div_seg', 'adjacent_overlap_binary_verb_sent',
    'adjacent_overlap_2_verb_sent', 'adjacent_overlap_2_verb_sent_div_seg', 'adjacent_overlap_binary_2_verb_sent',
    # POS: w2v semantic similarity sentence(s)
    'word2vec_1_all_sent', 'word2vec_2_all_sent',
    # NEG: Lemma TTR
    'lemma_ttr', 'lemma_mattr',
    # NEG: Pronoun to noun ratio
    'pronoun_noun_ratio'
]


def apply_taaco(corpus: List[str], taaco_dir, indices) -> Dict[str, List[float]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        # convert & write corpus into docs
        data_path = f"{temp_dir}/data"
        results_path = f"{temp_dir}/results.csv"
        os.mkdir(os.path.join(temp_dir, 'data'))
        logger.debug(f"{len(corpus)} documents to process with taaco.")
        for i, doc in enumerate(corpus):
            with open(os.path.join(data_path, f'{i:03}.txt'), "w+") as f:
                f.write(doc)
        logger.debug(f"{len(glob.glob(os.path.join(data_path, '*')))} files created.")
        # run taaco
        cmd = f'python cli_folders.py --indir {data_path} --outfile {results_path} ' \
              f'--working-dir {temp_dir} --config taaco.ini'.split()
        subprocess.call(cmd, shell=False, cwd=taaco_dir, stdout=subprocess.DEVNULL)
        df = pd.read_csv(results_path, sep=',', header=0)
        return {x: list(df[x]) for x in df.columns[1:] if x in indices}


@click.command()
@click.option("--input", type=str, default="data/stresstest.json")
@click.option("--reference", type=str, default="data/drop_nfl.json")
@click.option("--output", type=str, default="metrics/quality.json")
@click.option("--attr", type=str, default='passage')
@click.option("--taaco-dir", type=str, default='lib/taaco/')
@click.option('--indices', type=str, default=None)
def quality(input, reference, output, attr, taaco_dir, indices):
    if not indices:
        indices = DEFAULT_INDICES
    else:
        indices = indices.split(",")

    sample = load_json(input)
    reference = load_json(reference)
    # scores = Dict[str, np.ndarray]
    getter: Callable[[Any], str] = itemgetter(attr)

    corpus: List[str] = [s['paragraphs'][0]['context'] for s in sample['data']]
    n = len(corpus)
    logger.debug(f"Evaluating sample with n={n} paragraphs.")

    result = apply_taaco(corpus, taaco_dir, indices)
    # for index, values in result.items():
    #    scores[index] = np.array(values, dtype=np.float)

    corpus_reference: List[str] = [getter(s) for s in reference]
    n_reference = len(corpus_reference)
    scores_reference = apply_taaco(corpus_reference, taaco_dir, indices)

    final_result = dict()

    for index, values in result.items():
        # t_975 = t.ppf(1 - .025, df=n - 1)
        # ci95 = t_975 * values.std() / math.sqrt(len(values))
        values = np.array(values)
        mean, var, ci95 = get_mean_var_ci(values, alpha=0.025)
        printable_result = f'{mean:.4f} +/- {ci95:.4f}'

        values_reference = np.array(scores_reference[index])
        mean_ref, var_ref, ci95_ref = get_mean_var_ci(values_reference, alpha=0.025)
        printable_result_reference = f'{mean_ref:.4f} +/- {ci95_ref:.4f}'

        # t_975_reference = t.ppf(1 - .025, df=n_reference - 1)
        # ci95_reference = t_975_reference * values_reference.std()
        click.echo(f"Mean for index {click.style(index, fg='green')} (n={n}): "
                   f"{click.style(printable_result, fg='green', bold=True)}")
        click.echo(f"Reference mean for index {click.style(index, fg='green')} (n={n_reference}): "
                   f"{click.style(printable_result_reference, fg='green', bold=True)}")
        final_result[index] = {
            'ours': {
                'n': len(sample),
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
        }
    write_json(final_result, output)


@click.command()
@click.option("--input", type=str, default="data/football/drop_dataset_dev.json")
@click.option("--output", type=str, default="data/football/drop_nfl.json")
def process_drop(input, output):
    with open(input) as f:
        data = json.load(f)
    drop = [v for k, v in data.items() if "nfl" in k]
    with open(output, "w+") as f:
        json.dump(drop, f)
