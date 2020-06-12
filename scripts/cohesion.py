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
        cmd = f'python cli_folders.py --indir {data_path} --outdir {results_path} ' \
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

    samples = load_json(input)
    reference = load_json(reference)
    scores = defaultdict(list)
    getter: Callable[[Any], str] = itemgetter(attr)
    logger.debug(f"Evaluating {len(samples)} sample(s).")

    for i, sample in enumerate(tqdm(samples)):
        logger.debug(f"Sample #{i} has {len(sample)} paragraphs.")
        corpus: List[str] = [getter(s) for s in sample]
        result = apply_taaco(corpus, taaco_dir, indices)
        for index, values in result.items():
            scores[index].append(np.array(values, dtype=np.float).mean())

    corpus_reference: List[str] = [getter(s) for s in reference]
    scores_reference = apply_taaco(corpus_reference, taaco_dir, indices)

    final_result = {
        "num_samples": len(samples)
    }

    for index, values in scores.items():
        values = np.array(values)
        values_reference = np.array(scores_reference[index])
        t_975 = t.ppf(1 - .025, df=len(samples) - 1)
        ci95 = t_975 * values.std() / math.sqrt(len(values))

        printable_result = f'{values.mean():.4f} +/- {ci95:.4f}'
        printable_result_reference = f'{values_reference.mean():.4f}'

        click.echo(f"Average over {len(samples)} runs for {click.style(index, fg='green')} index: "
                   f"{click.style(printable_result, fg='green', bold=True)}")
        click.echo(f"Reference average for {click.style(index, fg='green')} metric: "
                   f"{click.style(printable_result_reference, fg='green', bold=True)}")
        final_result[index] = {
            'ours': {
                'human_readable': printable_result,
                'mean': values.mean(),
                'variance': values.var(),
                '95ci': ci95,
            },
            "reference": {
                'human_readable': printable_result_reference,
                'mean': values_reference.mean(),
            },
            "difference": {
                "difference": values.mean() - values_reference.mean(),
                # "within_ci": bool(within_ci)
            }
        }
    write_json(final_result, output)


@click.command()
@click.option("--input", type=str, default="data/drop_dataset_dev.json")
@click.option("--output", type=str, default="data/drop_nfl.json")
def process_drop(input, output):
    with open(input) as f:
        data = json.load(f)
    drop = [v for k, v in data.items() if "nfl" in k]
    with open(output, "w+") as f:
        json.dump(drop, f)
