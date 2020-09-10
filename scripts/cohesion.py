import glob
import os
import random
import subprocess
import tempfile
from collections import defaultdict
from operator import itemgetter
import numpy as np
import click

from typing import Dict, List, Any, Callable

import tabulate
from loguru import logger
import json
import pandas as pd

from scripts.utils import write_json
from stresstest.eval_utils import get_mean_var_ci
from stresstest.util import load_json

# POS: w2v semantic similarity sentence(s)
POS = [
    'word2vec_1_all_sent',
    'word2vec_2_all_sent'
]

NEG = [
    # NEG: adjacent sentence verb lemma overlap
    'adjacent_overlap_verb_sent', 'adjacent_overlap_verb_sent_div_seg', 'adjacent_overlap_binary_verb_sent',
    'adjacent_overlap_2_verb_sent', 'adjacent_overlap_2_verb_sent_div_seg', 'adjacent_overlap_binary_2_verb_sent',
    # NEG: Lemma TTR
    'lemma_ttr', 'lemma_mattr',
    # NEG: Pronoun to noun ratio
    'pronoun_noun_ratio'
]

DEFAULT_INDICES = POS + NEG

MEASURE_TO_INDEX = {
    "w2v similarity": POS,
    'ttr': ['lemma_ttr', 'lemma_mattr'],
    'pronoun noun ratio': ['pronoun_noun_ratio'],
    'adjacent sentence verb lemma overlap': ['adjacent_overlap_verb_sent', 'adjacent_overlap_verb_sent_div_seg',
                                             'adjacent_overlap_binary_verb_sent', 'adjacent_overlap_2_verb_sent',
                                             'adjacent_overlap_2_verb_sent_div_seg',
                                             'adjacent_overlap_binary_2_verb_sent']
}

INDEX_TO_MEASURE = {v: k for k, vs in MEASURE_TO_INDEX.items() for v in vs}


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
@click.option('--random-seed', type=int, default=56)
@click.option('--subsample', type=int, default=200)
@click.option("--taaco-dir", type=str, default='lib/taaco/')
@click.option('--indices', type=str, default=None)
def quality(input, reference, output, attr, random_seed, subsample, taaco_dir, indices):
    if random_seed:
        random.seed(random_seed)
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
    if subsample:
        corpus = random.sample(corpus, subsample)

    result = apply_taaco(corpus, taaco_dir, indices)
    # for index, values in result.items():
    #    scores[index] = np.array(values, dtype=np.float)

    corpus_reference: List[str] = [getter(s) for s in reference]
    n_reference = len(corpus_reference)
    scores_reference = apply_taaco(corpus_reference, taaco_dir, indices)

    final_result = dict()
    overall = 0
    overall_ref = 0
    overall_pos = 0
    overall_pos_reference = 0
    overall_neg = 0
    overall_neg_reference = 0
    by_measure = defaultdict(list)
    by_measure_ref = defaultdict(list)
    for index, values in result.items():
        # t_975 = t.ppf(1 - .025, df=n - 1)
        # ci95 = t_975 * values.std() / math.sqrt(len(values))
        values = np.array(values)
        mean, var, ci95 = get_mean_var_ci(values, alpha=0.025)
        printable_result = f'{mean:.4f} +/- {ci95:.4f}'

        values_reference = np.array(scores_reference[index])
        mean_ref, var_ref, ci95_ref = get_mean_var_ci(values_reference, alpha=0.025)
        printable_result_reference = f'{mean_ref:.4f} +/- {ci95_ref:.4f}'

        by_measure[INDEX_TO_MEASURE[index]].append(mean)
        by_measure_ref[INDEX_TO_MEASURE[index]].append(mean_ref)

        if index in NEG:
            overall_neg = mean
            overall_neg_reference += mean_ref
            overall += (1 - mean)
            overall_ref += (1 - mean_ref)
        else:
            overall_pos += mean
            overall_pos_reference += mean_ref
            overall += mean
            overall_ref += mean_ref
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
    ours = ((overall_pos / len(POS)) + overall_neg / len(NEG)) / 2
    ref = ((overall_pos_reference / len(POS)) + overall_neg_reference / len(NEG)) / 2

    ours_smooth = overall / len(indices)
    ref_smooth = overall_ref / len(indices)

    by_measure_avg = [
        (sum(v) / len(v)) if k == 'w2v similarity' else 1 - (sum(v) / len(v)) for k, v in by_measure.items()
    ]
    by_measure_ref_avg = [
        (sum(v) / len(v)) if k == 'w2v similarity' else 1 - (sum(v) / len(v)) for k, v in by_measure_ref.items()
    ]
    rows = [[k, o, r] for k, o, r in zip(by_measure, by_measure_avg, by_measure_ref_avg)]
    click.echo(tabulate.tabulate(rows, headers=['Measure', "Ours", "Reference"]))
    by_measure_avg = sum(by_measure_avg) / len(by_measure_avg)

    by_measure_avg_ref = sum(by_measure_ref_avg) / len(by_measure_ref_avg)
    final_result['overall'] = {
        'ours': ours,
        "reference": ref,
        'ours_smooth': ours_smooth,
        'ref_smooth': ref_smooth,
        'by_measure_avg': by_measure_avg,
        'by_measure_avg_ref': by_measure_avg_ref
    }
    click.echo(f"Overall: {click.style(f'{ours:03f}', fg='green')} (n={n}): ")
    click.echo(f"Reference overall: {click.style(f'{ref:03f}', fg='green')} (n={n_reference}): ")
    click.echo(f"Overall smooth: {click.style(f'{ours_smooth:03f}', fg='green')} (n={n}): ")
    click.echo(f"Reference overall smooth: {click.style(f'{ref_smooth:03f}', fg='green')} (n={n_reference}): ")
    click.echo(f"Overall by measure: {click.style(f'{by_measure_avg:03f}', fg='green')} (n={n}): ")
    click.echo(
        f"Reference overall by measure: {click.style(f'{by_measure_avg_ref:03f}', fg='green')} (n={n_reference}): ")
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
