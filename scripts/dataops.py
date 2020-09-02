import json
import math
import os
import random
import sys
from collections import defaultdict
from itertools import accumulate

import click

from scripts.utils import FormatParam, write_json, match_prediction_to_gold
from stresstest.ds_utils import filter_generic, export_brat_format
from stresstest.util import load_json, sample_iter
from stresstest.eval_utils import align as do_align


@click.command()
@click.option("--in-path", default='data/in.json')
@click.option("--out-path", default='data/out.json')
@click.option("--ds-format", type=FormatParam(), default=None)
@click.option("--split", type=str, default=False)
def export(in_path, out_path, ds_format, split):
    click.echo(f'Reading dataset from {click.style(in_path, fg="green")}')
    dataset = load_json(in_path)['data']

    if split:
        splits = []
        ratios = list(accumulate(map(float, split.split(":"))))
        if ratios and ratios[-1] != 100:
            click.secho(f"Split {split} doesn't add up to 100!", fg='red')
            sys.exit(1)
        for start_percent, end_percent in zip([0.0] + ratios, ratios):
            start_idx = math.ceil(start_percent / 100 * len(dataset))
            end_idx = math.ceil(end_percent / 100 * len(dataset))
            splits.append((int(end_percent - start_percent), dataset[start_idx:end_idx]))
        for i, (ratio, split) in enumerate(splits):
            if ds_format:
                split = ds_format(split)
            else:
                split = {'version': 0.1, 'data': split}
            out_name = f"{out_path.rsplit('.json', 1)[0]}-{i}-{str(ratio)}-split.json"
            click.echo(f'Writing out to: {click.style(out_name, fg="green")}')
            write_json(split, out_name)
    else:
        if ds_format:
            dataset = ds_format(dataset)
        else:
            dataset = {'version': 0.1, 'data': dataset}
        click.echo(f'Writing out to: {click.style(out_path, fg="green")}')
        write_json(dataset, out_path, pretty=False)


@click.command()
@click.argument('in-files', nargs=-1)
@click.argument('out-file', nargs=1)
def combine(in_files, out_file):
    combined_dataset = []
    if in_files:
        for in_file in in_files:
            click.echo(f"Reading in: {click.style(in_file, fg='green')}")
            ds = load_json(in_file)
            combined_dataset.extend(ds)
        click.echo(f"Writing out: {click.style(out_file, fg='green')}")
        write_json(combined_dataset, out_file, pretty=False)
    else:
        click.echo(f"Nothing to write...")


@click.command()
@click.option("--baseline", type=str)
@click.option("--intervention", type=str)
@click.option("--control", type=str)
@click.option('--out-folder', type=str)
@click.option('--subsample', type=int, default=0)
@click.option('--seed', type=int, default=56)
def align(baseline, intervention, control, out_folder, subsample, seed):
    random.seed(seed)
    aligned_baseline, aligned_intervention, aligned_control = do_align(load_json(baseline), load_json(intervention),
                                                                       load_json(control))
    assert len(aligned_baseline) == len(aligned_intervention) == len(aligned_control)
    if subsample and subsample < len(aligned_baseline):
        choices = random.sample(range(len(aligned_baseline)), subsample)
        aligned_baseline = [e for i, e in enumerate(aligned_baseline) if i in choices]
        aligned_intervention = [e for i, e in enumerate(aligned_intervention) if i in choices]
        aligned_control = [e for i, e in enumerate(aligned_control) if i in choices]
    gold_baseline = to_squad_fmt(aligned_baseline)
    gold_intervention = to_squad_fmt(aligned_intervention)
    gold_control = to_squad_fmt(aligned_control)
    baseline_out = os.path.join(out_folder, f"aligned-{subsample if subsample else ''}-{os.path.basename(baseline)}")
    write_json(gold_baseline, baseline_out, pretty=False)
    intervention_out = os.path.join(out_folder,
                                    f"aligned-{subsample if subsample else ''}-{os.path.basename(intervention)}")
    write_json(gold_intervention, intervention_out, pretty=False)
    control_out = os.path.join(out_folder, f"aligned-{subsample if subsample else ''}-{os.path.basename(control)}")
    write_json(gold_control, control_out, pretty=False)


@click.command()
@click.argument("gold-file", nargs=1)
@click.argument('prediction-file', nargs=1)
def convert_allennlp(gold_file, prediction_file):
    with open(prediction_file) as f:
        contents = f.read()
    results = {}
    for pred, entry in zip(contents.splitlines(), sample_iter(load_json(gold_file))):
        results[entry.qa_id] = json.loads(pred)['best_span_str']
        write_json(results, prediction_file)


@click.command()
@click.argument('in-files', nargs=-1)
@click.option("--out-folder", type=str, default='data/brat-data')
@click.option('--subsample', type=int, default=0)
@click.option('--seed', type=int, default=56)
@click.option('--coannotate', type=float, default=0.2)
def export_brat(in_files, out_folder, subsample, seed, coannotate):
    random.seed(seed)
    for in_file in in_files:
        d = load_json(in_file)
        total_paragraphs = sum(len(doc['paragraphs']) for doc in d['data'])
        fd = filter_generic(d, modifier_in_context)
        ds_name = os.path.basename(os.path.dirname(in_file))
        res = export_brat_format(fd, {"Modifier": ["almost", "nearly", "all but", "Almost", "Nearly", "All but"]})
        click.echo(f"{click.style(str(len(res)), fg='green')} occurrences (out of {total_paragraphs}) "
                   f"for dataset {click.style(ds_name, fg='blue')}.")
        if subsample:
            res = random.sample(res, subsample)
        out_path = os.path.join(out_folder, ds_name)
        os.makedirs(out_path, exist_ok=True)
        if coannotate:
            res_co = set(random.sample(range(len(res)), int(len(res) * coannotate)))
            out_path_co = os.path.join(out_folder, f"{ds_name}-coannotate")
            os.makedirs(out_path_co, exist_ok=True)
        else:
            res_co = set()
            out_path_co = None
        for i, (text, ann) in enumerate(res):
            with open(os.path.join(out_path, f"{i:03}.txt"), "w+") as f:
                f.write(text)
            with open(os.path.join(out_path, f"{i:03}.ann"), "w+") as f:
                f.write(ann)
            if i in res_co:
                with open(os.path.join(out_path_co, f"{i:03}.txt"), "w+") as f:
                    f.write(text)
                with open(os.path.join(out_path_co, f"{i:03}.ann"), "w+") as f:
                    f.write(ann)


def modifier_in_context(p):
    words = ['almost', 'nearly', 'all but']
    return any(w in p['context'].lower() for w in words)


def to_squad_fmt(flat_ds):
    unflattened = defaultdict(list)
    result = []
    for datum_id, passage, qa_id, question, answer, qa in flat_ds:
        unflattened[datum_id].append((passage, qa))
    for k, passages_and_qas in unflattened.items():
        passages, qas = zip(*passages_and_qas)
        entry = {
            'title': k,
            'paragraphs': [
                {'id': k, 'context': passages[0], 'passage_sents': qas[0]['passage_sents'], 'qas': list(qas)}],
        }
        result.append(entry)
    return {"version": 0.1, "data": result}
