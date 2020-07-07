import math
import random
import sys
from itertools import accumulate

import click

from scripts.utils import FormatParam, write_json
from stresstest.util import load_json


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
