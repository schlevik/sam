import os
import random
import uuid

import click

from scripts.utils import Domain, BASELINE, INTERVENTION, write_json, CONTROL
from stresstest.classes import Config
from stresstest.comb_utils import split
from stresstest.ds_utils import to_squad
from stresstest.generate_utils import generate_and_realise
from stresstest.print_utils import visualize
from stresstest.util import do_import


@click.command()
@click.option("--config")
@click.option("--out-path", default='data/modifier/')
@click.option("--seed", type=int, default=False)
@click.option("--do-print", type=int, default=None)
@click.option("--do-save", is_flag=True, default=False)
@click.option("--domain", type=Domain(), default='football')
@click.option("--num-workers", type=int, default=8)
@click.option("--split-templates", type=float, default=False)
@click.option("--modifier-type", type=str, default='RB')
def generate_balanced(config, out_path, seed, do_print, do_save, domain, num_workers, split_templates,
                      modifier_type):
    if seed:
        random.seed(seed)
    uuid4 = lambda: uuid.UUID(int=random.getrandbits(128)).hex

    cfg = Config(config)

    max_sents = cfg["world.num_sentences"]

    modify_event_type = cfg['modify_event_type']
    if split_templates:
        first, second = split(domain.templates_modifier, event_types_to_split=[modify_event_type],
                              split_ratio=split_templates)
        template_splits = [first, second]

        click.echo(f"Splitting templates with a {split_templates} ratio.")

        for event_type, templates in domain.templates_modifier['sentences'].items():
            click.echo(f"For event type '{event_type}'")
            click.echo(f"First split: {[templates.index(t) for t in first['sentences'][event_type]]}")
            click.echo(f"Second split: {[templates.index(t) for t in second['sentences'][event_type]]}")
    else:
        template_splits = [domain.templates_modifier]

    for i, templates in enumerate(template_splits):
        num_modifiers = cfg['num_modifiers']
        source_dataset_name = cfg['for_dataset']
        reasoning_map = {do_import(k, relative_import="stresstest.reasoning"): v for k, v in
                         cfg['reasoning_map'].items()}
        total = sum(t * num_modifiers for t in reasoning_map.values())
        split_str = f"-{i}" if split_templates else ""
        file_name = f"{{}}-{source_dataset_name}-{modifier_type.lower()}{split_str}.json"
        click.echo(
            f"Generating from '{click.style(config, fg='green')}': {click.style(str(total), fg='green', bold=True)} "
            f"passages, and questions.")
        click.echo(f"Saving baseline in "
                   f"{click.style(os.path.join(out_path, file_name.format(BASELINE)), fg='blue', bold=True)}.")
        click.echo(f"Saving modified in "
                   f"{click.style(os.path.join(out_path, file_name.format(INTERVENTION)), fg='blue', bold=True)}.")
        click.echo(f"Saving control in "
                   f"{click.style(os.path.join(out_path, file_name.format(CONTROL)), fg='blue', bold=True)}.")

        res = generate_and_realise(
            modify_event_type=modify_event_type,
            config=cfg,
            bundle=domain,
            reasonings=reasoning_map,
            modifier_type=modifier_type,
            max_modifiers=num_modifiers,
            use_mod_distance=False,
            mute=False,
            num_workers=num_workers
        )
        (event_plans, events, template_choices, worlds, baseline_stories, mqs, qs, modified_stories,
         control_stories) = zip(*res)
        baseline, modified, control = to_squad(
            uuid4, event_plans, events, template_choices, baseline_stories, mqs, qs, modified_stories, control_stories
        )
        if do_print:
            lines = visualize(
                event_plans, events, template_choices, worlds, baseline_stories, mqs, qs, modified_stories,
                control_stories, n=do_print
            )
            for line in lines:
                click.echo(line)
        if do_save:
            write_json({"version": 0.1, "data": baseline}, os.path.join(out_path, file_name.format(BASELINE)),
                       pretty=False)
            write_json({"version": 0.1, "data": modified}, os.path.join(out_path, file_name.format(INTERVENTION)),
                       pretty=False)
            write_json({"version": 0.1, "data": control}, os.path.join(out_path, file_name.format(CONTROL)),
                       pretty=False)