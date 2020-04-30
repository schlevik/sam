import json
import click

from stresstest.resources.templates import sentences


def write_json(result, output):
    with open(output, "w+") as f:
        json.dump(result, f, indent=4, separators=(',', ': '))


def get_templates(action: str = None, n: int = None, command: str = "Executing command"):
    actions = list(sentences.keys())
    if action is not None:
        actions = [action]

    if n is not None:
        for action in actions:
            sentences[action] = [sentences[action][n]]

    actions_str = click.style(', '.join(actions), fg='blue')
    n_str = click.style(text=str(n) if n else "all", fg='green', bold=True)
    click.echo(f"{click.style(command, fg='red')} for actions: '{actions_str}'; sentences: {n_str} !")
    return actions, sentences
