from typing import List

import click

from stresstest.classes import Question


def fmt_dict(dct: dict):
    return "{{{}}}".format(', '.join(f"'{k}': {v}" for k, v in dct.items() if k != "self" and not k.startswith("_")))


def highlight(text, colors):
    result = []
    for t in text.split():
        if t in colors.keys():
            result.append(click.style(t, fg=colors[t]))
        else:
            result.append(t)
    return " ".join(result)


def highlight_passage_and_question(passage: List[str], question: Question, highlight_all_numbers=True):
    res = []
    color_map = {}
    if highlight_all_numbers:
        color_map = {str(d): 'green' for d in range(0, 100)}
    color_map.update({t: 'red' for t in question.answer.split()})
    for i, p in enumerate(passage):
        if i in question.evidence:
            i = click.style(str(i), fg='blue', bold=True)
        res.append(f"[{i}] {highlight(p, color_map)}")
    res.append(click.style(question.realized, fg='magenta'))
    res.append(question.answer)
    return "\n".join(res)
