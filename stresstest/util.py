from itertools import chain, islice
import click
import ujson as json
from typing import Dict, Any, List, Generator, Iterable, TypeVar

from stresstest.classes import Entry


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def num_questions(sample: List[Dict[str, Any]]):
    return sum(len(datum['qas']) for datum in sample)


T = TypeVar('T')


def batch(iterable: Iterable[T], batch_size=10) -> List[Iterable[T]]:
    iterator = iter(iterable)
    for first in iterator:
        yield list(chain([first], islice(iterator, batch_size - 1)))


def sample_iter(sample: List[Dict[str, Any]]) -> Generator[Entry, None, None]:
    for datum in sample:
        passage = datum['passage']
        for qa in datum['qas']:
            yield Entry(datum['id'], passage, qa['id'], qa['question'], qa['answer'], qa)


def fmt_dict(dct: dict):
    return "{{{}}}".format(', '.join(f"'{k}': {v}" for k, v in dct.items() if k != "self" and not k.startswith("_")))


def do_import(class_string: str, relative_import: str = ""):
    try:
        module_name, cls_name = class_string.rsplit(".", 1)
    except ValueError:
        cls_name = class_string
        module_name = ""

    if relative_import:
        x = ".".join((relative_import, module_name)).rstrip(".")
        try:
            mod = __import__(x, fromlist=cls_name)
        except ModuleNotFoundError:
            mod = __import__(module_name, fromlist=cls_name)
    else:
        mod = __import__(module_name, fromlist=cls_name)
    cls = getattr(mod, cls_name)

    return cls


def highlight(text, colors):
    result = []
    for t in text.split():
        if t in colors.keys():
            result.append(click.style(t, fg=colors[t]))
        else:
            result.append(t)
    return " ".join(result)
