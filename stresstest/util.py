from itertools import chain, islice
import ujson as json
from typing import Dict, Any, List, Generator, Iterable, TypeVar

from stresstest.classes import Entry


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def num_questions(sample: List[Dict[str, Any]]):
    if isinstance(sample, dict):
        sample = sample['data']
    return sum(len(datum['paragraphs'][0]['qas']) for datum in sample)


T = TypeVar('T')


def batch(iterable: Iterable[T], batch_size=10) -> List[Iterable[T]]:
    iterator = iter(iterable)
    for first in iterator:
        yield list(chain([first], islice(iterator, batch_size - 1)))


def sample_iter(sample: List[Dict[str, Any]]) -> Generator[Entry, None, None]:
    if isinstance(sample, dict):
        sample = sample['data']
    for datum in sample:
        datum_id = datum['title']
        datum = datum['paragraphs'][0]
        passage = datum['context']
        for qa in datum['qas']:
            qa['passage_sents'] = datum['passage_sents']
            yield Entry(datum_id, passage, qa['id'], qa['question'], qa.get('answer') or qa['answers'][0]['text'], qa)


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


