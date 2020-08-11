from copy import deepcopy
from itertools import chain, islice
import ujson as json
from typing import Dict, Any, List, Generator, Iterable, TypeVar, Union

from loguru import logger

from stresstest.classes import Entry, Bundle


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


def sample_iter(sample: Union[List[Dict[str, Any]], Dict]) -> Generator[Entry, None, None]:
    if isinstance(sample, dict):
        sample = sample['data']
    for datum in sample:
        datum_id = datum['title']
        for datum in datum['paragraphs']:
            passage = datum['context']
            for qa in datum['qas']:
                qa['passage_sents'] = datum.get('passage_sents', None)
                if not datum_id:
                    datum_id = qa['id']
                if qa['answers']:
                    yield Entry(datum_id, passage, qa['id'], qa['question'], qa.get('answer') or qa['answers'][0]['text'], qa)
                else:
                    logger.info(f"{qa['id']} has no answer!")

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


def only(sents_or_bundle, n, action='test'):
    if isinstance(n, int):
        n = [n]
    sents_or_bundle = deepcopy(sents_or_bundle)
    if isinstance(sents_or_bundle, Bundle):
        all_templates = sents_or_bundle.templates_modifier['sentences'][action]
        sents_or_bundle.templates_modifier['sentences'][action] = []
        for i, s in enumerate(all_templates):
            if i in n:
                sents_or_bundle.templates_modifier['sentences'][action].append(s)
    elif isinstance(sents_or_bundle, dict):
        all_templates = sents_or_bundle['sentences'][action]
        sents_or_bundle['sentences'][action] = []
        for i, s in enumerate(all_templates):
            if i in n:
                sents_or_bundle['sentences'][action].append(s)
    else:
        raise NotImplementedError
        # sents_or_bundle[action] = [sents_or_bundle[action][n]]
    return sents_or_bundle
