import ujson as json
from typing import Dict, Any, List


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def num_questions(sample: List[Dict[str, Any]]):
    return sum(len(datum['qas']) for datum in sample)


def sample_iter(sample: List[Dict[str, Any]], full=False):
    for datum in sample:
        passage = datum['passage']
        for qa in datum['qas']:
            if full:
                yield datum['id'], passage, qa['id'], qa['question'], qa['answer'], qa
            else:
                yield datum['id'], passage, qa['id'], qa['question'], qa['answer']


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
