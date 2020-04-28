import ujson as json
from typing import Dict, Any, List


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def sample_iter(sample: List[Dict[str, Any]]):
    for datum in sample:
        passage = datum['passage']
        for qa in datum['qas']:
            yield datum['id'], passage, qa['id'], qa['question'], qa['answer']
