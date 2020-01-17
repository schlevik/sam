from typing import List

from stresstest.classes import Path
from stresstest.util import get_sentence_of_word, in_same_sentence


def bare_minimum(target: str, action: str, path: Path, candidates) -> List[int]:
    alphnum = list(path.alph_num())
    result = []
    for i in candidates:
        if alphnum[i] == target:
            if (action in alphnum) and \
                    action in alphnum[get_sentence_of_word(i, alphnum)]:
                result.append(i)

    return result


def is_not_modified(target: str, action: str, path: Path, candidates):
    alphnum = list(path.alph_num())
    result = []
    for i in candidates:
        sentence = alphnum[get_sentence_of_word(i, alphnum)]
        if "altering" not in sentence:
            result.append(i)

    return result
