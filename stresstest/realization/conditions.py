from abc import ABC, abstractmethod
from typing import Iterable

from ailog import Loggable

from stresstest.classes import Choices, Path
from stresstest.util import get_sentence_of_word


class BaseCondition(Loggable, ABC):
    def __init__(self, key: str):
        self.key = key

    def __call__(self, *,
                 possible_choices: Choices,
                 keys: Iterable[str],
                 path: Path = None,
                 position: int = None,
                 realised_path: Path = None):
        # TODO: check if == is enough or need to relax to re.match(f"{key}$")
        key = ".".join(keys)
        if key == self.key:
            return self.evaluate_condition(path=path,
                                           realised_path=realised_path,
                                           possible_choices=possible_choices,
                                           position=position)
        return possible_choices

    @abstractmethod
    def evaluate_condition(self, *,
                           possible_choices,
                           path=None,
                           realised_path=None,
                           position=None):
        ...


class SingularPluralCondition(BaseCondition):
    def __init__(self):
        super().__init__("path.VBZ-VBP")

    def evaluate_condition(self, *, possible_choices, path=None,
                           realised_path: Path = None,
                           position=None):
        if not (position and path):
            raise ValueError(
                f"{self.__class__.__name__} requires the position and "
                f"realised_path kwargs!")
        sentence = path[get_sentence_of_word(position, path)]
        if any(a.endswith('team') for a in sentence):
            possible_choices.remove('was')
        elif any(a.endswith('player') for a in sentence):
            possible_choices.remove('were')
        return possible_choices


class ModifierCondition(BaseCondition):

    def __init__(self):
        super().__init__("path.MODIFIER")

    def evaluate_condition(self, *, possible_choices: Choices, path=None,
                           realised_path=None, position=None):
        if not (position and path):
            raise ValueError(
                f"{self.__class__.__name__} requires the position and "
                f"realised_path kwargs!")
        sentence = path[get_sentence_of_word(position, path)]
        if "modifier.altering" not in sentence:
            print("removing altering")
            possible_choices.remove('$ALTERING')
            print(possible_choices)
        if "modifier.non-altering" not in sentence:
            print("removing non-altering")
            possible_choices.remove('$NON-ALTERING')
        print(possible_choices)
        return possible_choices
