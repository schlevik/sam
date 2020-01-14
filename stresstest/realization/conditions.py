from abc import ABC, abstractmethod

from stresstest.classes import Choices, Path


class BaseCondition(ABC):
    def __init__(self, key: str):
        self.key = key

    def __call__(self, path: Path, possible_choices: Choices, *keys: str):
        # TODO: check if == is enough or need to relax to re.match(f"{key}$")
        key = ".".join(keys)
        if key == self.key:
            return self.evaluate_condition(path, possible_choices)
        return possible_choices

    @abstractmethod
    def evaluate_condition(self, path, possible_choices):
        ...
