import random
from abc import ABC, abstractmethod
from typing import Sequence, List, Iterable, TypeVar, Mapping, Dict, Any

from aiconf import ConfigReader
from ailog import Loggable

from stresstest.util import alphnum


class Stringifier(Loggable, ABC):
    @abstractmethod
    def to_string_path(self):
        ...

    @abstractmethod
    def to_string_question(self):
        ...

    @abstractmethod
    def to_string_answer(self):
        ...


T = TypeVar("T")

JsonDict = Dict[str, Any]


class Path(Loggable, Sequence):
    def __init__(self):
        self.steps = []

    def push(self, node):
        self.logger.debug(f"{self} ++ {node}")
        self.steps.append(node)

    def __str__(self):
        return str(self.steps)

    def __repr__(self):
        return repr(self.steps)

    def rindex(self, i):
        return len(self.steps) - self.steps[::-1].index(i) - 1

    def from_index(self, i):
        p = Path()
        p.steps = self.steps[i:]
        return p

    def __getitem__(self, i):
        return self.steps[i]

    def __len__(self):
        return len(self.steps)

    def alph_num(self):
        return (alphnum(e) for e in self)

    @property
    def last(self):
        return self.steps[-1]

    def occurrences(self, node: str):
        return sum(x == node for x in self)


class Choices(Sequence[T]):
    def __init__(self, l: Iterable[T]):
        self.choices = list(l)

    def __getitem__(self, i):
        return self.choices[i]

    def __len__(self):
        return len(self.choices)

    def remove(self, i):
        if isinstance(i, Iterable) and not isinstance(i, str):
            for e in i:
                self.remove(e)
        elif i in self.choices:
            self.choices.remove(i)

    def __repr__(self):
        return repr(self.choices)

    def random(self) -> T:
        return random.choice(self)

    def random_with_conditions(self, path, conditions, *args, **kwargs):
        choices = self.choices
        for condition in conditions:
            choices = condition(path, choices, *args, **kwargs)
        return random.choice(choices)


class Templates(Loggable, Mapping):
    def __getitem__(self, k):
        return self.cfg[k]

    def templates(self, *keys: str):
        return self.cfg['.'.join(keys)]

    def as_choices(self, *keys: str):
        return Choices(self.templates(*keys))

    def random_with_conditions(self, path, keys, *args, **kwargs):
        self.logger.info("BLABLA")
        print("")
        self.logger.info(self.templates(*keys))
        return self \
            .as_choices(*keys) \
            .random_with_conditions(path, self.conditions, *args, **kwargs)

    def __len__(self) -> int:
        return len(self.cfg)

    def __iter__(self):
        return iter(self.cfg)

    def __init__(self, templates_path: str, conditions):
        self.cfg: JsonDict = ConfigReader(templates_path).read_config()
        self.conditions = conditions
