import random
from abc import ABC, abstractmethod
from typing import Sequence, Iterable, TypeVar, Mapping, Dict, Any, Optional, \
    Union

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
        if isinstance(i, slice):
            p = Path()
            p.steps = self.steps[i]
            return p
        return self.steps[i]

    def __len__(self):
        return len(self.steps)

    def alph_num(self):
        p = Path()
        p.steps = [alphnum(e) for e in self]
        return p

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

    def remove(self, i: Union[Iterable, str]) -> None:
        """
        Removes a choice if present.
        Does nothing otherwise.

        Args:
            i: Element to remove. If argument is an Iterable of
               elements removes  all elements.
        """
        if isinstance(i, Iterable) and not isinstance(i, str):
            for e in i:
                self.remove(e)
        elif i in self.choices:
            self.choices.remove(i)

    def __repr__(self):
        return f"Choices({repr(self.choices)})"

    def random(self) -> Optional[T]:
        """
        Returns a random value if present, returns None otherwise.


        Returns:
            Random value if present None otherwise.
        """
        try:
            return random.choice(self)
        except IndexError:
            return None

    def random_with_conditions(self, *, conditions, **kwargs):
        """

        Args:
            conditions:
            **kwargs: Kwargs that should match the conditions

        Returns:

        """
        choices = self
        for condition in conditions:
            choices = condition(possible_choices=choices,
                                **kwargs)
        return choices.random()


class Templates(Loggable, Mapping):
    def __getitem__(self, k):
        return self.cfg[k]

    def templates(self, *keys: str):
        """
        Gives the templates for a sequence of given keys.

        Keys are joined with the point operator. i.e. if you want to
        access "a.b.c" you can call::

            random_with_conditions(keys=['a','b','c'])
        Args:
            *keys: Keys to choose the template from. Joins the keys with
            the point (".") operator.

        Returns:

        """
        return self.cfg['.'.join(keys)]

    def as_choices(self, *keys: str):
        """
        Generates choices from a template for a sequence of given keys.

        Keys are joined with the point operator. i.e. if you want to
        access "a.b.c" you can call::

            random_with_conditions(keys=['a','b','c'])

        Args:
            *keys: Keys to choose the template from. Joins the keys with
            the point (".") operator.

        Returns: ``Choices``

        """
        return Choices(self.templates(*keys))

    def random_with_conditions(self, keys, **kwargs):
        """
        Chooses a random template from the given keys satisfying the
        conditions.

        Keys are joined with the point operator. i.e. if you want to
        access "a.b.c" you can call::

            random_with_conditions(keys=['a','b','c'])

        Args:
            keys: Keys to chose the template from. Joins the keys with
            the point (".") operator.
            **kwargs: Keyword arguments passed to the conditions.
            Depend on the type of condition you're using.

        Returns: A randomly chosen template that satisfies all
        conditions.

        """
        # add keys to kwargs in key we want to pass on the keys
        kwargs['keys'] = keys
        return self \
            .as_choices(*keys) \
            .random_with_conditions(conditions=self.conditions, **kwargs)

    def __len__(self) -> int:
        return len(self.cfg)

    def __iter__(self):
        return iter(self.cfg)

    def __init__(self, templates_path: str, conditions):
        self.cfg: JsonDict = ConfigReader(templates_path).read_config()
        self.conditions = conditions
