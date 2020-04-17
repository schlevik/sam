import json
import random
import re
import string
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Sequence, Iterable, TypeVar, Mapping, Dict, Any, Optional, Union, List, Iterator, Tuple
from quickconf import ConfigReader

T = TypeVar("T")
V = TypeVar("V")
JsonDict = Dict[str, Any]


class Choices(Iterable[T]):
    """
    Thin wrapper over a sequence.

    Allows to safely remove elements and chose random elements,
    possibly subject to given :class:`rule` s.
    """

    def __iter__(self) -> Iterator[T]:
        return iter(self.choices)

    def __init__(self, l: Iterable[T]):
        self.choices = set(l)

    def __hash__(self):
        return hash(self.choices)

    def __eq__(self, other):
        return isinstance(other, Choices) and self.choices == other.choices

    def __len__(self):
        return len(self.choices)

    def __sub__(self, other):
        # if not isinstance(other, Choices) or not isinstance(other, list):
        c = Choices(self.choices)
        c.remove(other)
        return c

    def remove(self, i: Union[Iterable[T], T]) -> None:
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
            return random.choice(tuple(self))
        except IndexError:
            return None

    def remove_all_but(self, *nodes: T) -> None:
        self.remove([n for n in self if n not in nodes])

    def all_but(self, *nodes: T) -> 'Choices':
        c = Choices(self.choices)
        c.remove_all_but(*nodes)
        return c


class Config(Mapping):
    """
    Thin wrapper around the template config tree.

    Allows to represent chosen branches and leaves as
    :class:`Choices` and select random ones with :class:`Rule` s.
    """

    def __init__(self, templates_path: str):
        self.cfg = ConfigReader(templates_path).read_config()

    def __getitem__(self, k):
        return self.cfg[k]

    def as_choices(self, *keys: str) -> Choices:
        """
        Generates choices from a template for a sequence of given keys.

        Keys are joined with the point operator. i.e. if you want to
        access "a.b.c" you can call::

            as_choices(keys=['a','b','c'])

        Args:
            *keys: Keys to choose the template from. Joins the keys with
            the point (".") operator.

        Returns:
            ``Choices``

        """
        return Choices(self.cfg['.'.join(keys)])

    def random(self, *keys):
        return self.as_choices(*keys).random()

    def pprint(self):
        return json.dumps(self.cfg, indent=4)

    def __len__(self) -> int:
        """
        Wraps the ConfigTree function.

        Returns:
            Number of top level keys in the config.

        """
        return len(self.cfg)

    def __iter__(self):
        """
        Wraps the ConfigTree function.

        Returns:
            Iterator over the top level keys of the config.

        """
        return iter(self.cfg)


alt_opt_pattern = re.compile(r"([^(\[\]]\S*|\(.+?\)|\[.+?\])\s*")


class S(List[str]):
    def __init__(self, iterable: List[str]):
        # this will tokenise things separated by whitespace but keep expressions in []() brackets together
        super().__init__([alt_opt_pattern.findall(template) for template in iterable])

    def random(self) -> Tuple[str, int]:
        choice = random.randint(0, len(self) - 1)
        return deepcopy(self[choice]), choice


class YouIdiotException(Exception):
    ...
