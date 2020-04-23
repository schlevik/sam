import json
import random
import re
from abc import abstractmethod, ABC
from copy import deepcopy
from typing import Iterable, TypeVar, Mapping, Dict, Any, Optional, Union, List, Iterator, Tuple, Callable

from loguru import logger
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


class F:
    options: List[str] = None
    number: int = None

    @classmethod
    def make(cls, callable: Callable[[Dict], str], options_or_number=None):
        logger.debug(f"Making F with options_or_number={options_or_number}")
        instance = cls()
        if not options_or_number:
            options_or_number = 1
        if not isinstance(callable, Callable):
            raise ValueError(f"Called F.make with first argument of type {callable} but only "
                             f"Callable[(Context) -> str] is allowed!")
        instance.__call__ = callable
        if isinstance(options_or_number, int):
            instance.number = options_or_number
        elif isinstance(options_or_number, list):
            instance.options = options_or_number
        else:
            raise ValueError(f"Called F.make with second argument: {options_or_number} "
                             f"of type {type(options_or_number)}, but only int or List[str] is allowed!")

        return instance

    def __call__(self, ctx: dict) -> str:
        # TODO: maybe better dict -> S?
        raise NotImplementedError()


class S(List[str]):
    def __init__(self, iterable: List[str]):
        # this will tokenise things separated by whitespace but keep expressions in []() brackets together
        super().__init__([alt_opt_pattern.findall(template) for template in iterable])

    def random(self, exclude=None, mask=None) -> Tuple[str, int]:
        if exclude and mask:
            raise YouIdiotException("template.random with both 'exclude' and 'mask' keywords called!")
        exclude = exclude or []
        if exclude:
            mask = [1] * len(self)
            for i in exclude:
                mask[i] = 0
        choice = random.choices(range(len(self)), mask)[0]
        return deepcopy(self[choice]), choice


class YouIdiotException(Exception):
    ...
