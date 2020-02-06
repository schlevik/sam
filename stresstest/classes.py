import json
import random
import string
from abc import ABC, abstractmethod
from typing import Sequence, Iterable, TypeVar, Mapping, Dict, Any, Optional, \
    Union, List, Iterator

from quickconf import ConfigReader
from quicklog import Loggable

T = TypeVar("T")

JsonDict = Dict[str, Any]


class Path(Loggable, Sequence[str]):
    """
    A thin wrapper around a list representing the path of nodes chosen
    on the content graph.
    """

    def __add__(self, other):
        if isinstance(other, Path):
            return Path(self.steps + other.steps)
        elif isinstance(other, list):
            return Path(self.steps + other)
        else:
            raise ValueError("Can only concatenate paths or lists!")

    def __init__(self, iterable: Iterable[str] = None):
        """
        Creates the instance from scratch or given an iterable of steps.

        If steps are given, copies them over one by one rather than
        referencing the given iterable.
        Args:
            iterable:
                Optional iterable of steps to initialise the path with.
        """
        self.steps = [e for e in iterable] if iterable else []

    def sentences(self) -> List['Sentence']:
        sentences = []
        first_sos = self.index('sos')
        # sentences.append(Sentence(self[:first_sos]))
        for i, item in enumerate(self.steps[first_sos:]):
            if item == 'eos' or i + 1 == len(self.steps):
                sentences.append(
                    Sentence(self.steps[sum(map(len, sentences)):i + 2]))

        return sentences

    def push(self, node: str):
        """
        Adds a node at the end of the existing path.

        Args:
            node: Node to add.
        """
        self.logger.debug(f"{self} ++ {node}")
        self.steps.append(node)

    def __str__(self):
        return str(self.steps)

    def __repr__(self):
        return repr(self.steps)

    def rindex(self, element):
        """
        The left index of an element for the first occurrence from the
        right.

        Args:
            element: Element to get the index for.

        Returns:
            Index of the first occurrence of the given element.

        Raises:
            :class:`ValueError`: if ``element`` is not in the path.

        """
        return len(self.steps) - self.steps[::-1].index(element) - 1

    def __eq__(self, other):
        return isinstance(other, Path) and self.steps == other.steps

    def from_index(self, i):
        """
        Creates a copy of the path from a given index to its end::

             p = Path([1,2,3])
             q = p.from_index(1)

             p[1:] == q # True

        Args:
            i: Index to copy the path from.

        Returns:
            A copy of the path from a given index.

        """
        if i >= len(self):
            raise ValueError(f"Index i ({i}) to create Path from must be "
                             f"smaller than this path's length ({len(self)})")
        return Path(self.steps[i:])

    def __getitem__(self, i):
        """
        Returns the item at i.

        If `i` is a slice, creates a `copy` of the path rather than
        giving direct access to the path slice.
        Args:
            i: index or slice to get the item(s).

        Returns:
             The element at i or a copy of the path at i if i is a
             slice.

        """
        if isinstance(i, slice):
            p = Path(self.steps[i])
            return p
        return self.steps[i]

    def __len__(self):
        return len(self.steps)

    def alph_num(self):
        """
        Returns a copy of the path with only alphanumeric node names.

        Returns: Copy of the path without any punctuation characters.

        """
        return Path("".join(c for c in e if c not in string.punctuation)
                    for e in self)

    @property
    def last(self):
        """
        Returns the current last element in the path.

        Returns:
            current last element in the path.

        Raises:
            :class:`IndexError`: if path is empty.

        """
        try:
            return self.steps[-1]
        except IndexError:
            raise IndexError("Cannot access the last member of an "
                             "empty path!")

    def occurrences(self, node: str):
        """
        Counts the occurrences of a node in the path.

        Args:
            node: Node to count

        Returns:
            Number of occurrences of a node.
        """
        return sum(x == node for x in self)


class Sentence(Path):
    def __init__(self, iterable: Iterable[str] = None):
        super().__init__(iterable)
        self.modalities: JsonDict = dict()


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

    def random_with_rules(self, *, rules: List['Rule'],
                          **kwargs) -> Optional[T]:
        """
        Returns a random choice that conforms to all the given
        rules.

        Does not change the instance.

        Args:
            rules: List of rules to apply.
            **kwargs: Kwargs that should match the rules

        Returns:
            Random choice conforming to given rules.

        """
        choices = Choices(self)
        for rule in rules:
            choices = rule(choices=choices,
                           **kwargs)
        return choices.random()

    def remove_all_but(self, *nodes: T) -> None:
        self.remove([n for n in self if n not in nodes])

    def all_but(self, *nodes: T) -> 'Choices':
        c = Choices(self.choices)
        c.remove_all_but(*nodes)
        return c


class Rule(Loggable, ABC):
    """
    Base rule class.

    All rules should implement this.
    """

    @abstractmethod
    def __call__(self, *, path: Path, choices: Choices, **kwargs) -> Choices:
        """
        rules should implement this method.

        Args:
            path: The path this rule is applied upon.
            choices:
                :class:`Choices` prior to the application of the
                implementing rule.
            **kwargs:

        Returns:
            Valid :class:`Choices` after the application of the
                implementing rule.
        """
        ...


class Config(Loggable, Mapping):
    """
    Thin wrapper around the template config tree.

    Allows to represent chosen branches and leaves as
    :class:`Choices` and select random ones with :class:`Rule` s.
    """

    def __init__(self, templates_path: str, rules: List[Rule]):
        self.cfg: JsonDict = ConfigReader(templates_path).read_config()
        self.rules = rules

    def __getitem__(self, k):
        return self.cfg[k]

    def templates(self, *keys: str) -> JsonDict:
        """
        Gives the templates for a sequence of given keys.

        Keys are joined with the point operator. i.e. if you want to
        access "a.b.c" you can call::

            random_with_rules(keys=['a','b','c'])

        Args:
            *keys: Keys to choose the template from. Joins the keys with
            the point (".") operator.

        Returns:

        """
        return self.cfg['.'.join(keys)]

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
        return Choices(self.templates(*keys))

    def random(self, *keys):
        return self.as_choices(*keys).random()

    def random_with_rules(self, keys: List[str], **kwargs):
        """
        Chooses a random template from the given keys satisfying the
        rules.

        Keys are joined with the point operator. i.e. if you want to
        access "a.b.c" you can call::

            random_with_rules(keys=['a','b','c'])

        Args:
            keys: Keys to chose the template from. Joins the keys with
            the point (".") operator.
            **kwargs: Keyword arguments passed to the rules.
            Depend on the type of rule you're using.

        Returns:
            A randomly chosen template that satisfies all
                rules.

        """

        # add keys to kwargs in case we want to pass on the keys
        kwargs['keys'] = keys
        return self \
            .as_choices(*keys) \
            .random_with_rules(rules=self.rules, **kwargs)

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
