import inspect
import json
import random
import re
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable, TypeVar, Mapping, Dict, Any, Optional, Union, List, Iterator, Tuple, Callable

from loguru import logger
from pyhocon import ConfigFactory

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
        self.choices = list(l)

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
        if isinstance(i, Iterable) and not isinstance(i, str) and not isinstance(i, Mapping):
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

    def keep_only(self, *nodes: T) -> 'Choices':
        c = Choices(self.choices)
        c.remove_all_but(*nodes)
        return c


class Config(Mapping):
    """
    Thin wrapper around the template config tree.

    Allows to represent chosen branches and leaves as
    :class:`Choices` and select random ones with :class:`Rule` s.
    """

    def __init__(self, templates_path: Union[str, dict]):
        if isinstance(templates_path, str):
            self.cfg = ConfigFactory().parse_file(templates_path)
        else:
            self.cfg = ConfigFactory().from_dict(templates_path)

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
    do_call = None

    @classmethod
    def make(cls, callable: Callable[[Dict], str], options_or_number=None):
        logger.debug(f"Making F with options_or_number={options_or_number}")
        instance = cls()
        if not options_or_number:
            options_or_number = 1
        if not isinstance(callable, Callable):
            raise ValueError(f"Called F.make with first argument of type {callable} but only "
                             f"Callable[(Context) -> str] is allowed!")
        instance.do_call = callable
        if isinstance(options_or_number, int):
            instance.number = options_or_number
        elif isinstance(options_or_number, list):
            instance.options = options_or_number
        else:
            raise ValueError(f"Called F.make with second argument: {options_or_number} "
                             f"of type {type(options_or_number)}, but only int or List[str] is allowed!")

        return instance

    def __call__(self, ctx: 'Context') -> str:
        return self.do_call(ctx)


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


Entry = namedtuple("Entry", ["id", "passage", "qa_id", "question", "answer", "qa"])


@dataclass
class World:
    num_sentences: int


@dataclass
class Bundle:
    generator: Any
    planned_generator: Any
    # templates: Dict[str, Any]
    generator_modifier: Any
    templates_modifier: Dict[str, Any]
    reasoning_map: Dict[str, List[str]]
    has_template_attribute: Callable[[str, Union[str, List[str]]], str]
    world: Callable[[Any], World]


@dataclass
class Event:
    sentence_nr: int
    event_type: str = None
    attributes: Dict[str, Union[str, Any]] = None
    actor: Any = None
    features: List[str] = field(default_factory=list)


@dataclass
class Context:
    world: World = None
    sentences: List[Event] = field(default_factory=list)
    chosen_templates: List[Tuple[str, int]] = None
    visits: Dict[int, List[str]] = field(default_factory=dict)
    sent: Event = None
    realized: List[str] = field(default_factory=list)
    choices: List[List[Tuple[str, Any]]] = field(default_factory=list)
    stack: List[str] = field(default_factory=list)
    word: str = None
    other: Dict[str, Any] = field(default_factory=dict)
    realizer: Any = None
    sent_nr: int = None

    @property
    def current_choices(self) -> List[Tuple[str, Any]]:
        return self.choices[self.sent_nr]


class QuestionTypes:
    DIRECT = "direct"
    OVERALL = 'overall'


class ReasoningTypes:
    Retrieval = "retrieval"  # just retrieve as it is
    MultiRetrieval = 'multi-retrieval'  # retrieve multiple passages as it is
    OrderingEasy = 'ordering-easy'  # order of appearance == actual order (temporal, math, etc)
    OrderingHard = 'ordering-hard'  # order of appearance != actual order (e.g. "x, but before that y" => y < x)


@dataclass
class Question:
    type: str
    target: str
    evidence: List[int]
    event_type: str
    reasoning: str  # something like retrieval, counting, etc
    answer: str
    question_data: Dict[str, Any] = field(default_factory=dict)
    realized: Optional[str] = None


class Model:
    """
    Unifying interface to all those stupid things.
    """

    def __init__(self, name, something_with_predict, gpu=False):
        self.name = name
        self.predictor = something_with_predict
        self.gpu = gpu

    def predict(self, entry: Entry) -> str:
        # TODO: something something GPU
        question = entry.question
        passage = entry.passage
        return self.predictor.predict(question=question, passage=passage)['best_span_str']

    def predict_batch(self, batch: Iterable[Entry]) -> Iterable[str]:
        return (self.predictor.predict(question=entry.question, passage=entry.passage) for entry in batch)

    @classmethod
    def make(cls, path, gpu=False):
        ...


@dataclass(frozen=True)
class EventPlan:
    num_modifications: int
    modification_distance: int
    first_modification: int
    modify_event_type: str
    question_target: str
    event_types: Tuple[str]
    must_haves: List[str]
    reasoning_type: 'Reasoning'
    to_question: Callable[[List[Event], bool, Any], Question] = field(repr=False, compare=False)
    # question_plan: QuestionPlan

    JUST: str = field(default='just', init=False, repr=False, compare=False)
    ANY: str = field(default='any', init=False, repr=False, compare=False)
    NOT: str = field(default='not', init=False, repr=False, compare=False)
    MOD: str = field(default='modified', init=False, repr=False, compare=False)


@dataclass
class Reasoning:
    name: str
    cardinality_event_plans: Callable[[int], int] = field(repr=False, compare=False)
    questions_per_event_plan: int
    generate_all_event_plans: Callable[[int, str, List[str]], List[EventPlan]] = field(repr=False, compare=False)
