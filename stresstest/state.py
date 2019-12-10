from abc import ABC, abstractmethod
from typing import Callable, List

from ailog import Loggable

from stresstest.graph import Path


class Condition:
    def __init__(self, conditions: List[Callable[[Path], bool]]):
        self.conditions = conditions

    def evaluate(self, path_so_far: Path):
        return all(condition(path_so_far) for condition in self.conditions)


at_least_one_goal_condition = Condition(
    [lambda p: any(isinstance(GoalClause, c) for c in p)])


class State(Loggable, ABC):
    def __init__(self, clause_type: str):
        self.clause_type = clause_type

    @property
    @abstractmethod
    def preconditions(self):
        ...


class StartClause(State):
    def __init__(self):
        super().__init__("start")
        ...

    preconditions = []


class EndClause(State):
    def __init__(self):
        super().__init__("end")
        ...

    preconditions = [
        at_least_one_goal_condition
    ]


class GoalClause(State):
    def __init__(self):
        super().__init__("goal")
        ...

    preconditions = []


class TeamClause(State):
    def __init__(self):
        super().__init__("team")
        ...

    preconditions = []
