from typing import List, Callable

import networkx as nx
from ailog import Loggable

from stresstest.classes import Path, Choices


class ReasonableStrategy(Loggable):
    def __init__(self, conditions):
        self.conditions: List[Callable[[Path, Choices], str]] = conditions

    def __call__(self, graph: nx.Graph, path: Path) -> str:
        possible_choices = Choices(graph.neighbors(path.last))
        for condition in self.conditions:
            possible_choices = condition(path, possible_choices)
        try:
            return possible_choices.random()
        except IndexError:
            names = [f.__name__ for f in self.conditions]
            self.logger.error(f"The set of conditions {names} yielded no choice"
                              f" for the node {path.last} in the path \n{path}")
            raise ValueError("Better check your config bro...")
