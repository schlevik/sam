import random
from typing import Callable

import networkx as nx
from ailog import Loggable

from stresstest.stringify import Stringifier


class Path(Loggable):
    def __init__(self):
        self.steps = []
        self.stringified = dict()

    def push(self, node):
        self.logger.debug(f"{self} ++ {node}")
        self.steps.append(node)

    def __str__(self):
        return str(self.steps)

    def __repr__(self):
        return repr(self.steps)

    def stringify(self, stringifier: Stringifier):
        s, stringified_map = stringifier.to_string(self.steps)
        self.stringified = stringified_map
        return s


def random_strategy(node: str, graph: nx.Graph, path: Path) -> str:
    return random.choice([n for n in graph.neighbors(node) if n not in path.steps])


def generate_path(graph: nx.Graph, start_node: str, end_node: str,
                  strategy: Callable[[str, nx.Graph, Path], str]) -> Path:
    path = Path()
    node = start_node
    path.push(node)
    while node != end_node:
        node = strategy(node, graph, path)
        path.push(node)

    return path
