import random
from typing import Callable

import networkx as nx
from ailog import Loggable


class Path(Loggable):
    def __init__(self):
        self.steps = []

    def push(self, node):
        self.logger.debug(f"{self} ++ {node}")
        self.steps.append(node)

    def __str__(self):
        return str(self.steps)


def random_strategy(node: str, graph: nx.Graph, *args, **kwargs):
    return random.choice(list(graph.neighbors(node)))


def generate_path(graph: nx.Graph, start_node: str, end_node: str,
                  strategy: Callable[[str, nx.Graph, Path], str]) -> Path:
    path = Path()
    node = start_node
    path.push(node)
    while node != end_node:
        node = strategy(node, graph, path)
        path.push(node)

    return path
