import random
from typing import Callable

import networkx as nx

from stresstest.classes import Path, Choices


def random_strategy(graph: nx.Graph, path: Path) -> str:
    """
    Random Strategy. Chooses a random adjacent node that is not in path.

    Works for sentences.

    Args:
        graph: Content graph.
        path: path so far.

    Returns:
        : Next node from the graph.
    """
    return random.choice(
        [n for n in graph.neighbors(path.steps[-1]) if n not in path.steps])


def reasonable_strategy(graph: nx.Graph, path: Path) -> str:
    possible_choices = Choices(graph.neighbors(path.last))
    return possible_choices.random()


def generate_path(graph: nx.Graph, start_node: str, end_node: str,
                  strategy: Callable[[nx.Graph, Path], str]) -> Path:
    path = Path()
    node = start_node
    path.push(node)
    while node != end_node:
        node = strategy(graph, path)
        path.push(node)

    return path
