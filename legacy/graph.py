import random
from typing import Callable

import networkx as nx

from legacy.classes import Path


def random_strategy(graph: nx.Graph, path: Path) -> str:
    """
    Random Strategy. Chooses a random adjacent node that is not in path.

    Works for sentences.

    Args:
        graph:
            Content graph.
        path:
            path so far.

    Returns:
        : Next node from the graph.
    """
    return random.choice(
        [n for n in graph.neighbors(path.steps[-1]) if n not in path.steps])


def generate_path(graph: nx.Graph, start_node: str, end_node: str,
                  strategy: Callable[[nx.Graph, Path], str]) -> Path:
    """
    Generates the logical representation of content in form of a path
    through a given content graph.
    Args:
        graph:
            The given content graph.
        start_node:
            The start of the path. The first node in the generated path
            will always be this.
        end_node:
            The end node of the path. The generation will stop as soon
            as the end node is reached.
        strategy:
            The strategy responsible for the selection of a next node
            in the path given all the previous choices.

    Returns:

    """
    path = Path()
    node = start_node
    path.push(node)
    while node != end_node:
        node = strategy(graph, path)
        path.push(node)

    return path
