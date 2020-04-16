import os
import shutil
import tempfile
import stresstest
from typing import List, Set, Dict

import networkx as nx
from quicklog import Loggable
from pyvis.network import Network

from stresstest.classes import Choices
from legacy.classes import Path


def load_graph(path: str) -> nx.Graph:
    g = nx.read_graphml(path)
    return expand(convert(g))


def convert(graph: nx.Graph) -> nx.Graph:
    """
    Converts a graph where nodes are referred to as numbers to a graph
    where nodes are referred to as their labels.

    Also gets rid of unnecessary data.

    Assumes that input graph has numeric node identifiers and a data
    field ``label``.

    Args:
        graph: Graph to convert.

    Returns: (newly constructed) Converted Graph

    """
    node_to_label: Dict[int, str] = dict(graph.nodes('label'))
    g = nx.DiGraph()
    for u, v in graph.edges():
        g.add_edge(node_to_label[u], node_to_label[v])
    return g


def expand(graph: nx.Graph) -> nx.Graph:
    """
    Expands the graph along the `.attribute` paths.

    More concretely, for every path of nodes that start with a node
    without leading point (head) followed by a arbitrary number of nodes
    with a leading point (attributes), removes all outgoing neighbors
    from the head and adds them as outgoing neighbors to the last
    attribute.

    Assumes that the given graph is well formed to do so.

    Args:
        graph: Graph to expand.

    Returns:
        Graph expanded towards nodes that start with ``.``.

    """

    graph = nx.DiGraph(graph)

    def is_root(n: str):
        return not n.startswith(".") and any(
            c.startswith('.') for c in graph.neighbors(n))

    def get_leaves(n: str):
        result = set()
        candidates = set(c for c in graph.neighbors(n) if c.startswith('.'))
        while candidates:
            c = candidates.pop()
            c_neighbors = list(graph.neighbors(c))
            if not c_neighbors:
                result.add(c)
            else:
                candidates.update(c_neighbors)
        return result

    tree_roots: List[str] = [n for n in graph.nodes() if is_root(n)]
    tree_leaves: List[Set[str]] = [get_leaves(n) for n in tree_roots]
    for root, leaves in zip(tree_roots, tree_leaves):
        real_neighbors = set(
            n for n in graph.neighbors(root) if not n.startswith("."))
        for l in leaves:
            graph.add_edges_from((l, n) for n in real_neighbors)
        graph.remove_edges_from((root, n) for n in real_neighbors)
    return graph


class PyVisPrinter(Loggable):
    """Class to visualise a (serialized) dataset entry."""

    def __init__(self, path=None):
        self.path = tempfile.mkdtemp(prefix='vis-', dir='/tmp') or path
        self.logger.debug(f"Creating graphs in f{self.path}")

    def clean(self):
        shutil.rmtree(self.path)

    def print_graph_and_question(self, graph: nx.Graph):

        vis = Network(height="100%",
                      width="100%",
                      bgcolor="#222222",
                      font_color="white", directed=True)

        for idx, node in enumerate(graph.nodes()):
            vis.add_nodes(
                [node],
                title=[node],
                label=[node],
                color=["yellow"] if node.startswith(".") else ['green'])

        for i, (source, target) in enumerate(graph.edges()):
            if source in vis.get_nodes() and target in vis.get_nodes():
                vis.add_edge(source, target)
            else:
                self.logger.warning("source or target not in graph!")

        name = os.path.join(self.path, f'.html')
        vis.options.physics = False
        vis.show(name)


def in_sentence(path: Path):
    """
    Returns whether the current snippet of the path is in an (open)
    sentence.
    Args:
        path: Path to inspect.

    Returns:
        Whether the path is in the sentence, i.e. if there is an open
        sentence tag (`sos`) but no closed one (`eos`).

    """
    try:
        sos_index = path.rindex("sos")
    except ValueError:
        return False
    try:
        eos_index = path.rindex("eos")
    except ValueError:
        return True
    return sos_index > eos_index


def choices_at(graph: nx.Graph, node: str) -> Choices:
    return Choices(graph.neighbors(node))


def get_sentence_of_word(word: int, path) -> slice:
    """
    Obtains the sentence of a given word position.

    More concretely, returns the positions of the closest "sos" node to
    the left and the closes "eos" position to the right as a ``slice``.

    Args:
        word: Position of word in :class:`stresstest.classes.Path`.
        path: Path to get the sentence from.

    Returns:
        The start and end indices of the sentence to be used as a
        slice.

    """

    sos_index = word
    while sos_index >= 0 and path[sos_index] != 'sos':
        sos_index -= 1
    eos_index = word

    while eos_index < len(path) and path[eos_index] != 'eos':
        eos_index += 1

    if sos_index < 0 or eos_index >= len(path):
        raise ValueError(f"'{path[word]}' is not in sentence for '{path}'!")
    return slice(sos_index, eos_index)


def in_same_sentence(one: int, other: int, path: 'stresstest.classes.Path'):
    """
    Determines whether two words are in the same sentence.

    More concretely checks whether the closest ``sos`` and ``eos`` nodes as
    returned by :any:`get_sentence_of_word` are equal for both words.

    Returns False if any or both words are not in a sentence.

    Args:
        one: Position of one word in the path.
        other: Position of the other word in the path.
        path: Path to get the sentence from.

    Returns:
        ``True`` if words are in same sentence, else ``False``.

    """
    try:
        return get_sentence_of_word(one, path) \
               == get_sentence_of_word(other, path)
    except ValueError:
        return False
