import os
import shutil
import string
import tempfile
from typing import List, Set

import networkx as nx
from ailog import Loggable
from pyvis.network import Network


def load_graph(path) -> nx.Graph:
    g = nx.read_graphml(path)
    return expand(convert(g))


def convert(graph: nx.Graph) -> nx.Graph:
    """
    Converts a graph where nodes are referred to as numbers to a graph
    where nodes are referred to as their labels.

    Also gets rid of unnecessary data.
    Args:
        graph: Graph to convert.

    Returns: (newly constructed) Converted Graph

    """
    node_to_label = dict(graph.nodes('label'))
    g = nx.DiGraph()
    for u, v in graph.edges():
        g.add_edge(node_to_label[u], node_to_label[v])
    return g


def expand(graph: nx.Graph) -> nx.Graph:
    """
    Expands the graph along the .attribute paths.

    Args:
        graph:

    Returns:

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


# def random_sample_keys(cfg: Mapping):
#    return random.sample(list(cfg.keys()))


def alphnum(s):
    return "".join(c for c in s if c not in string.punctuation)


def in_sentence(path):
    try:
        sos_index = path.rindex("sos")
    except ValueError:
        return False
    try:
        eos_index = path.rindex("eos")
    except ValueError:
        return True
    return sos_index > eos_index


def get_sentence_of_word(word: int, path: 'Path') -> slice:
    sos_index = word
    while path[sos_index] != 'sos':
        sos_index -= 1
    eos_index = word
    while path[eos_index] != 'eos':
        eos_index += 1
    return slice(sos_index, eos_index)


def in_same_sentence(one: int, other: int, path: 'Path'):
    return get_sentence_of_word(one, path) == get_sentence_of_word(other, path)
