import networkx as nx
import pytest
from quicklog import setup_logging

from stresstest.classes import Path
from stresstest.util import load_graph, convert, expand, in_sentence, \
    get_sentence_of_word, in_same_sentence

setup_logging('tests/resources/logging.conf')
graph = load_graph('tests/resources/unnamed0.graphml')


class TestUtilFuncs:
    def test_convert_works(self):
        graph = nx.DiGraph()
        graph.add_node(0, label="a")
        graph.add_node(1, label="b")
        graph.add_node(2, label="c")
        graph.add_edge(0, 1)
        converted_graph = convert(graph)
        assert ("a", "b") in converted_graph.edges
        assert ('b', 'a') not in converted_graph  # directed graph

    def test_expand_works(self):
        graph = nx.DiGraph()
        graph.add_edge('a', 'b')
        graph.add_edge('a', '.c')
        expanded_graph = expand(graph)
        assert ('a', 'b') not in expanded_graph.edges
        assert (".c", 'b') in expanded_graph.edges

    def test_in_sentence(self):
        path = Path(['sos', 'a', 'b', 'c'])
        assert in_sentence(path)
        path += ["eos"]
        assert not in_sentence(path)
        path = Path(['start', 'idle'])
        assert not in_sentence(path)

    def test_sentence_of_word(self):
        path = Path(['start', 'idle', 'sos', 'a', 'b',
                     'c', 'eos', 'sos', 'a', 'eos', 'idle', ])
        assert get_sentence_of_word(4, path) == slice(2, 6)

    def test_sentence_of_word_fails_when_not_in_sentence(self):
        path = Path(['start', 'idle', 'sos', 'a', 'b',
                     'c', 'eos', 'sos', 'a', 'eos', 'idle'])

        with pytest.raises(ValueError):
            get_sentence_of_word(10, path)
        with pytest.raises(ValueError):
            get_sentence_of_word(0, path)

    def test_in_same_sentence_works(self):
        path = Path(['start', 'idle', 'sos', 'a', 'b',
                     'c', 'eos', 'sos', 'a', 'eos', 'idle', ])
        assert in_same_sentence(3, 5, path)
        assert not in_same_sentence(3, 8, path)

    def test_in_same_returns_false_when_not_in_sentence(self):
        path = Path(['start', 'idle', 'sos', 'a', 'b',
                     'c', 'eos', 'sos', 'a', 'eos', 'idle', ])
        assert not in_same_sentence(0, 1, path)
        assert not in_same_sentence(0, 4, path)
        assert not in_same_sentence(1, 9, path)
        assert not in_same_sentence(1, 9, path)
