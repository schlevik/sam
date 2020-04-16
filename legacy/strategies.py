from typing import List, Mapping

import networkx as nx
from overrides import overrides
from quicklog import Loggable

from stresstest.classes import Choices
from legacy.classes import Path
from legacy.rules import PassageRule


class ReasonableStrategy(Loggable):
    """
    A "reasonable" (as opposed to random) strategy to select next
    steps for a given graph.

    The "reasoning" of the strategy is expressed as a given set of
    symbolic rules that impose constraints on which nodes can be
    selected as a next element in the path.
    """

    def __init__(self, rules: List[PassageRule]):
        """
        Instantiates the strategy.

        Args:
            rules: Rules that guide the content generation.
        """
        self.rules: List[PassageRule] = rules

    def __call__(self, graph: nx.Graph, path: Path) -> str:
        """
        For a given content graph and the path generated so far,
        randomly selects a neighbor of the most recently selected node
        in the graph that also confirms with the rules.

        Args:
            graph:
                Content graph to select the next node from.
            path:
                Path so far. Cannot be empty.

        Returns:

        """
        possible_choices = Choices(graph.neighbors(path.last))
        for rule in self.rules:
            possible_choices = rule(path, possible_choices)
            self.logger.info(f"type: {type(possible_choices)}")
        result = possible_choices.random()
        if not result:
            names = [f.__class__.__name__ for f in self.rules]
            self.logger.error(f"The set of rules {names} yielded no choice for "
                              f"the node '{path.last}' in the path \n{path}")
            raise ValueError("Better check your config bro...")
        return result


class VariableResolvingStrategy(ReasonableStrategy):
    def __init__(self, rules: List[PassageRule],
                 underscores: Mapping[str, List[str]]):
        super().__init__(rules)
        self.underscores = underscores

    @overrides
    def __call__(self, graph: nx.Graph, path: Path):
        result = super()(graph, path)
        if result.startswith("_"):
            return Choices(self.underscores[result]). \
                random_with_rules(rules=self.rules, path=path)
