import pytest
from quicklog import setup_logging

from legacy.strategies import ReasonableStrategy
from stresstest.classes import Choices
from legacy.classes import Path
from legacy.rules import PassageRule
from legacy.util import load_graph

setup_logging('tests/resources/logging.conf')

full_one_sentence = Path(
    ['start', 'idle', 'sos', 'attribution',
     '.player', 'action-description', '.goal'])

graph = load_graph('tests/resources/unnamed0.graphml')


class TestRule(PassageRule):
    def evaluate_rule(self, *, path: Path, choices: Choices) -> Choices:
        choices.remove_all_but("idle")
        return choices


class TestRuleImpossible(PassageRule):
    def evaluate_rule(self, *, path: Path,
                      choices: Choices) -> Choices:
        choices.remove_all_but()
        return choices


class TestReasonableStrategy:
    def test_strategy_applies_rules(self):
        r = TestRule()
        s = ReasonableStrategy([r])
        path = Path(['start'])
        assert s(graph, path) == "idle"

    def test_strategy_throws_error_when_rule_combination_impossible(self):
        r = TestRuleImpossible()
        s = ReasonableStrategy([r])
        path = Path(['start'])
        with pytest.raises(ValueError):
            s(graph, path)