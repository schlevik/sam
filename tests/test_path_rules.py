import unittest

from quicklog import setup_logging

from stresstest.classes import Choices, Path
from stresstest.passage.rules import AtLeastOneSentence, UniqueElaborations, \
    NoFoulTeam, NPlayersMention, GoalWithDistractor
from stresstest.util import choices_at, load_graph

setup_logging('tests/resources/logging.conf')

full_one_sentence = Path(
    ['start', 'idle', 'sos', 'attribution',
     '.player', 'action-description', '.goal'])

graph = load_graph('tests/resources/unnamed0.graphml')


class TestAtLeastOneSentence(unittest.TestCase):
    choices = Choices(['sos', 'end'])
    expected = Choices(['sos'])

    def test_fires_when_no_sentence(self):
        path = Path(['start', 'idle'])
        rule = AtLeastOneSentence()
        assert (rule.evaluate_rule(path=path,
                                   choices=self.choices) ==
                self.expected)

    def test_doesnt_fire_when_one_sentence(self):
        path = full_one_sentence
        rule = AtLeastOneSentence()

        assert (rule(path=path,
                     choices=self.choices) ==
                self.choices)

    def test_doesnt_add_new_choices(self):
        path = full_one_sentence
        rule = AtLeastOneSentence()
        empty_choices = Choices([])
        assert (rule.evaluate_rule(path=path,
                                   choices=empty_choices) ==
                empty_choices)


class TestUniqueElaborations(unittest.TestCase):
    def test_removes_impossible_elaboration(self):
        choices = Choices(['elaboration', 'eos'])
        expected = Choices(['eos'])
        path = Path(full_one_sentence.steps + ['elaboration', '._distance',
                                               'elaboration', '._position',
                                               'elaboration', '._type'])
        rule = UniqueElaborations()
        assert (rule.evaluate_rule(path=path,
                                   choices=choices) == expected)

    def test_removes_used_distance_elaboration(self):
        choices = Choices(['._distance',
                           '._position',
                           '._type', 'eos'])
        expected = choices - '._distance'
        path = full_one_sentence + ['elaboration', '._distance', 'elaboration']
        rule = UniqueElaborations()
        assert (rule.evaluate_rule(path=path,
                                   choices=choices) == expected)

    def test_removes_used_type_elaboration(self):
        choices = Choices(['._distance',
                           '._position',
                           '._type', 'eos'])
        expected = choices - '._type'
        path = full_one_sentence + ['elaboration', '._type', 'elaboration']
        rule = UniqueElaborations()
        assert (rule.evaluate_rule(path=path,
                                   choices=choices) == expected)

    def test_removes_used_position_elaboration(self):
        choices = Choices(['._distance',
                           '._position',
                           '._type', 'eos'])
        expected = choices - '._position'
        path = full_one_sentence + ['elaboration', '._position', 'elaboration']
        rule = UniqueElaborations()
        assert (rule.evaluate_rule(path=path,
                                   choices=choices) == expected)

    def test_does_not_affect_new_sentence(self):
        path = full_one_sentence + ['elaboration', '._position',
                                    'eos'] + full_one_sentence + ['elaboration']
        choices = Choices(['._distance',
                           '._position',
                           '._type', 'eos'])
        path = full_one_sentence + ['elaboration', '._position', 'elaboration']
        rule = UniqueElaborations()
        assert (rule.evaluate_rule(path=path,
                                   choices=choices) == choices)

    def test_doesnt_add_new_choices(self):
        path = full_one_sentence
        rule = AtLeastOneSentence()
        empty_choices = Choices([])
        assert (rule.evaluate_rule(path=path,
                                   choices=empty_choices) ==
                empty_choices)


class TestNoFoulTeam(unittest.TestCase):

    def test_no_foul_team(self):
        path = Path(['start', 'idle', 'sos', 'attribution', '._team',
                     'action-description'])
        choices = Choices(['.goal', '.foul', '.freekick'])
        expected = choices - '.foul'
        rule = NoFoulTeam()
        assert (rule.evaluate_rule(path=path,
                                   choices=choices) == expected)

    def test_does_not_affect_player(self):
        path = Path(['start', 'idle', 'sos', 'attribution', '._player',
                     'action-description'])
        choices = Choices(['.goal', '.foul', '.freekick'])
        rule = NoFoulTeam()
        assert (rule.evaluate_rule(path=path,
                                   choices=choices) == choices)

    def test_does_not_affect_new_sentence(self):
        path = Path(['start', 'idle', 'sos', 'attribution', '._team',
                     'action-description', '.goal', 'eos', 'idle', 'sos',
                     'attribution', '._player'])
        choices = Choices(['.goal', '.foul', '.freekick'])
        rule = NoFoulTeam()
        assert (rule.evaluate_rule(path=path,
                                   choices=choices) == choices)

    def test_doesnt_add_new_choices(self):
        path = Path(['start', 'idle', 'sos', 'attribution', '._player',
                     'action-description'])
        rule = NoFoulTeam()
        empty_choices = Choices([])
        assert (rule.evaluate_rule(path=path,
                                   choices=empty_choices) ==
                empty_choices)


class TestNPlayerMention(unittest.TestCase):
    def test_no_stop_until_two_player_mentioned(self):
        path = full_one_sentence + ['eos', 'idle']
        choices = Choices(['end', 'sos'])
        expected = Choices(['sos'])
        rule = NPlayersMention()
        assert (rule.evaluate_rule(path=path,
                                   choices=choices) ==
                expected)

    def test_enforces_player_mention_on_attribution(self):
        path = full_one_sentence + ['eos', 'idle', 'sos', 'attribution']
        choices = Choices(['._player', '._team'])
        expected = choices - '._team'
        rule = NPlayersMention()

        assert (rule.evaluate_rule(path=path,
                                   choices=choices) ==
                expected)

    def test_doesnt_alter_when_two_players_mentioned(self):
        path = full_one_sentence + ['eos', 'idle']
        path = path + path
        path = path + ['sos', 'attribution']
        choices = Choices(['._player', '._team'])
        rule = NPlayersMention()
        assert (rule.evaluate_rule(path=path,
                                   choices=choices) ==
                choices)

    def test_doesnt_add_new_choices(self):
        path = full_one_sentence + ['eos', 'idle']
        rule = NPlayersMention()
        empty_choices = Choices([])
        assert (rule.evaluate_rule(path=path,
                                   choices=empty_choices) ==
                empty_choices)


class TestGoalWithDistractor(unittest.TestCase):
    def test_works_as_expected(self):
        path = Path(['start', 'idle', 'sos'])
        choices = choices_at(graph, 'sos')
        rule = GoalWithDistractor()
        rule.evaluate_rule(path=path, choices=choices)
        assert (rule.sentence_position == 1)

        old_next = 'attribution'

        for n in ['._player', 'modifier', '.altering',
                  'action-description', '.goal', 'eos', 'sos']:
            path += [old_next]
            choices = choices_at(graph, old_next)
            new_next = n
            expected = Choices([new_next])
            assert (rule.evaluate_rule(path=path,
                                       choices=choices) == expected)
            old_next = new_next
        choices = choices_at(graph, 'sos')
        path += ['sos']
        rule.evaluate_rule(path=path, choices=choices)
        assert rule.sentence_position == 2

        old_next = 'attribution'
        for n in ['._player',
                  'action-description', '.goal', 'eos']:
            path += [old_next]
            choices = choices_at(graph, old_next)
            new_next = n
            expected = Choices([new_next])
            assert (rule.evaluate_rule(path=path,
                                       choices=choices) == expected)
            old_next = new_next
