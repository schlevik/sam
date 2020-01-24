import unittest
from random import choice

import pytest
from flaky import flaky
from quicklog import setup_logging

from stresstest.classes import Choices, Path, Rule
from stresstest.util import choices_at, load_graph

setup_logging('tests/resources/logging.conf')
graph = load_graph('tests/resources/unnamed0.graphml')


class TestPath:

    def test_equals(self):
        path_one = Path(['a', 'b', 'c'])

        path_two = Path(['a', 'b', 'c'])
        assert path_one is not path_two
        assert path_one == path_two

    def test_not_equals(self):
        path_one = Path(['a', 'b', 'c' 'e'])

        path_two = Path(['a', 'b', 'c', 'd'])
        assert not path_one == path_two

    def test_addition_is_immutable(self):
        path_one = Path(['a'])

        path_two = Path(['b'])

        list_one = ['c']
        path_three = path_one + path_two
        path_four = path_one + list_one
        assert path_three == Path(['a', 'b'])
        assert path_four == Path(['a', 'c'])
        assert path_one == Path(['a'])
        assert path_two == Path(['b'])

    def test_push_is_mutable(self):
        path_one = Path(['a'])
        path_one.push('b')
        assert path_one == Path(['a', 'b'])

    def test_rindex_works_as_expected(self):
        assert Path(['a', 'b', 'c']).rindex('c') == 2
        assert Path(['a', 'b', 'c', 'b']).rindex('b') == 3

    def test_rindex_raises_value_error_when_not_in_path(self):
        with pytest.raises(ValueError):
            Path(['a']).rindex('b')

    def test_from_path_works(self):
        assert Path(['a', 'b']).from_index(1) == Path('b')

    def test_from_path_fails_if_index_gte_length(self):
        with pytest.raises(ValueError):  # >=
            Path(['a']).from_index(1)
        with pytest.raises(ValueError):  # strictly >
            Path(['a']).from_index(2)

    def test_subclassing_sequence_correctly(self):
        p = Path(['a', 'b', 'c'])
        assert p[1] == 'b'
        assert p[:2] == Path(['a', 'b'])
        assert len(p) == 3

    def test_occurrences(self):
        p = Path(['a', 'c', 'c'])
        assert p.occurrences('a') == 1
        assert p.occurrences('c') == 2
        assert p.occurrences('b') == 0

    def test_last(self):
        p = Path(['a', 'c', 'c'])
        assert p.last == 'c'
        with pytest.raises(IndexError):
            x = Path([]).last


class TestChoices:
    def test_subclassing_correctly(self):
        c = Choices([1, 2, 3])
        for x, y in zip(c, [1, 2, 3]):
            assert x == y

    def test_eq(self):
        assert Choices([1, 2]) == Choices([1, 2])
        assert Choices([1, 2]) is not Choices([1, 2])

        def test_no_duplicate_choices(self):
            assert Choices([1, 2, 2]) == Choices([1, 2])

    def test_remove_is_mutable(self):
        # choices = Choices([1, 2])
        choices = Choices([1, 2])
        choices.remove(1)
        assert choices == Choices([2])

    def test_remove_can_remove_non_existing_choices(self):
        choices = Choices([1, 2])
        choices.remove(3)
        assert choices == Choices([1, 2])

    def test_remove_iterable(self):
        choices = Choices(range(10))
        choices.remove(range(5))
        assert choices == Choices(range(5, 10))

    def test_sub_is_immutable(self):
        choices = Choices(range(10))
        choices2 = choices - 9
        assert choices2 == Choices(range(9))
        assert choices != choices2

    def test_remove_all_but_works(self):
        c = Choices([1, 2, 3])
        c.remove_all_but(1)

        assert c == Choices([1])
        c.remove_all_but(2)
        assert c == Choices([])

    @flaky(max_runs=10, min_passes=10)
    def test_random_with_conditions(self):
        class TestRule(Rule):
            def __call__(self, *, path: Path, choices: Choices,
                         **kwargs) -> Choices:
                assert 'x' in kwargs
                assert kwargs['x'] == 3
                choices.remove_all_but(1)
                return choices

        assert Choices([1, 2, 3]) \
                   .random_with_rules(path=Path([]),
                                      rules=[TestRule()],
                                      x=3) == 1
