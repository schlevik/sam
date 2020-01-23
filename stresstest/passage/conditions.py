import logging
from abc import abstractmethod

from stresstest.classes import Path, Choices, Condition
from stresstest.util import in_sentence


class PassageCondition(Condition):
    """
    Base class for passage conditions.
    """

    def __call__(self, path: Path, choices: Choices, **kwargs):
        return self.evaluate_condition(path=path, possible_choices=choices)

    @abstractmethod
    def evaluate_condition(self, *, path: Path,
                           possible_choices: Choices) -> Choices:
        """
        Passage conditions should implement the logic in this method.

        Args:
            path: Content Path so far.
            possible_choices: Possible neighbors in the content graph
                as choices, minus all choices that did not meet
                conditions applied prior to this one.

        Returns:
            Choices from the given ones that also satisfy the
            implementing condition.

        """
        ...


class AtLeastOneSentence(PassageCondition):
    """
    This condition ensures that there's at least one content sentence.

    """

    def evaluate_condition(self, *, path: Path,
                           possible_choices: Choices) -> Choices:
        if path.last == 'idle' and 'sos' not in path:
            return Choices(['sos'])
        return possible_choices


class UniqueElaborations(PassageCondition):
    """
    This condition ensures the elaborations
    (e.g. ``elaboration.distance``) are unique. Also ensures there's
    no elaboration step after all possible elaborations are used up.

    """

    def __init__(self, max_elaborations=3):
        self.max_elaborations = max_elaborations

    def evaluate_condition(self, *, path: Path,
                           possible_choices: Choices) -> Choices:

        if in_sentence(path):
            current_sentence = path.from_index(path.rindex('sos'))
            if current_sentence.occurrences(
                    'elaboration') >= self.max_elaborations:
                possible_choices.remove(['elaboration'])

            if path.last == 'elaboration':
                to_remove = [c for c in possible_choices if
                             c in current_sentence]
                possible_choices.remove(to_remove)
        return possible_choices


class NoFoulTeam(PassageCondition):
    """
    Ensures that when attribution is team, action foul is not possible.

    """

    def evaluate_condition(self, *, path: Path,
                           possible_choices: Choices) -> Choices:

        if in_sentence(path):
            current_sentence = path.from_index(path.rindex('sos'))

            # don't foul team
            if "._team" in current_sentence.steps:
                possible_choices.remove(".foul")
        return possible_choices


class TwoPlayersMention(PassageCondition):
    """
    Ensures that at least two players are mentioned.

    """

    def evaluate_condition(self, *, path: Path,
                           possible_choices: Choices) -> Choices:

        if path.last == 'idle' and path.count('._player') < 2:
            return Choices(['sos'])
        if path.last == 'attribution' and path.count('._player') < 2:
            return Choices(['._player'])
        return possible_choices


class GoalWithDistractor(PassageCondition):
    """
    This will make the passage contain two goals, one with distractor
    (almost) and one without.

    """
    predifined_first = ['attribution', '._player', 'modifier', '.altering',
                        'action-description', '.goal', 'eos', 'sos']
    predifined_second = ['attribution', '._player',
                         'action-description', '.goal', 'eos']

    def __init__(self):
        self.sentence_position = 0
        self.in_predefined_first = False

        self.in_predefined_second = False
        self._internal_counter = 0

    def _run_predefined(self, which):
        if self._internal_counter < len(which) - 2:
            self._internal_counter += 1
            return Choices([which[self._internal_counter]])
        else:
            self._internal_counter = 0
            self.in_predefined_first = False
            self.in_predefined_second = False
            return Choices([which[-1]])

    def evaluate_condition(self, *, path: Path,
                           possible_choices: Choices) -> Choices:
        if path.last == 'sos':
            self.sentence_position += 1

        print(self.sentence_position)
        if self.sentence_position == 1 and path.last == 'attribution':
            self.in_predefined_first = True

        if self.in_predefined_first:
            return self._run_predefined(self.predifined_first)

        if self.sentence_position == 2 and path.last == 'attribution':
            self.in_predefined_second = True

        if self.in_predefined_second:
            return self._run_predefined(self.predifined_second)

        return possible_choices
