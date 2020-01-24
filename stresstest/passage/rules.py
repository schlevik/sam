from abc import abstractmethod

from stresstest.classes import Path, Choices, Rule
from stresstest.util import in_sentence


class PassageRule(Rule):
    """
    Base class for passage rules.
    """

    def __call__(self, path: Path, choices: Choices, **kwargs):
        return self.evaluate_rule(path=path, choices=choices)

    @abstractmethod
    def evaluate_rule(self, *, path: Path,
                      choices: Choices) -> Choices:
        """
        Passage rules should implement the logic in this method.

        Args:
            path: Content Path so far.
            choices: Possible neighbors in the content graph
                as choices, minus all choices that did not meet
                rules applied prior to this one.

        Returns:
            Choices from the given ones that also satisfy the
            implementing rule.

        """
        ...


class AtLeastOneSentence(PassageRule):
    """
    This rule ensures that there's at least one content sentence.

    """

    def evaluate_rule(self, *, path: Path,
                      choices: Choices) -> Choices:
        if path.last == 'idle' and 'sos' not in path:
            choices.remove_all_but('sos')
        return choices


class UniqueElaborations(PassageRule):
    """
    This rule ensures the elaborations
    (e.g. ``elaboration.distance``) are unique. Also ensures there's
    no elaboration step after all possible elaborations are used up.

    """

    def __init__(self, max_elaborations=3):
        self.max_elaborations = max_elaborations

    def evaluate_rule(self, *, path: Path,
                      choices: Choices) -> Choices:

        if in_sentence(path):
            current_sentence = path.from_index(path.rindex('sos'))
            if current_sentence.occurrences(
                    'elaboration') >= self.max_elaborations:
                choices.remove(['elaboration'])

            if path.last == 'elaboration':
                to_remove = [c for c in choices if
                             c in current_sentence]
                choices.remove(to_remove)
        return choices


class NoFoulTeam(PassageRule):
    """
    Ensures that when attribution is team, action foul is not possible.

    """

    def evaluate_rule(self, *, path: Path,
                      choices: Choices) -> Choices:

        if in_sentence(path):
            current_sentence = path.from_index(path.rindex('sos'))

            # don't foul team
            if "._team" in current_sentence.steps:
                choices.remove(".foul")
        return choices


class NPlayersMention(PassageRule):
    """
    Ensures that at least two players are mentioned.

    Disables the transition into any state other than start of sentence
    if there is less than `n` player mentions.

    Enforces a player attribution as long as there are less than
    `n` player mentions

    """

    def __init__(self, n=2):
        self.n = n

    def evaluate_rule(self, *, path: Path,
                      choices: Choices) -> Choices:

        if path.last == 'idle' and path.count('._player') < self.n:
            choices.remove_all_but('sos')
        if path.last == 'attribution' and path.count('._player') < self.n:
            choices.remove_all_but('._player')
        return choices


class GoalWithDistractor(PassageRule):
    """
    This will make the passage contain two goals, one with distractor
    (almost) and one without.

    Will run a pre-defined sequence of steps in the first sentence,
    ensuring a modified goal action attributed to a player.

    Will run a pre-defined sequence of steps in the second sentence,
    ensuring a non-modified goal action attributed to a player.

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

    def evaluate_rule(self, *, path: Path,
                      choices: Choices) -> Choices:
        if path.last == 'sos':
            self.sentence_position += 1

        if self.sentence_position == 1 and path.last == 'attribution':
            self.in_predefined_first = True

        if self.in_predefined_first:
            return self._run_predefined(self.predifined_first)

        if self.sentence_position == 2 and path.last == 'attribution':
            self.in_predefined_second = True

        if self.in_predefined_second:
            return self._run_predefined(self.predifined_second)

        return choices
