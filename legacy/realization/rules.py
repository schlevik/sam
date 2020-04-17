from abc import abstractmethod
from typing import Iterable

from stresstest.classes import Choices, Rule
from legacy.classes import Path
from legacy.util import get_sentence_of_word


class BaseRealisationRule(Rule):
    """
    This is the base class for the implementing realisation rules.

    Rules should inherit and implement `evaluate_rule`.
    """

    def __init__(self, key: str):
        """
        This rule will only trigger when the given key is being
        realised.

        Args:
            key: Key that triggers this rule
        """
        self.key = key

    def __call__(self, *,
                 choices: Choices,
                 keys: Iterable[str],
                 path: Path = None,
                 realised_path: Path = None,
                 position: int = None):
        """
        Evaluates the rule if the given key equals to the
        implementing rule's key.

        Args:
            choices:
                Choices to choose from so far.
            keys:
                Key to compare the rule key to.
            path:
                (non-realised) path.
            realised_path:
                (partly) realised path.
            position:
                Position in path.

        Returns:
            Set of possible choices. Some choices were potentially
            removed by the rule.

        """
        key = ".".join(keys)
        if key == self.key:
            return self.evaluate_rule(path=path,
                                      realised_path=realised_path,
                                      choices=choices,
                                      position=position)
        return choices

    @abstractmethod
    def evaluate_rule(self, *,
                      choices,
                      path=None,
                      realised_path=None,
                      position=None):
        """
        Should evaluate the rule to remove unwanted choices
        according to the rule logic.

        Args:
            choices: possible choices before applying rule
            path: (non realised) path
            realised_path: (partly) realised path
            position: Position in path

        Returns:
            Should return a set of rules that are allowed from
            the point of view of this rule.

        """
        ...


class SingularPlural(BaseRealisationRule):
    """
    Sets the right numerus for verbs.

    Used to make grammatical usage of was for player and were for team.
    """
    def __init__(self):
        super().__init__("path.VBZ-VBP")

    def evaluate_rule(self, *, choices, path=None,
                      realised_path: Path = None,
                      position=None):
        if not (position and path):
            raise ValueError(
                f"{self.__class__.__name__} requires the position and "
                f"realised_path kwargs!")
        sentence = path[get_sentence_of_word(position, path)]
        if any(a.endswith('team') for a in sentence):
            choices.remove('was')
        elif any(a.endswith('player') for a in sentence):
            choices.remove('were')
        return choices


class Modifier(BaseRealisationRule):
    """
    Sets the right modifier realisation for templates that make use
    of a modifier.

    Checks whether there is a modifier required in the sentence,
    its type (altering/non-altering) and removes the corresponding
    choices.
    """
    def __init__(self):
        super().__init__("path.MODIFIER")

    def evaluate_rule(self, *, choices: Choices, path=None,
                      realised_path=None, position=None):
        if not (position and path):
            raise ValueError(
                f"{self.__class__.__name__} requires the position and "
                f"realised_path kwargs!")
        sentence = path[get_sentence_of_word(position, path)]
        if "modifier.altering" not in sentence:
            choices.remove('$ALTERING')
        if "modifier.non-altering" not in sentence:
            choices.remove('$NON-ALTERING')
        return choices
