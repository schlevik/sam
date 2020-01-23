from abc import ABC, abstractmethod
from typing import Iterable

from quicklog import Loggable

from stresstest.classes import Choices, Path
from stresstest.util import get_sentence_of_word


class BaseRealisationRule(Loggable, ABC):
    """
    This is the base class for the implementing realisation conditions.

    Conditions should inherit and implement `evaluate_condition`.
    """

    def __init__(self, key: str):
        """
        This condition will only trigger when the given key is being
        realised.

        Args:
            key: Key that triggers this condition
        """
        self.key = key

    def __call__(self, *,
                 possible_choices: Choices,
                 keys: Iterable[str],
                 path: Path = None,
                 realised_path: Path = None,
                 position: int = None):
        """
        Evaluates the condition if the given key equals to the
        implementing condition's key.

        Args:
            possible_choices:
                Choices to choose from so far.
            keys:
                Key to compare the condition key to.
            path:
                (non-realised) path.
            realised_path:
                (partly) realised path.
            position:
                Position in path.

        Returns:
            Set of possible choices. Some choices were potentially
            removed by the condition.

        """
        key = ".".join(keys)
        if key == self.key:
            return self.evaluate_condition(path=path,
                                           realised_path=realised_path,
                                           possible_choices=possible_choices,
                                           position=position)
        return possible_choices

    @abstractmethod
    def evaluate_condition(self, *,
                           possible_choices,
                           path=None,
                           realised_path=None,
                           position=None):
        """
        Should evaluate the condition to remove unwanted choices
        according to the condition logic.

        Args:
            possible_choices: possible choices before applying condition
            path: (non realised) path
            realised_path: (partly) realised path
            position: Position in path

        Returns:
            Should return a set of conditions that are allowed from
            the point of view of this condition.

        """
        ...


class SingularPlural(BaseRealisationRule):
    """
    Sets the right numerus for verbs.

    Used to make grammatical usage of was for player and were for team.
    """
    def __init__(self):
        super().__init__("path.VBZ-VBP")

    def evaluate_condition(self, *, possible_choices, path=None,
                           realised_path: Path = None,
                           position=None):
        if not (position and path):
            raise ValueError(
                f"{self.__class__.__name__} requires the position and "
                f"realised_path kwargs!")
        sentence = path[get_sentence_of_word(position, path)]
        if any(a.endswith('team') for a in sentence):
            possible_choices.remove('was')
        elif any(a.endswith('player') for a in sentence):
            possible_choices.remove('were')
        return possible_choices


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

    def evaluate_condition(self, *, possible_choices: Choices, path=None,
                           realised_path=None, position=None):
        if not (position and path):
            raise ValueError(
                f"{self.__class__.__name__} requires the position and "
                f"realised_path kwargs!")
        sentence = path[get_sentence_of_word(position, path)]
        if "modifier.altering" not in sentence:
            possible_choices.remove('$ALTERING')
        if "modifier.non-altering" not in sentence:
            possible_choices.remove('$NON-ALTERING')
        return possible_choices
