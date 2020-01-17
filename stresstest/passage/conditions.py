import logging

from stresstest.classes import Path, Choices
from stresstest.util import in_sentence


def at_least_one_sent(path: Path,
                      possible_choices: Choices) -> Choices:
    """
    This condition ensures that there's at least one content sentence.

    Args:
        path:
        possible_choices:

    Returns:

    """
    if path.last == 'idle' and 'sos' not in path:
        return Choices(['sos'])
    return possible_choices


MAX_ELABORATIONS = 3


def unique_elaborations(path: Path,
                        possible_choices: Choices) -> Choices:
    """
    This condition ensures the elaborations (e.g. elaboration.distance)
    are unique. Also ensures there's no elaboration step after all
    possible elaborations are used up.

    Args:
        path:
        possible_choices:

    Returns:

    """
    if in_sentence(path):
        current_sentence = path.from_index(path.rindex('sos'))
        if current_sentence.occurrences('elaboration') >= MAX_ELABORATIONS:
            possible_choices.remove(['elaboration'])

        if path.last == 'elaboration':
            to_remove = [c for c in possible_choices if c in current_sentence]
            possible_choices.remove(to_remove)
    return possible_choices


def no_foul_team(path: Path, possible_choices: Choices):
    """
    Ensures that when attribution is team, action foul is not possible.

    Args:
        path:
        possible_choices:

    Returns:

    """
    if in_sentence(path):
        current_sentence = path.from_index(path.rindex('sos'))

        # don't foul team
        if "._team" in current_sentence.steps:
            possible_choices.remove(".foul")
    return possible_choices


def two_players_mention(path: Path, possible_choices: Choices):
    if path.last == 'idle' and path.count('._player') < 2:
        return Choices(['sos'])
    if path.last == 'attribution' and path.count('._player') < 2:
        return Choices(['._player'])
    return possible_choices


__all__ = [at_least_one_sent, unique_elaborations, no_foul_team,
           two_players_mention]
