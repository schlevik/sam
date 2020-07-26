from dataclasses import dataclass, field
from typing import Tuple, List, Dict

from stresstest.classes import World


@dataclass
class Team:
    id: str
    name: str


@dataclass
class Player:
    id: str
    first: str
    last: str
    team: Team
    position: str


class FootballWorld(World):
    gender: str
    teams: Tuple[Team, Team]
    num_players: int
    players: List[Player]
    players_by_id: Dict[str, Player]

    MALE: str = field(default='male', init=False, repr=False, compare=False)
    FEMALE: str = field(default='female', init=False, repr=False, compare=False)
