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


@dataclass
class FootballWorld(World):
    gender: str = None
    teams: Tuple[Team, Team] = None
    num_players: int = 0
    players: List[Player] = field(default_factory=list)
    players_by_id: Dict[str, Player] = field(default_factory=dict)

    MALE: str = field(default='male', init=False, repr=False, compare=False)
    FEMALE: str = field(default='female', init=False, repr=False, compare=False)
    MOD: str = field(default='modified', init=False, repr=False, compare=False)
