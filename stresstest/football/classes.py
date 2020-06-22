from typing import Tuple, List, Dict

from stresstest.classes import DataObjectMixin, World


class Team(DataObjectMixin):
    id: str
    name: str


class Player(DataObjectMixin):
    id: str
    first: str
    last: str
    team: Team
    position: str


class FootballWorld(World):
    MALE = 'male'
    FEMALE = 'female'
    gender: str
    teams: Tuple[Team, Team]
    num_players: int
    players: List[Player]
    players_by_id: Dict[str, Player]
