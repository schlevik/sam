from stresstest.state import StartClause, GoalClause, TeamClause, \
    EndClause
from stresstest.graph import Path

import markovify
test_path = Path(
    StartClause(),
    GoalClause(),
    TeamClause(),
    EndClause()
)
