from stresstest.state import StartClause, GoalClause, TeamClause, \
    EndClause
from stresstest.classes import Path

test_path = Path(
    StartClause(),
    GoalClause(),
    TeamClause(),
    EndClause()
)
