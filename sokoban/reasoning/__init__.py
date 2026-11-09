"""
Reasoning and Search module: Deadlock detection, A* search, Backtracking, and Metrics.
"""

from sokoban.reasoning.deadlock import (
    DeadlockDetector, is_deadlocked_state, get_action_deadlock_mask, compute_static_deadlocks
)
from sokoban.reasoning.astar import AStarSolver, astar_solve, min_cost_bipartite_heuristic
from sokoban.reasoning.backtracking import BacktrackingSolver, backtracking_solve
from sokoban.reasoning.search_eval import evaluate_search_algorithms

__all__ = [
    "DeadlockDetector", "is_deadlocked_state", "get_action_deadlock_mask",
    "compute_static_deadlocks", "AStarSolver", "astar_solve",
    "min_cost_bipartite_heuristic", "BacktrackingSolver", "backtracking_solve",
    "evaluate_search_algorithms"
]
