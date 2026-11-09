"""
Backtracking Solver with Cycle Detection, Transposition Table, and Deadlock Pruning.
"""

import time
from typing import List, Tuple, Optional, Set, Dict
from sokoban.core.grid import SokobanState, Action, ACTIONS
from sokoban.reasoning.deadlock import DeadlockDetector


class BacktrackingSolver:
    """Depth-First Backtracking search with cycle detection and deadlock pruning."""

    def __init__(self, state: SokobanState, max_depth: int = 50, max_expansions: int = 25000):
        self.initial_state = state
        self.max_depth = max_depth
        self.max_expansions = max_expansions
        self.deadlock_detector = DeadlockDetector(
            set(state.walls), set(state.targets), state.height, state.width
        )
        self.expanded_nodes = 0
        self.visited_states: Dict[int, int] = {}  # hash -> min depth reached

    def solve(self) -> Tuple[Optional[List[Action]], int, float, bool]:
        start_time = time.perf_counter()
        self.expanded_nodes = 0
        self.visited_states.clear()

        path: List[Action] = []
        success = self._dfs(self.initial_state, path, depth=0)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return (path if success else None), self.expanded_nodes, elapsed_ms, success

    def _dfs(self, state: SokobanState, path: List[Action], depth: int) -> bool:
        self.expanded_nodes += 1

        if state.is_solved():
            return True

        if depth >= self.max_depth or self.expanded_nodes >= self.max_expansions:
            return False

        state_h = hash(state)
        if state_h in self.visited_states and self.visited_states[state_h] <= depth:
            return False
        self.visited_states[state_h] = depth

        for act in ACTIONS:
            next_state, is_valid, is_push = state.apply_action(act)
            if not is_valid or next_state is None:
                continue

            if is_push:
                deadlocked, _ = self.deadlock_detector.is_deadlock(next_state)
                if deadlocked and not next_state.is_solved():
                    continue

            path.append(act)
            if self._dfs(next_state, path, depth + 1):
                return True
            path.pop()  # Backtrack

        return False


def backtracking_solve(state: SokobanState, max_depth: int = 50, max_expansions: int = 25000) -> Tuple[Optional[List[Action]], int, float, bool]:
    solver = BacktrackingSolver(state, max_depth=max_depth, max_expansions=max_expansions)
    return solver.solve()
