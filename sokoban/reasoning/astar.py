"""
A* Heuristic Search Planner for Sokoban with Bipartite Matching and Deadlock Pruning.
"""

import time
import heapq
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from scipy.optimize import linear_sum_assignment

from sokoban.core.grid import (
    SokobanState, Action, ACTIONS, canonical_state_hash
)
from sokoban.reasoning.deadlock import DeadlockDetector


def min_cost_bipartite_heuristic(boxes: Set[Tuple[int, int]], targets: Set[Tuple[int, int]]) -> float:
    """
    Admissible and consistent heuristic using Minimum Cost Bipartite Matching
    via Hungarian algorithm (linear_sum_assignment).
    """
    if not boxes or not targets:
        return 0.0

    boxes_list = list(boxes)
    targets_list = list(targets)
    n_boxes = len(boxes_list)
    n_targets = len(targets_list)

    cost_matrix = np.zeros((n_boxes, n_targets), dtype=np.float32)
    for i, (br, bc) in enumerate(boxes_list):
        for j, (tr, tc) in enumerate(targets_list):
            cost_matrix[i, j] = abs(br - tr) + abs(bc - tc)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_min_dist = float(cost_matrix[row_ind, col_ind].sum())
    return total_min_dist


class AStarSolver:
    """A* Search planner with Hungarian heuristic and state transposition table."""

    def __init__(self, state: SokobanState, max_expansions: int = 15000):
        self.initial_state = state
        self.max_expansions = max_expansions
        self.deadlock_detector = DeadlockDetector(
            set(state.walls), set(state.targets), state.height, state.width
        )

    def solve(self) -> Tuple[Optional[List[Action]], int, float, bool]:
        """
        Execute A* search.
        Returns: (action_sequence, expanded_nodes, elapsed_ms, success)
        """
        start_time = time.perf_counter()

        if self.initial_state.is_solved():
            return [], 0, 0.0, True

        # Priority queue entries: (f_score, g_score, counter, state, action_history)
        counter = 0
        h_0 = min_cost_bipartite_heuristic(set(self.initial_state.boxes), set(self.initial_state.targets))
        open_set = [(h_0, 0, counter, self.initial_state, [])]

        # Transposition table: hash -> lowest g_score
        g_scores: Dict[int, int] = {hash(self.initial_state): 0}
        expanded_nodes = 0

        while open_set and expanded_nodes < self.max_expansions:
            f, g, _, current_state, path = heapq.heappop(open_set)
            expanded_nodes += 1

            if current_state.is_solved():
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return path, expanded_nodes, elapsed_ms, True

            curr_hash = hash(current_state)
            if g > g_scores.get(curr_hash, float('inf')):
                continue

            for act in ACTIONS:
                next_state, is_valid, is_push = current_state.apply_action(act)
                if not is_valid or next_state is None:
                    continue

                # Deadlock pruning
                if is_push:
                    is_dead, _ = self.deadlock_detector.is_deadlock(next_state)
                    if is_dead and not next_state.is_solved():
                        continue

                next_hash = hash(next_state)
                next_g = g + 1

                if next_g < g_scores.get(next_hash, float('inf')):
                    g_scores[next_hash] = next_g
                    h = min_cost_bipartite_heuristic(set(next_state.boxes), set(next_state.targets))
                    f_score = next_g + h
                    counter += 1
                    heapq.heappush(open_set, (f_score, next_g, counter, next_state, path + [act]))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return None, expanded_nodes, elapsed_ms, False


def astar_solve(state: SokobanState, max_expansions: int = 15000) -> Tuple[Optional[List[Action]], int, float, bool]:
    solver = AStarSolver(state, max_expansions=max_expansions)
    return solver.solve()
