"""
Unit tests for A* and Backtracking search solvers.
"""

import unittest
from sokoban.core.levels import get_level
from sokoban.reasoning.astar import astar_solve, min_cost_bipartite_heuristic
from sokoban.reasoning.backtracking import backtracking_solve


class TestSearchSolvers(unittest.TestCase):

    def test_bipartite_heuristic(self):
        boxes = {(2, 2), (3, 3)}
        targets = {(1, 1), (3, 3)}
        h = min_cost_bipartite_heuristic(boxes, targets)
        # Box (3,3) matched to target (3,3) -> 0; Box (2,2) to target (1,1) -> |2-1| + |2-1| = 2
        self.assertEqual(h, 2.0)

    def test_astar_solver_microban_1(self):
        state = get_level("microban_1")
        path, nodes, time_ms, succ = astar_solve(state, max_expansions=3000)
        self.assertTrue(succ)
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)

        # Apply path to verify it reaches solved state
        curr = state
        for act in path:
            curr, valid, _ = curr.apply_action(act)
            self.assertTrue(valid)
        self.assertTrue(curr.is_solved())

    def test_backtracking_solver_microban_1(self):
        state = get_level("microban_1")
        path, nodes, time_ms, succ = backtracking_solve(state, max_depth=60)
        self.assertTrue(succ)
        self.assertIsNotNone(path)

        curr = state
        for act in path:
            curr, valid, _ = curr.apply_action(act)
            self.assertTrue(valid)
        self.assertTrue(curr.is_solved())


if __name__ == "__main__":
    unittest.main()
