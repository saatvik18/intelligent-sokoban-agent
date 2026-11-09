"""
Unit tests for the Strict Hybrid Game AI Pipeline.
"""

import unittest
from sokoban.core.levels import get_level
from sokoban.core.environment import SokobanEnv
from sokoban.hybrid.pipeline import HybridSokobanAgent


class TestHybridPipeline(unittest.TestCase):

    def test_hybrid_step_decision(self):
        state = get_level("microban_1")
        agent = HybridSokobanAgent(use_deadlock_mask=True, use_rag_strategy=True)
        action, meta = agent.select_hybrid_action(state)

        self.assertIsNotNone(action)
        self.assertIn("action_name", meta)
        self.assertIn("deadlock_mask", meta)
        self.assertIn("rag_advice", meta)
        self.assertIn("heuristic_bipartite_cost", meta)

    def test_hybrid_episode_solve(self):
        env = SokobanEnv(level_name="microban_1", max_steps=10)
        agent = HybridSokobanAgent(use_deadlock_mask=True, use_rag_strategy=True)
        solved, path, reward, trace = agent.solve(env, max_steps=10)

        self.assertEqual(len(path), len(trace))
        self.assertIsInstance(reward, float)


if __name__ == "__main__":
    unittest.main()
