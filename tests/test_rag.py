"""
Unit tests for RAG knowledge base, retriever, and strategic reasoning advisor.
"""

import unittest
from sokoban.core.levels import get_level
from sokoban.rag_llm.knowledge_base import SOKOBAN_TACTICAL_KNOWLEDGE
from sokoban.rag_llm.retriever import SokobanRAGRetriever
from sokoban.rag_llm.strategy_advisor import SokobanStrategyAdvisor


class TestRAGStrategicReasoning(unittest.TestCase):

    def test_knowledge_base_loaded(self):
        self.assertGreater(len(SOKOBAN_TACTICAL_KNOWLEDGE), 3)

    def test_retriever_query(self):
        retriever = SokobanRAGRetriever()
        results = retriever.retrieve("corner deadlock danger wall intersection", top_k=2)
        self.assertEqual(len(results), 2)
        doc, sim = results[0]
        self.assertIn("Corner", doc.title)
        self.assertGreater(sim, 0.1)

    def test_strategy_advisor_context_vector(self):
        state = get_level("microban_1")
        advisor = SokobanStrategyAdvisor()
        ctx_vec, advice = advisor.generate_strategy_context(state)

        self.assertEqual(ctx_vec.shape, (16,))
        self.assertIn("explanation", advice)
        self.assertIn("primary_rule", advice)


if __name__ == "__main__":
    unittest.main()
