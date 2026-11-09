"""
Strategy Advisor combining RAG retrieval, rule-based reasoning, and strategic context vector generation.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from sokoban.core.grid import SokobanState
from sokoban.rag_llm.knowledge_base import SOKOBAN_TACTICAL_KNOWLEDGE, KnowledgeDocument
from sokoban.rag_llm.retriever import SokobanRAGRetriever
from sokoban.reasoning.deadlock import is_deadlocked_state, get_action_deadlock_mask
from sokoban.reasoning.astar import min_cost_bipartite_heuristic


class SokobanStrategyAdvisor:
    """
    LLM/SLM + RAG Strategic Context and Reasoning Engine.
    Grounds high-level strategy in domain knowledge and produces both:
    1. Textual strategic explanations and tactical advice.
    2. Continuous 16-dimensional strategic context embedding vector for DRL.
    """

    def __init__(self, retriever: Optional[SokobanRAGRetriever] = None):
        self.retriever = retriever or SokobanRAGRetriever()
        self._cache: Dict[int, Tuple[np.ndarray, Dict[str, Any]]] = {}

    def generate_strategy_context(self, state: SokobanState) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Synthesizes RAG-retrieved knowledge and state metrics into:
        - 16-dim context vector for DRL policy network.
        - Rich strategic advice dictionary.
        """
        st_hash = hash(state)
        if st_hash in self._cache:
            return self._cache[st_hash]
        # 1. RAG Retrieval
        retrieved = self.retriever.retrieve_for_state(state, top_k=2)

        # 2. Compute state metrics
        bipartite_dist = min_cost_bipartite_heuristic(set(state.boxes), set(state.targets))
        boxes_placed = state.num_boxes_on_target()
        total_targets = len(state.targets)
        completion_ratio = boxes_placed / max(1, total_targets)
        deadlock_mask = get_action_deadlock_mask(state)
        num_valid_actions = int(np.sum(deadlock_mask))

        # 3. Aggregate strategic context vector
        context_vec = np.zeros(16, dtype=np.float32)
        total_sim = 0.0

        for doc, sim in retrieved:
            context_vec += np.array(doc.context_vector, dtype=np.float32) * sim
            total_sim += sim

        if total_sim > 1e-6:
            context_vec /= total_sim

        # Inject real-time state features into context vector slots
        context_vec[0] = float(completion_ratio)                    # Target fill progress
        context_vec[1] = float(min(1.0, bipartite_dist / 20.0))     # Normalized Hungarian cost
        context_vec[2] = float(num_valid_actions / 4.0)             # Degree of action freedom
        context_vec[3] = 1.0 if not state.is_solved() else 0.0      # Active task indicator

        # Normalize vector
        norm = np.linalg.norm(context_vec)
        if norm > 1e-6:
            context_vec = context_vec / norm

        # 4. Generate structured advice & explanation
        primary_doc = retrieved[0][0]
        advice_info = {
            "retrieved_documents": [doc.title for doc, _ in retrieved],
            "primary_rule": primary_doc.strategic_rule,
            "category": primary_doc.category,
            "bipartite_heuristic_dist": round(bipartite_dist, 2),
            "completion_ratio": round(completion_ratio, 2),
            "safe_actions_mask": deadlock_mask.tolist(),
            "explanation": f"Strategy: {primary_doc.strategic_rule} [Context: {primary_doc.title} (Relevance: {round(retrieved[0][1], 3)})]"
        }

        result = (context_vec, advice_info)
        self._cache[st_hash] = result
        return result

    def explain_action(self, state: SokobanState, chosen_action_name: str) -> str:
        """Generate grounded natural language explanation for an action choice."""
        _, advice = self.generate_strategy_context(state)
        return (
            f"Action Selected: [{chosen_action_name}] | "
            f"Grounding: {advice['primary_rule']} | "
            f"Current Heuristic Cost: {advice['bipartite_heuristic_dist']}"
        )
