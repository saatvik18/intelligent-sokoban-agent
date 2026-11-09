"""
Semantic RAG Retriever for Sokoban Knowledge Base using TF-IDF and Cosine Vector Space.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sokoban.rag_llm.knowledge_base import SOKOBAN_TACTICAL_KNOWLEDGE, KnowledgeDocument
from sokoban.core.grid import SokobanState
from sokoban.reasoning.deadlock import DeadlockDetector


class SokobanRAGRetriever:
    """Semantic vector retriever over Sokoban strategic documents."""

    def __init__(self, documents: Optional[List[KnowledgeDocument]] = None):
        self.documents = documents or SOKOBAN_TACTICAL_KNOWLEDGE
        self.corpus = [
            f"{doc.title} {doc.category} {' '.join(doc.keywords)} {doc.content} {doc.strategic_rule}"
            for doc in self.documents
        ]
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=512)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

        self.query_cache: Dict[str, List[Tuple[KnowledgeDocument, float]]] = {}

    def retrieve(self, query: str, top_k: int = 2, threshold: float = 0.05) -> List[Tuple[KnowledgeDocument, float]]:
        """Retrieve top-k most relevant knowledge documents given a natural language query."""
        if query in self.query_cache:
            return self.query_cache[query]

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        ranked_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in ranked_indices[:top_k]:
            sim = float(similarities[idx])
            if sim >= threshold:
                results.append((self.documents[idx], sim))

        if not results:  # Fallback to top-1 if below threshold
            results.append((self.documents[ranked_indices[0]], float(similarities[ranked_indices[0]])))

        self.query_cache[query] = results
        return results

    def retrieve_for_state(self, state: SokobanState, top_k: int = 2) -> List[Tuple[KnowledgeDocument, float]]:
        """Construct semantic query from board state characteristics and retrieve tactical documents."""
        detector = DeadlockDetector(set(state.walls), set(state.targets), state.height, state.width)
        boxes = set(state.boxes)
        targets = set(state.targets)

        query_terms = []
        # Check corner proximity
        for b in boxes:
            if b not in targets:
                r, c = b
                adj_walls = sum([(r-1, c) in detector.walls, (r+1, c) in detector.walls,
                                 (r, c-1) in detector.walls, (r, c+1) in detector.walls])
                if adj_walls >= 2:
                    query_terms.append("corner deadlock risk wall intersection avoid freeze")
                elif adj_walls == 1:
                    query_terms.append("wall line corridor box packing order")

        # Check box clusters
        if len(boxes) > 1:
            for b1 in boxes:
                for b2 in boxes:
                    if b1 != b2 and abs(b1[0] - b2[0]) + abs(b1[1] - b2[1]) == 1:
                        query_terms.append("2x2 square cluster adjacent boxes separation clearance")

        # Reachability & target filling
        query_terms.append("target ordering subgoal bipartite hungarian matching reachability")

        query = " ".join(query_terms)
        return self.retrieve(query, top_k=top_k)
