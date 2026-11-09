"""
Sokoban Tactical & Strategic Knowledge Base for RAG.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class KnowledgeDocument:
    doc_id: str
    category: str
    title: str
    content: str
    keywords: List[str]
    strategic_rule: str
    context_vector: List[float]  # 16-dimensional strategic embedding


SOKOBAN_TACTICAL_KNOWLEDGE: List[KnowledgeDocument] = [
    KnowledgeDocument(
        doc_id="DOC_CORNER_DEADLOCK",
        category="deadlock_avoidance",
        title="Corner Deadlock Prevention and Wall Intersection Traps",
        content="""A corner deadlock occurs when a box is pushed into a cell bounded by two orthogonal walls or static obstacles without a target. Once a box enters an un-targeted corner, it can never be pulled or pushed out in standard Sokoban rules. Always avoid pushing any unplaced box towards non-target corners. Check adjacent tiles for wall intersections before executing a push action.""",
        keywords=["corner", "deadlock", "orthogonal", "wall intersection", "trap", "unplaced box", "cannot move"],
        strategic_rule="Never push a box into an orthogonal wall intersection unless that coordinate is an explicitly marked goal target.",
        context_vector=[1.0, 0.0, 0.0, 0.2, 0.8, 0.1, 0.0, 0.0, 0.9, 0.0, 0.1, 0.0, 0.0, 0.3, 0.0, 0.5]
    ),
    KnowledgeDocument(
        doc_id="DOC_LINE_CORRIDOR",
        category="corridor_management",
        title="Wall Line Deadlocks and Corridor Box Packing Order",
        content="""A line deadlock occurs when a box is pushed against a straight wall segment where no targets exist, or when multiple boxes are lined up against a wall with fewer targets than boxes. Furthermore, in narrow one-tile corridors, boxes must be pushed in strict order: always fill the deepest goal first to avoid blocking subsequent box arrivals.""",
        keywords=["line", "wall", "corridor", "straight", "narrow", "packing", "deepest target", "order"],
        strategic_rule="In corridors and along outer walls, pack targets from deepest/furthest coordinate outward to preserve passage access.",
        context_vector=[0.2, 1.0, 0.1, 0.7, 0.3, 0.0, 0.4, 0.1, 0.8, 0.0, 0.0, 0.5, 0.2, 0.1, 0.0, 0.6]
    ),
    KnowledgeDocument(
        doc_id="DOC_2X2_SQUARE",
        category="deadlock_avoidance",
        title="2x2 Box Cluster and Structural Immobilization",
        content="""A 2x2 square deadlock occurs when four blocks (consisting of boxes and walls, with at least one box) form a contiguous 2x2 grid where none can be pushed outward. Never push two boxes adjacent to each other against a flat wall or form a box quad. Maintain minimum separation clearance between parallel boxes.""",
        keywords=["2x2", "square", "cluster", "quad", "adjacent", "immobilization", "group"],
        strategic_rule="Avoid forming 2x2 clusters of boxes and walls; preserve at least one cell of maneuvering space between adjacent boxes.",
        context_vector=[0.9, 0.3, 1.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.7, 0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.8]
    ),
    KnowledgeDocument(
        doc_id="DOC_TARGET_ORDER",
        category="subgoal_planning",
        title="Optimal Sub-Goal Target Ordering and Fill Priority",
        content="""Target ordering determines whether a puzzle is solvable or creates self-inflicted deadlocks. Target cells located in tight alcoves, corners, or single-entrance rooms must be filled first. If an outer target is filled early, the box on it becomes an obstacle that traps inner targets permanently.""",
        keywords=["subgoal", "target order", "priority", "alcove", "furthest", "self-inflicted", "fill sequence"],
        strategic_rule="Prioritize filling inner, alcove, and corner targets before filling outer transit corridor targets.",
        context_vector=[0.1, 0.6, 0.2, 1.0, 0.4, 0.3, 0.5, 0.2, 0.2, 0.7, 0.3, 0.4, 0.1, 0.8, 0.2, 0.3]
    ),
    KnowledgeDocument(
        doc_id="DOC_PLAYER_MOBILITY",
        category="reachability",
        title="Preserving Player Mobility and Flood-Fill Reachability",
        content="""A box push that cuts off the player's access to the other side of the room will freeze the game if future pushes require approaching the box from the opposite direction. Always evaluate player flood-fill reachability before finalizing a box push.""",
        keywords=["reachability", "player mobility", "flood fill", "trapped player", "opposite direction", "connectivity"],
        strategic_rule="Ensure player maintains dual-sided reachability around active boxes to avoid self-trapping.",
        context_vector=[0.3, 0.4, 0.1, 0.3, 1.0, 0.8, 0.6, 0.1, 0.5, 0.2, 0.4, 0.1, 0.7, 0.2, 0.3, 0.1]
    ),
    KnowledgeDocument(
        doc_id="DOC_BIPARTITE_MATCH",
        category="heuristic_search",
        title="Bipartite Matching and Hungarian Distance Optimization",
        content="""The global lower bound on moves is given by the minimum-cost bipartite matching between the current box positions and goal targets. When choosing which box to push, select the box that reduces the total Hungarian matching distance without elevating deadlock hazard.""",
        keywords=["bipartite", "hungarian", "matching", "distance", "lower bound", "heuristic", "global cost"],
        strategic_rule="Select pushes that minimize the Hungarian bipartite matching sum to targets while preserving path clearances.",
        context_vector=[0.0, 0.1, 0.0, 0.5, 0.2, 0.2, 1.0, 0.8, 0.1, 0.9, 0.5, 0.6, 0.3, 0.4, 0.8, 0.2]
    ),
    KnowledgeDocument(
        doc_id="DOC_SACRIFICE_OPENING",
        category="tactical_maneuvers",
        title="Sacrificial Repositioning and Tactical Openings",
        content="""In congested rooms, it is often necessary to push a box temporarily away from its target or off a target to clear a corridor for another box. These sacrificial moves temporarily increase heuristic distance but unlock essential degrees of freedom.""",
        keywords=["sacrifice", "temporary", "opening", "repositioning", "congested", "degrees of freedom", "clearance"],
        strategic_rule="Allow temporary backward moves when necessary to unlock congested corridors for higher-priority boxes.",
        context_vector=[0.1, 0.2, 0.1, 0.3, 0.5, 0.4, 0.4, 1.0, 0.2, 0.5, 0.8, 0.3, 0.4, 0.6, 0.2, 0.7]
    )
]
