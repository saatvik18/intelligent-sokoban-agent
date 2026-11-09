"""
Search and Planning evaluation harness for A* and Backtracking across benchmarks.
"""

from typing import Dict, Any, List
import numpy as np

from sokoban.core.grid import SokobanState
from sokoban.core.levels import get_benchmark_suite
from sokoban.reasoning.astar import astar_solve
from sokoban.reasoning.backtracking import backtracking_solve


def evaluate_search_algorithms(levels: Dict[str, SokobanState] = None) -> Dict[str, Any]:
    """
    Evaluates A* and Backtracking against standard benchmarks.
    Metrics evaluated:
      - Solution/path length
      - Number of expanded nodes
      - Search time (ms)
      - Optimality gap
    """
    if levels is None:
        levels = get_benchmark_suite()

    results: Dict[str, Dict[str, Any]] = {
        "astar": {"solved_count": 0, "path_lengths": [], "nodes_expanded": [], "search_times_ms": []},
        "backtracking": {"solved_count": 0, "path_lengths": [], "nodes_expanded": [], "search_times_ms": []},
        "per_level": {}
    }

    total_levels = len(levels)

    for name, state in levels.items():
        # Run A*
        a_path, a_nodes, a_time, a_succ = astar_solve(state, max_expansions=3000)
        # Run Backtracking
        b_path, b_nodes, b_time, b_succ = backtracking_solve(state, max_depth=40, max_expansions=3000)

        a_len = len(a_path) if a_succ else None
        b_len = len(b_path) if b_succ else None

        # Optimality gap of backtracking relative to A* (A* with admissible heuristic is optimal)
        opt_gap = 0.0
        if a_succ and b_succ and a_len and a_len > 0:
            opt_gap = max(0.0, (b_len - a_len) / a_len)

        results["per_level"][name] = {
            "astar": {"solved": a_succ, "path_len": a_len, "nodes": a_nodes, "time_ms": round(a_time, 2)},
            "backtracking": {"solved": b_succ, "path_len": b_len, "nodes": b_nodes, "time_ms": round(b_time, 2)},
            "optimality_gap": round(opt_gap, 4)
        }

        if a_succ:
            results["astar"]["solved_count"] += 1
            results["astar"]["path_lengths"].append(a_len)
            results["astar"]["nodes_expanded"].append(a_nodes)
            results["astar"]["search_times_ms"].append(a_time)

        if b_succ:
            results["backtracking"]["solved_count"] += 1
            results["backtracking"]["path_lengths"].append(b_len)
            results["backtracking"]["nodes_expanded"].append(b_nodes)
            results["backtracking"]["search_times_ms"].append(b_time)

    # Compute aggregate summary
    summary = {
        "total_benchmark_levels": total_levels,
        "astar_solve_rate": round(results["astar"]["solved_count"] / total_levels, 4),
        "astar_avg_path_length": round(float(np.mean(results["astar"]["path_lengths"])), 2) if results["astar"]["path_lengths"] else 0.0,
        "astar_avg_nodes_expanded": round(float(np.mean(results["astar"]["nodes_expanded"])), 2) if results["astar"]["nodes_expanded"] else 0.0,
        "astar_avg_time_ms": round(float(np.mean(results["astar"]["search_times_ms"])), 2) if results["astar"]["search_times_ms"] else 0.0,
        "backtracking_solve_rate": round(results["backtracking"]["solved_count"] / total_levels, 4),
        "backtracking_avg_path_length": round(float(np.mean(results["backtracking"]["path_lengths"])), 2) if results["backtracking"]["path_lengths"] else 0.0,
        "backtracking_avg_nodes_expanded": round(float(np.mean(results["backtracking"]["nodes_expanded"])), 2) if results["backtracking"]["nodes_expanded"] else 0.0,
        "backtracking_avg_time_ms": round(float(np.mean(results["backtracking"]["search_times_ms"])), 2) if results["backtracking"]["search_times_ms"] else 0.0,
        "average_optimality_gap": round(float(np.mean([v["optimality_gap"] for v in results["per_level"].values()])), 4),
        "detailed_results": results["per_level"]
    }

    return summary
