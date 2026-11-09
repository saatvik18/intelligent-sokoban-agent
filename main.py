"""
Unified CLI Entry Point for Intelligent Sokoban Puzzle-Solving Agent.
"""

import argparse
import sys

from sokoban.core.levels import BENCHMARK_LEVELS, get_level
from sokoban.reasoning.astar import astar_solve
from sokoban.reasoning.backtracking import backtracking_solve
from sokoban.hybrid.pipeline import HybridSokobanAgent
from sokoban.visualizer.dashboard import run_step_by_step_solver, run_interactive_dashboard
from sokoban.reasoning.search_eval import evaluate_search_algorithms
from run_experiments import run_full_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Sokoban Puzzle-Solving Hybrid AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode solve --level microban_1       # Solve puzzle step-by-step with visualizer & RAG explanations
  python main.py --mode interactive --level microban_2 # Play puzzle with live AI Co-Pilot advisor
  python main.py --mode evaluate --episodes 80         # Run full 7-tier benchmark evaluation
  python main.py --mode search --level microban_3      # Run pure A* and Backtracking planners
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["solve", "interactive", "evaluate", "search", "web", "list_levels"],
        default="solve",
        help="Execution mode"
    )
    parser.add_argument(
        "--level",
        type=str,
        default="microban_1",
        help="Target benchmark level name"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=80,
        help="Number of training episodes for DRL evaluation"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.3,
        help="Step delay in seconds for solver visualization"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for Web UI server"
    )

    args = parser.parse_args()

    if args.mode == "list_levels":
        print("\nAvailable Benchmark Levels:")
        for name in BENCHMARK_LEVELS:
            print(f"  • {name}")
        return

    if args.mode == "web":
        from web_app.server import run_server
        run_server(port=args.port)

    elif args.mode == "solve":
        run_step_by_step_solver(level_name=args.level, delay_sec=args.speed)

    elif args.mode == "interactive":
        run_interactive_dashboard(level_name=args.level)

    elif args.mode == "search":
        state = get_level(args.level)
        print(f"\nEvaluating Search Planners on level: '{args.level}'")
        print(state.to_ascii())

        print("\n>>> Running A* (Hungarian Bipartite Heuristic)...")
        a_path, a_nodes, a_time, a_succ = astar_solve(state)
        print(f"A* Solved: {a_succ} | Path Length: {len(a_path) if a_succ else 'N/A'} | Nodes: {a_nodes} | Time: {a_time:.2f}ms")

        print("\n>>> Running Backtracking Search...")
        b_path, b_nodes, b_time, b_succ = backtracking_solve(state, max_depth=60)
        print(f"Backtracking Solved: {b_succ} | Path Length: {len(b_path) if b_succ else 'N/A'} | Nodes: {b_nodes} | Time: {b_time:.2f}ms")

    elif args.mode == "evaluate":
        run_full_experiments(train_episodes=args.episodes)


if __name__ == "__main__":
    main()
