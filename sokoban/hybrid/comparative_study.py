"""
Comparative Ablation Study and Performance Benchmark:
Evaluates Performance(Hybrid) - Performance(DRL-only) across benchmark levels.
"""

from typing import Dict, Any, List
import numpy as np

from sokoban.core.grid import SokobanState
from sokoban.core.environment import SokobanEnv
from sokoban.core.levels import get_benchmark_suite
from sokoban.reasoning.astar import astar_solve
from sokoban.drl.drl_eval import train_drl_agent
from sokoban.hybrid.pipeline import HybridSokobanAgent


def run_comparative_ablation_study(
    benchmark_names: List[str] = None,
    train_episodes: int = 50,
    eval_episodes_per_level: int = 4
) -> Dict[str, Any]:
    """
    Executes comprehensive comparative ablation study comparing:
    1. DRL-only (Baseline, no deadlock mask, no RAG)
    2. Search-only (Pure A* Search)
    3. Search + DRL (Deadlock-masked DRL)
    4. Full Hybrid Agent (Perception + Search + RAG + Masked DRL)

    Computes: Performance_Hybrid - Performance_DRL-only
    """
    if benchmark_names is None:
        benchmark_names = ["microban_1", "microban_2", "microban_3", "microban_4"]

    benchmarks = get_benchmark_suite()
    selected_levels = {k: benchmarks[k] for k in benchmark_names if k in benchmarks}

    # 1. Pre-train agents under different configurations
    print("[1/4] Training DRL-only Baseline Agent...")
    drl_only_agent, _ = train_drl_agent(
        level_name="microban_1",
        num_episodes=train_episodes,
        use_deadlock_mask=False,
        use_rag_strategy=False
    )

    print("[2/4] Training Hybrid Agent (with Search & RAG Context)...")
    hybrid_drl_agent, _ = train_drl_agent(
        level_name="microban_1",
        num_episodes=train_episodes,
        use_deadlock_mask=True,
        use_rag_strategy=True
    )

    # 2. Instantiate Agent instances for Ablation
    drl_only_pipeline = HybridSokobanAgent(
        drl_agent=drl_only_agent,
        use_deadlock_mask=False,
        use_search_fallback=False,
        use_rag_strategy=False
    )

    masked_drl_pipeline = HybridSokobanAgent(
        drl_agent=hybrid_drl_agent,
        use_deadlock_mask=True,
        use_search_fallback=False,
        use_rag_strategy=False
    )

    full_hybrid_pipeline = HybridSokobanAgent(
        drl_agent=hybrid_drl_agent,
        use_deadlock_mask=True,
        use_search_fallback=True,
        use_rag_strategy=True
    )

    results = {
        "drl_only": {"wins": [], "rewards": [], "lengths": []},
        "search_only": {"wins": [], "path_lengths": [], "nodes": []},
        "search_plus_drl": {"wins": [], "rewards": [], "lengths": []},
        "full_hybrid": {"wins": [], "rewards": [], "lengths": []},
        "per_level_comparison": {}
    }

    print("[3/4] Evaluating across benchmark suites...")
    for lvl_name, state in selected_levels.items():
        level_env = SokobanEnv(level_name=lvl_name, max_steps=80)

        # 1. Search-only Evaluation
        path, nodes, _, s_succ = astar_solve(state, max_expansions=5000)
        results["search_only"]["wins"].append(1.0 if s_succ else 0.0)
        if s_succ:
            results["search_only"]["path_lengths"].append(len(path))
            results["search_only"]["nodes"].append(nodes)

        # 2. DRL-only Evaluation
        drl_wins, drl_rews, drl_lens = [], [], []
        for _ in range(eval_episodes_per_level):
            s, p, r, _ = drl_only_pipeline.solve(level_env, max_steps=60)
            drl_wins.append(1.0 if s else 0.0)
            drl_rews.append(r)
            drl_lens.append(len(p))

        # 3. Search + DRL Evaluation
        mask_wins, mask_rews, mask_lens = [], [], []
        for _ in range(eval_episodes_per_level):
            s, p, r, _ = masked_drl_pipeline.solve(level_env, max_steps=60)
            mask_wins.append(1.0 if s else 0.0)
            mask_rews.append(r)
            mask_lens.append(len(p))

        # 4. Full Hybrid Evaluation
        hyb_wins, hyb_rews, hyb_lens = [], [], []
        for _ in range(eval_episodes_per_level):
            s, p, r, _ = full_hybrid_pipeline.solve(level_env, max_steps=60)
            hyb_wins.append(1.0 if s else 0.0)
            hyb_rews.append(r)
            hyb_lens.append(len(p))

        results["drl_only"]["wins"].extend(drl_wins)
        results["drl_only"]["rewards"].extend(drl_rews)
        results["drl_only"]["lengths"].extend(drl_lens)

        results["search_plus_drl"]["wins"].extend(mask_wins)
        results["search_plus_drl"]["rewards"].extend(mask_rews)
        results["search_plus_drl"]["lengths"].extend(mask_lens)

        results["full_hybrid"]["wins"].extend(hyb_wins)
        results["full_hybrid"]["rewards"].extend(hyb_rews)
        results["full_hybrid"]["lengths"].extend(hyb_lens)

        results["per_level_comparison"][lvl_name] = {
            "drl_only_win_rate": round(float(np.mean(drl_wins)), 2),
            "drl_only_avg_reward": round(float(np.mean(drl_rews)), 2),
            "search_only_solved": s_succ,
            "search_plus_drl_win_rate": round(float(np.mean(mask_wins)), 2),
            "full_hybrid_win_rate": round(float(np.mean(hyb_wins)), 2),
            "full_hybrid_avg_reward": round(float(np.mean(hyb_rews)), 2)
        }

    # Aggregate summaries
    drl_win_rate = float(np.mean(results["drl_only"]["wins"]))
    drl_avg_reward = float(np.mean(results["drl_only"]["rewards"]))
    drl_avg_len = float(np.mean(results["drl_only"]["lengths"]))

    hyb_win_rate = float(np.mean(results["full_hybrid"]["wins"]))
    hyb_avg_reward = float(np.mean(results["full_hybrid"]["rewards"]))
    hyb_avg_len = float(np.mean(results["full_hybrid"]["lengths"]))

    mask_win_rate = float(np.mean(results["search_plus_drl"]["wins"]))
    search_solve_rate = float(np.mean(results["search_only"]["wins"]))

    # Core required metric: Performance_Hybrid - Performance_DRL-only
    delta_win_rate = hyb_win_rate - drl_win_rate
    delta_reward = hyb_avg_reward - drl_avg_reward
    delta_efficiency = drl_avg_len - hyb_avg_len  # Positive means Hybrid is faster/shorter

    summary = {
        "hybrid_vs_drl_only": {
            "delta_win_rate (Performance_Hybrid - Performance_DRL-only)": round(delta_win_rate, 4),
            "delta_average_reward": round(delta_reward, 4),
            "delta_step_efficiency": round(delta_efficiency, 2),
        },
        "model_performance_summary": {
            "drl_only_baseline": {
                "win_rate": round(drl_win_rate, 4),
                "avg_reward": round(drl_avg_reward, 2),
                "avg_episode_length": round(drl_avg_len, 2)
            },
            "search_only_astar": {
                "solve_rate": round(search_solve_rate, 4),
                "avg_path_length": round(float(np.mean(results["search_only"]["path_lengths"])), 2) if results["search_only"]["path_lengths"] else 0.0
            },
            "search_plus_drl": {
                "win_rate": round(mask_win_rate, 4),
                "avg_reward": round(float(np.mean(results["search_plus_drl"]["rewards"])), 2)
            },
            "full_hybrid_agent": {
                "win_rate": round(hyb_win_rate, 4),
                "avg_reward": round(hyb_avg_reward, 2),
                "avg_episode_length": round(hyb_avg_len, 2)
            }
        },
        "level_breakdown": results["per_level_comparison"]
    }

    return summary
