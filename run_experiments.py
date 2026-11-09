"""
Comprehensive Evaluation Experiment Runner for Sokoban Hybrid AI Agent.
Computes and reports all required metrics across all 7 architectural tiers.
"""

import json
import time
from typing import Dict, Any

from sokoban.reasoning.search_eval import evaluate_search_algorithms
from sokoban.perception.perception_eval import evaluate_perception_models
from sokoban.rag_llm.rag_eval import evaluate_rag_and_genai
from sokoban.drl.drl_eval import train_drl_agent, evaluate_drl_agent
from sokoban.hybrid.comparative_study import run_comparative_ablation_study


def run_full_experiments(train_episodes: int = 40) -> Dict[str, Any]:
    print("=" * 80)
    print("      INTELLIGENT SOKOBAN PUZZLE-SOLVING AGENT: BENCHMARK SUITE")
    print("=" * 80)
    start_total = time.time()

    # 1. Perception Evaluation (CNN & LSTM)
    print("\n>>> [1/5] Evaluating CNN Spatial Perception & LSTM Sequence Modeling...")
    perception_results = evaluate_perception_models(epochs=12)

    # 2. Search & Planning Evaluation (A* & Backtracking)
    print(">>> [2/5] Evaluating Search & Planning (A* and Backtracking)...")
    search_results = evaluate_search_algorithms()

    # 3. Generative AI & RAG Evaluation
    print(">>> [3/5] Evaluating LLM/SLM Generative Reasoning & RAG Knowledge Engine...")
    rag_genai_results = evaluate_rag_and_genai()

    # 4. Deep Reinforcement Learning Evaluation (DQN / PPO)
    print(">>> [4/5] Training and Evaluating Deep Reinforcement Learning Policy...")
    drl_agent, drl_metrics = train_drl_agent(level_name="microban_1", num_episodes=train_episodes)

    # 5. Overall Strict Hybrid Agent vs DRL-only Comparative Ablation
    print(">>> [5/5] Executing Comparative Ablation Study: Performance(Hybrid) - Performance(DRL-only)...")
    comparative_results = run_comparative_ablation_study(train_episodes=train_episodes, eval_episodes_per_level=6)

    total_time = round(time.time() - start_total, 2)

    report = {
        "execution_time_seconds": total_time,
        "metrics_by_tier": {
            "tier_1_deep_reinforcement_learning": {
                "average_reward": drl_metrics["average_reward"],
                "reward_variance": drl_metrics["reward_variance"],
                "episode_length": drl_metrics["episode_length"],
                "success_win_rate": drl_metrics["success_win_rate"],
                "convergence_episodes": drl_metrics["convergence_episodes"],
                "sample_efficiency": drl_metrics["sample_efficiency"]
            },
            "tier_2_search_and_planning": {
                "astar_avg_path_length": search_results["astar_avg_path_length"],
                "astar_avg_nodes_expanded": search_results["astar_avg_nodes_expanded"],
                "astar_avg_search_time_ms": search_results["astar_avg_time_ms"],
                "backtracking_avg_path_length": search_results["backtracking_avg_path_length"],
                "backtracking_avg_nodes_expanded": search_results["backtracking_avg_nodes_expanded"],
                "backtracking_avg_search_time_ms": search_results["backtracking_avg_time_ms"],
                "average_optimality_gap": search_results["average_optimality_gap"]
            },
            "tier_3_cnn_perception": perception_results["cnn_perception"],
            "tier_4_rnn_lstm_sequence_modeling": perception_results["lstm_sequence_modeling"],
            "tier_5_generative_ai": rag_genai_results["generative_ai"],
            "tier_6_rag_knowledge_retrieval": rag_genai_results["rag"],
            "tier_7_overall_hybrid_agent": comparative_results["hybrid_vs_drl_only"]
        },
        "model_ablation_comparison": comparative_results["model_performance_summary"],
        "level_breakdown": comparative_results["level_breakdown"]
    }

    # Save to JSON file
    output_path = "experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("                     EVALUATION REPORT SUMMARY")
    print("=" * 80)
    print(f"Total Execution Time: {total_time}s | Results saved to: {output_path}\n")

    print("--- 1. DEEP REINFORCEMENT LEARNING ---")
    for k, v in report["metrics_by_tier"]["tier_1_deep_reinforcement_learning"].items():
        print(f"  • {k.replace('_', ' ').title()}: {v}")

    print("\n--- 2. SEARCH AND PLANNING ---")
    for k, v in report["metrics_by_tier"]["tier_2_search_and_planning"].items():
        print(f"  • {k.replace('_', ' ').title()}: {v}")

    print("\n--- 3. CNN / PERCEPTION ---")
    for k, v in report["metrics_by_tier"]["tier_3_cnn_perception"].items():
        print(f"  • {k.replace('_', ' ').title()}: {v}")

    print("\n--- 4. RNN / LSTM SEQUENCE MODELING ---")
    for k, v in report["metrics_by_tier"]["tier_4_rnn_lstm_sequence_modeling"].items():
        print(f"  • {k.replace('_', ' ').title()}: {v}")

    print("\n--- 5. GENERATIVE AI STRATEGIC REASONING ---")
    for k, v in report["metrics_by_tier"]["tier_5_generative_ai"].items():
        print(f"  • {k.replace('_', ' ').title()}: {v}")

    print("\n--- 6. RAG KNOWLEDGE RETRIEVAL ---")
    for k, v in report["metrics_by_tier"]["tier_6_rag_knowledge_retrieval"].items():
        print(f"  • {k.replace('_', ' ').title()}: {v}")

    print("\n--- 7. OVERALL HYBRID AGENT COMPARISON (KEY METRIC) ---")
    for k, v in report["metrics_by_tier"]["tier_7_overall_hybrid_agent"].items():
        print(f"  • \033[1;32m{k}\033[0m: {v}")

    print("\n" + "=" * 80)
    return report


if __name__ == "__main__":
    run_full_experiments(train_episodes=40)
