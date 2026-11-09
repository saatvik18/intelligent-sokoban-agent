"""
Interactive Terminal Dashboard and Step-by-Step Solver Visualizer.
"""

import time
from typing import Optional
from sokoban.core.grid import SokobanState, ACTION_NAMES
from sokoban.core.environment import SokobanEnv
from sokoban.core.renderer import render_rich_terminal
from sokoban.core.levels import get_level
from sokoban.hybrid.pipeline import HybridSokobanAgent
from sokoban.reasoning.deadlock import get_action_deadlock_mask


def run_step_by_step_solver(
    level_name: str = "microban_1",
    delay_sec: float = 0.5,
    agent: Optional[HybridSokobanAgent] = None
):
    """Visual step-by-step solver with live pipeline insights."""
    if agent is None:
        agent = HybridSokobanAgent(use_deadlock_mask=True, use_rag_strategy=True)

    env = SokobanEnv(level_name=level_name, max_steps=60)
    agent.reset_trajectory()
    obs, info = env.reset()
    state = info["state"]

    print("\n" + "=" * 65)
    print(f"   HYBRID SOKOBAN AGENT: SOLVING LEVEL '{level_name.upper()}'")
    print("=" * 65)
    print(render_rich_terminal(state, status_text="INITIAL STATE"))
    time.sleep(delay_sec)

    step = 0
    while step < env.max_steps:
        step += 1
        action, meta = agent.select_hybrid_action(state, obs_tensor=obs, deterministic=True)
        act_name = meta["action_name"]
        mask = meta["deadlock_mask"]
        rag_advice = meta["rag_advice"]
        h_cost = meta["heuristic_bipartite_cost"]

        print("\n" + "-" * 65)
        print(f"STEP {step:02d} | Action Selected: \033[1;32m{act_name}\033[0m")
        print(f"Reasoning Deadlock Mask: [UP={mask[0]}, DOWN={mask[1]}, LEFT={mask[2]}, RIGHT={mask[3]}]")
        print(f"A* Bipartite Heuristic Cost: {h_cost}")
        print(f"RAG Tactical Context: \033[1;33m{rag_advice['primary_rule']}\033[0m")
        print(f"Retrieved Document: {rag_advice['retrieved_documents'][0] if rag_advice['retrieved_documents'] else 'N/A'}")
        print("-" * 65)

        obs, reward, terminated, truncated, next_info = env.step(action)
        state = next_info["state"]

        print(render_rich_terminal(state, status_text=f"STEP {step} (Reward: {reward:+.2f})"))
        time.sleep(delay_sec)

        if terminated:
            if next_info["is_solved"]:
                print("\n\033[1;32m[SUCCESS] PUZZLE SOLVED SUCCESSFULLY BY HYBRID AGENT! [SUCCESS]\033[0m")
            else:
                print("\n\033[1;31mGame terminated.\033[0m")
            break

        if truncated:
            print("\n\033[1;33mMax step limit reached.\033[0m")
            break


def run_interactive_dashboard(level_name: str = "microban_1"):
    """Interactive player mode with live AI advisor."""
    env = SokobanEnv(level_name=level_name)
    agent = HybridSokobanAgent(use_deadlock_mask=True, use_rag_strategy=True)
    obs, info = env.reset()
    state = info["state"]

    print("\n" + "=" * 65)
    print("        INTERACTIVE SOKOBAN DASHBOARD WITH AI CO-PILOT")
    print("=" * 65)
    print("Controls: 'w' (Up), 's' (Down), 'a' (Left), 'd' (Right), 'q' (Quit), 'h' (AI Hint)")

    key_map = {'w': 0, 's': 1, 'a': 2, 'd': 3}

    while True:
        print("\n" + render_rich_terminal(state))
        mask = get_action_deadlock_mask(state)
        _, meta = agent.select_hybrid_action(state, obs_tensor=obs)

        print(f"\nAI Co-Pilot Advice: {meta['rag_advice']['primary_rule']}")
        print(f"AI Recommended Action: {meta['action_name']} | Bipartite Cost: {meta['heuristic_bipartite_cost']}")
        print(f"Safe Moves: [UP: {mask[0]}, DOWN: {mask[1]}, LEFT: {mask[2]}, RIGHT: {mask[3]}]")

        user_input = input("\nEnter move (w/a/s/d/h/q): ").strip().lower()
        if user_input == 'q':
            print("Exiting...")
            break
        elif user_input == 'h':
            print(f"\n>>> AI RECOMMENDS: {meta['action_name']} (Reason: {meta['rag_advice']['explanation']})")
            continue
        elif user_input in key_map:
            act = key_map[user_input]
            obs, reward, terminated, truncated, next_info = env.step(act)
            state = next_info["state"]

            if terminated and next_info["is_solved"]:
                print(render_rich_terminal(state, status_text="SOLVED!"))
                print("\n[SUCCESS] Level Complete! Congratulations!")
                break
        else:
            print("Invalid key! Use w, a, s, d, h, or q.")
