"""
Lightweight REST API and Static File Server for Sokoban Hybrid AI Web UI.
Uses Python Standard Library http.server (Zero extra pip dependencies).
"""

import json
import os
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Ensure root workspace is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sokoban.core.grid import (
    SokobanState, Action, parse_level_string, ACTIONS, ACTION_NAMES
)
from sokoban.core.levels import BENCHMARK_LEVELS, get_level, generate_random_level
from sokoban.reasoning.deadlock import DeadlockDetector, get_action_deadlock_mask
from sokoban.reasoning.astar import AStarSolver, min_cost_bipartite_heuristic
from sokoban.rag_llm.strategy_advisor import SokobanStrategyAdvisor
from sokoban.hybrid.pipeline import HybridSokobanAgent

STATIC_DIR = os.path.join(BASE_DIR, "web_app")
advisor = SokobanStrategyAdvisor()
hybrid_agent = HybridSokobanAgent(use_deadlock_mask=True, use_rag_strategy=True)


def state_to_dict(state: SokobanState) -> dict:
    detector = DeadlockDetector(set(state.walls), set(state.targets), state.height, state.width)
    deadlock_cells = [list(pos) for pos in detector.static_deadlocks]
    is_dead, reason = detector.is_deadlock(state)
    mask = detector.get_valid_action_mask(state)
    h_cost = min_cost_bipartite_heuristic(set(state.boxes), set(state.targets))

    return {
        "height": state.height,
        "width": state.width,
        "walls": [list(pos) for pos in state.walls],
        "boxes": [list(pos) for pos in state.boxes],
        "targets": [list(pos) for pos in state.targets],
        "player": list(state.player),
        "is_solved": state.is_solved(),
        "num_boxes_on_target": state.num_boxes_on_target(),
        "total_targets": len(state.targets),
        "is_deadlocked": is_dead,
        "deadlock_reason": reason,
        "static_deadlock_cells": deadlock_cells,
        "valid_action_mask": mask.tolist(),
        "bipartite_heuristic_cost": round(float(h_cost), 2),
        "ascii": state.to_ascii()
    }


def dict_to_state(data: dict) -> SokobanState:
    walls = frozenset(tuple(pos) for pos in data["walls"])
    boxes = frozenset(tuple(pos) for pos in data["boxes"])
    targets = frozenset(tuple(pos) for pos in data["targets"])
    player = tuple(data["player"])
    height = data["height"]
    width = data["width"]
    return SokobanState(walls, boxes, targets, player, height, width)


class SokobanWebHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/levels":
            self.send_json_response({
                "levels": list(BENCHMARK_LEVELS.keys())
            })
        elif parsed.path == "/api/level":
            qs = parse_qs(parsed.query)
            level_name = qs.get("name", ["microban_1"])[0]
            try:
                state = get_level(level_name)
                ctx_vec, advice = advisor.generate_strategy_context(state)
                payload = state_to_dict(state)
                payload["level_name"] = level_name
                payload["rag_advice"] = advice
                self.send_json_response(payload)
            except Exception as e:
                self.send_error_response(str(e))
        else:
            # Serve static files (index.html, style.css, app.js)
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        data = json.loads(body) if body else {}

        if parsed.path == "/api/step":
            try:
                state = dict_to_state(data["state"])
                action = Action(int(data["action"]))
                next_state, is_valid, is_push = state.apply_action(action)

                if is_valid and next_state is not None:
                    ctx_vec, advice = advisor.generate_strategy_context(next_state)
                    resp = state_to_dict(next_state)
                    resp["is_valid"] = True
                    resp["is_push"] = is_push
                    resp["rag_advice"] = advice
                else:
                    resp = state_to_dict(state)
                    resp["is_valid"] = False
                    resp["is_push"] = False
                    _, advice = advisor.generate_strategy_context(state)
                    resp["rag_advice"] = advice

                self.send_json_response(resp)
            except Exception as e:
                self.send_error_response(str(e))

        elif parsed.path == "/api/advice":
            try:
                state = dict_to_state(data["state"])
                action, meta = hybrid_agent.select_hybrid_action(state)
                _, advice = advisor.generate_strategy_context(state)
                self.send_json_response({
                    "recommended_action": int(action),
                    "action_name": ACTION_NAMES[int(action)],
                    "rag_advice": advice,
                    "metadata": meta
                })
            except Exception as e:
                self.send_error_response(str(e))

        elif parsed.path == "/api/solve":
            try:
                state = dict_to_state(data["state"])
                path, nodes, elapsed_ms, succ = AStarSolver(state, max_expansions=5000).solve()

                steps_trace = []
                curr = state
                if succ and path:
                    for act in path:
                        nxt, valid, is_push = curr.apply_action(act)
                        _, adv = advisor.generate_strategy_context(curr)
                        steps_trace.append({
                            "action": int(act),
                            "action_name": ACTION_NAMES[int(act)],
                            "is_push": is_push,
                            "explanation": adv["explanation"],
                            "rule": adv["primary_rule"],
                            "doc_title": adv["retrieved_documents"][0] if adv["retrieved_documents"] else "Tactics",
                            "state": state_to_dict(nxt) if valid else state_to_dict(curr)
                        })
                        curr = nxt

                self.send_json_response({
                    "success": succ,
                    "solution_length": len(path) if succ else 0,
                    "expanded_nodes": nodes,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "steps": steps_trace
                })
            except Exception as e:
                self.send_error_response(str(e))

        elif parsed.path == "/api/generate":
            try:
                num_boxes = data.get("num_boxes", 2)
                seed = data.get("seed", None)
                state = generate_random_level(height=7, width=7, num_boxes=num_boxes, seed=seed)
                payload = state_to_dict(state)
                payload["level_name"] = "Procedural Level"
                _, advice = advisor.generate_strategy_context(state)
                payload["rag_advice"] = advice
                self.send_json_response(payload)
            except Exception as e:
                self.send_error_response(str(e))

        else:
            self.send_error_response("Endpoint not found", status=404)

    def send_json_response(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_error_response(self, message: str, status: int = 400):
        self.send_json_response({"error": message}, status=status)


def run_server(port: int = 8000):
    server = ThreadingHTTPServer(("0.0.0.0", port), SokobanWebHandler)
    print("=" * 65)
    print(f"   SOKOBAN HYBRID AI INTERACTIVE WEB UI SERVER RUNNING")
    print(f"   URL: http://localhost:{port}")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
