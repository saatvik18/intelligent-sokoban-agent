"""
Perception Evaluation Suite: Computes CNN and LSTM metrics across Sokoban state datasets.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, root_mean_squared_error

from sokoban.core.grid import SokobanState, Action
from sokoban.core.levels import get_benchmark_suite, generate_random_level
from sokoban.reasoning.deadlock import is_deadlocked_state
from sokoban.reasoning.astar import astar_solve
from sokoban.perception.cnn_encoder import SokobanCNNEncoder
from sokoban.perception.lstm_model import SokobanActionLSTM


def generate_perception_dataset(num_samples: int = 120) -> List[Dict[str, Any]]:
    """Generates synthetic trajectories and labeled states for perception models."""
    dataset = []
    benchmarks = get_benchmark_suite()

    # 1. Collect from benchmarks
    for name, state in benchmarks.items():
        is_dead, _ = is_deadlocked_state(state)
        label_solvable = 0 if is_dead else (1 if state.is_solved() else 1)
        path, _, _, succ = astar_solve(state, max_expansions=3000)

        curr = state
        seq_states = [curr.to_feature_tensor(fixed_size=(8, 8))]
        seq_actions = []

        if succ and path:
            for step_idx, act in enumerate(path):
                next_st, valid, _ = curr.apply_action(act)
                if valid and next_st:
                    curr = next_st
                    seq_states.append(curr.to_feature_tensor(fixed_size=(8, 8)))
                    seq_actions.append(int(act))

                    # Behavior classification: 0 = exploration, 1 = box push, 2 = goal reach
                    boxes_on_t = curr.num_boxes_on_target()
                    behavior = 2 if curr.is_solved() else (1 if boxes_on_t > 0 else 0)
                    rem_dist = float(len(path) - step_idx - 1)

                    dataset.append({
                        "tensor": curr.to_feature_tensor(fixed_size=(8, 8)),
                        "solvable_label": 1 if succ else 0,
                        "behavior_label": behavior,
                        "remaining_dist": rem_dist,
                        "optimal_next_action": int(path[step_idx + 1]) if step_idx + 1 < len(path) else int(act),
                        "seq_states": seq_states[-5:],  # Last up to 5 states
                        "seq_actions": seq_actions[-5:]
                    })

    # 2. Add procedural samples
    for i in range(max(10, num_samples - len(dataset))):
        st = generate_random_level(height=7, width=7, num_boxes=1, num_reverse_steps=6, seed=42 + i)
        is_dead, _ = is_deadlocked_state(st)
        tensor = st.to_feature_tensor(fixed_size=(8, 8))
        dataset.append({
            "tensor": tensor,
            "solvable_label": 0 if is_dead else 1,
            "behavior_label": 0,
            "remaining_dist": float(np.random.randint(1, 10)),
            "optimal_next_action": int(np.random.randint(0, 4)),
            "seq_states": [tensor],
            "seq_actions": [int(np.random.randint(0, 4))]
        })

    return dataset


def evaluate_perception_models(epochs: int = 15) -> Dict[str, Any]:
    """
    Trains and evaluates CNN State Encoder and LSTM Sequence Model.
    Returns dictionary with all specified perception metrics.
    """
    dataset = generate_perception_dataset(num_samples=150)
    split_idx = int(0.75 * len(dataset))
    train_data, test_data = dataset[:split_idx], dataset[split_idx:]

    # ------------------ 1. Evaluate CNN Encoder ------------------
    cnn_model = SokobanCNNEncoder(in_channels=6, embedding_dim=128, num_classes=2)
    cnn_optimizer = optim.AdamW(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_cls = nn.CrossEntropyLoss()

    train_tensors = torch.tensor(np.array([d["tensor"] for d in train_data]), dtype=torch.float32)
    train_labels = torch.tensor([d["solvable_label"] for d in train_data], dtype=torch.long)
    test_tensors = torch.tensor(np.array([d["tensor"] for d in test_data]), dtype=torch.float32)
    test_labels = torch.tensor([d["solvable_label"] for d in test_data], dtype=torch.long)

    # Train CNN
    cnn_model.train()
    for _ in range(epochs):
        cnn_optimizer.zero_grad()
        _, logits = cnn_model(train_tensors)
        loss = criterion_cls(logits, train_labels)
        loss.backward()
        cnn_optimizer.step()

    # Evaluate CNN
    cnn_model.eval()
    with torch.no_grad():
        _, test_logits = cnn_model(test_tensors)
        preds = torch.argmax(test_logits, dim=-1).cpu().numpy()
        targets = test_labels.cpu().numpy()

    cnn_acc = accuracy_score(targets, preds)
    cnn_prec = precision_score(targets, preds, zero_division=0)
    cnn_rec = recall_score(targets, preds, zero_division=0)
    cnn_f1 = f1_score(targets, preds, zero_division=0)

    # ------------------ 2. Evaluate LSTM Sequence Model ------------------
    lstm_model = SokobanActionLSTM(input_dim=128 + 4, hidden_dim=64, num_layers=1, num_actions=4, num_classes=3)
    lstm_optimizer = optim.AdamW(lstm_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_reg = nn.MSELoss()

    # Prepare sequence feature vectors using CNN embeddings
    with torch.no_grad():
        train_embeds = cnn_model.extract_features(train_tensors)
        test_embeds = cnn_model.extract_features(test_tensors)

    # Construct simple (B, 1, 132) sequences for evaluation
    def create_lstm_inputs(embeds, data):
        seq_list = []
        for i, emb in enumerate(embeds):
            act = data[i]["seq_actions"][-1] if data[i]["seq_actions"] else 0
            one_hot_act = torch.zeros(4)
            one_hot_act[act] = 1.0
            feat = torch.cat([emb, one_hot_act], dim=0).unsqueeze(0)  # (1, 132)
            seq_list.append(feat)
        return torch.stack(seq_list, dim=0)

    lstm_train_x = create_lstm_inputs(train_embeds, train_data)
    lstm_test_x = create_lstm_inputs(test_embeds, test_data)

    train_dist = torch.tensor([d["remaining_dist"] for d in train_data], dtype=torch.float32)
    test_dist = torch.tensor([d["remaining_dist"] for d in test_data], dtype=torch.float32)

    train_actions = torch.tensor([d["optimal_next_action"] for d in train_data], dtype=torch.long)
    test_actions = torch.tensor([d["optimal_next_action"] for d in test_data], dtype=torch.long)

    train_behaviors = torch.tensor([d["behavior_label"] for d in train_data], dtype=torch.long)
    test_behaviors = torch.tensor([d["behavior_label"] for d in test_data], dtype=torch.long)

    # Train LSTM
    lstm_model.train()
    for _ in range(epochs + 5):
        lstm_optimizer.zero_grad()
        act_logits, dist_preds, beh_logits, _ = lstm_model(lstm_train_x)
        loss_act = criterion_cls(act_logits, train_actions)
        loss_dist = criterion_reg(dist_preds, train_dist)
        loss_beh = criterion_cls(beh_logits, train_behaviors)
        loss = loss_act + 0.1 * loss_dist + loss_beh
        loss.backward()
        lstm_optimizer.step()

    # Evaluate LSTM
    lstm_model.eval()
    with torch.no_grad():
        act_logits, dist_preds, beh_logits, _ = lstm_model(lstm_test_x)
        pred_actions = torch.argmax(act_logits, dim=-1).cpu().numpy()
        pred_beh = torch.argmax(beh_logits, dim=-1).cpu().numpy()
        pred_dists = dist_preds.cpu().numpy()

        true_actions = test_actions.cpu().numpy()
        true_beh = test_behaviors.cpu().numpy()
        true_dists = test_dist.cpu().numpy()

    lstm_mae = float(mean_absolute_error(true_dists, pred_dists))
    lstm_rmse = float(root_mean_squared_error(true_dists, pred_dists))
    lstm_pred_acc = float(accuracy_score(true_actions, pred_actions))
    lstm_beh_acc = float(accuracy_score(true_beh, pred_beh))

    results = {
        "cnn_perception": {
            "accuracy": round(float(cnn_acc), 4),
            "precision": round(float(cnn_prec), 4),
            "recall": round(float(cnn_rec), 4),
            "f1_score": round(float(cnn_f1), 4)
        },
        "lstm_sequence_modeling": {
            "mae": round(lstm_mae, 4),
            "rmse": round(lstm_rmse, 4),
            "prediction_accuracy": round(lstm_pred_acc, 4),
            "behavior_classification_accuracy": round(lstm_beh_acc, 4)
        }
    }

    return results
