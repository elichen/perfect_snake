"""Patch an RNN Snake policy toward Hamiltonian-order-preserving shortcuts."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from distill.expert import find_aligned_cycle
from distill.rnn_model import SnakeRNNPolicy, load_rnn_policy_state
from rnn_eval_seeds import eval_seed
from snake_env import SnakeEnv


def _parse_ints(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _parse_floats(value: str) -> list[float]:
    result = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one float")
    return result


def _save_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _make_policy(*, board_size: int, hidden_size: int, device: str, state: dict[str, Any]) -> SnakeRNNPolicy:
    policy = SnakeRNNPolicy(board_size=board_size, n_channels=5, hidden_size=hidden_size).to(device)
    load_rnn_policy_state(policy, state)
    return policy


def _target_for_action(env: SnakeEnv, action: int) -> tuple[int, int]:
    new_dir = (env.direction + {0: -1, 1: 0, 2: 1}[int(action)]) % 4
    dr, dc = env.DIRECTIONS[new_dir]
    hr, hc = env.snake_head
    return hr + dr, hc + dc


def _cycle_ordered(env: SnakeEnv, cycle_index: dict[tuple[int, int], int]) -> bool:
    n_cycle = len(cycle_index)
    head_idx = cycle_index[env.snake_head]
    distances = [(cycle_index[pos] - head_idx) % n_cycle for pos in env.snake]
    return distances[0] == 0 and all(distances[idx] < distances[idx + 1] for idx in range(len(distances) - 1))


def _action_preserves_cycle_order(
    env: SnakeEnv,
    action: int,
    cycle_index: dict[tuple[int, int], int],
) -> tuple[bool, bool]:
    snapshot = env._snapshot_state()
    _, _, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        is_win = info.get("reason") == "win"
        env._restore_state(snapshot)
        return is_win, is_win
    ordered = _cycle_ordered(env, cycle_index)
    env._restore_state(snapshot)
    return ordered, False


def _static_shortest_first_actions(env: SnakeEnv) -> set[int]:
    """Return first relative actions on shortest static paths to food."""
    if env.food_pos is None or env.food_pos[0] < 0:
        return set()
    blocked = set(env.snake[:-1])
    blocked.discard(env.snake_head)
    queue = deque([(env.snake_head, None)])
    seen = {env.snake_head}
    found_distance: int | None = None
    first_actions: set[int] = set()
    depth = {env.snake_head: 0}

    while queue:
        pos, first_action = queue.popleft()
        pos_depth = depth[pos]
        if found_distance is not None and pos_depth >= found_distance:
            continue
        for action in range(3) if first_action is None else (0, 1, 2, 3):
            if first_action is None:
                new_dir = (env.direction + {0: -1, 1: 0, 2: 1}[action]) % 4
                next_first = action
            else:
                new_dir = int(action)
                next_first = first_action
            dr, dc = env.DIRECTIONS[new_dir]
            nr, nc = pos[0] + dr, pos[1] + dc
            next_pos = (nr, nc)
            if not (0 <= nr < env.n and 0 <= nc < env.n):
                continue
            if next_pos in blocked or next_pos in seen:
                continue
            next_depth = pos_depth + 1
            if next_pos == env.food_pos:
                found_distance = next_depth
                first_actions.add(int(next_first))
                continue
            seen.add(next_pos)
            depth[next_pos] = next_depth
            queue.append((next_pos, next_first))
    return first_actions


def _static_food_distance_after_action(env: SnakeEnv, action: int) -> float:
    snapshot = env._snapshot_state()
    _, _, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env._restore_state(snapshot)
        return 0.0 if info.get("reason") == "win" else float("inf")
    blocked = set(env.snake[:-1])
    blocked.discard(env.snake_head)
    queue = deque([(env.snake_head, 0)])
    seen = {env.snake_head}
    while queue:
        pos, distance = queue.popleft()
        if pos == env.food_pos:
            env._restore_state(snapshot)
            return float(distance)
        for dr, dc in env.DIRECTIONS.values():
            next_pos = (pos[0] + dr, pos[1] + dc)
            if not (0 <= next_pos[0] < env.n and 0 <= next_pos[1] < env.n):
                continue
            if next_pos in blocked or next_pos in seen:
                continue
            seen.add(next_pos)
            queue.append((next_pos, distance + 1))
    env._restore_state(snapshot)
    return float("inf")


def _static_path_candidates(
    env: SnakeEnv,
    *,
    max_nodes: int,
    max_candidates: int,
) -> list[list[int]]:
    if env.food_pos is None or env.food_pos[0] < 0:
        return []
    blocked = set(env.snake[:-1])
    blocked.discard(env.snake_head)
    start = (env.snake_head, int(env.direction))
    queue = deque([(start[0], start[1], [])])
    seen = {start}
    candidates: list[list[int]] = []
    nodes = 0

    while queue and nodes < max_nodes and len(candidates) < max_candidates:
        pos, direction, path = queue.popleft()
        nodes += 1
        for action in range(3):
            new_dir = (direction + {0: -1, 1: 0, 2: 1}[action]) % 4
            dr, dc = env.DIRECTIONS[new_dir]
            next_pos = (pos[0] + dr, pos[1] + dc)
            if not (0 <= next_pos[0] < env.n and 0 <= next_pos[1] < env.n):
                continue
            if next_pos in blocked:
                continue
            next_path = [*path, action]
            if next_pos == env.food_pos:
                candidates.append(next_path)
                continue
            key = (next_pos, new_dir)
            if key in seen:
                continue
            seen.add(key)
            queue.append((next_pos, new_dir, next_path))
    return candidates


def _path_preserves_cycle_order(
    env: SnakeEnv,
    path: list[int],
    cycle_index: dict[tuple[int, int], int],
) -> bool:
    snapshot = env._snapshot_state()
    score_before = int(env.score)
    valid = False
    for action in path:
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            valid = info.get("reason") == "win" or int(info.get("score", env.score)) > score_before
            break
        if not _cycle_ordered(env, cycle_index):
            valid = False
            break
        if int(env.score) > score_before:
            valid = True
            break
    env._restore_state(snapshot)
    return valid


def _head_can_reach_tail(env: SnakeEnv) -> bool:
    tail = env.snake[-1]
    blocked = set(env.snake[:-1])
    blocked.discard(env.snake_head)
    queue = deque([env.snake_head])
    seen = {env.snake_head}
    while queue:
        pos = queue.popleft()
        if pos == tail:
            return True
        for dr, dc in env.DIRECTIONS.values():
            next_pos = (pos[0] + dr, pos[1] + dc)
            if not (0 <= next_pos[0] < env.n and 0 <= next_pos[1] < env.n):
                continue
            if next_pos in blocked or next_pos in seen:
                continue
            seen.add(next_pos)
            queue.append(next_pos)
    return False


def _path_reaches_food_with_tail_escape(env: SnakeEnv, path: list[int]) -> bool:
    snapshot = env._snapshot_state()
    score_before = int(env.score)
    valid = False
    for action in path:
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            valid = info.get("reason") == "win"
            break
        if int(env.score) > score_before:
            valid = _head_can_reach_tail(env)
            break
    env._restore_state(snapshot)
    return valid


def _safe_fallback_action(env: SnakeEnv, cycle_index: dict[tuple[int, int], int]) -> int:
    first_nonterminal: int | None = None
    first_tail_reachable: int | None = None
    snapshot = env._snapshot_state()
    for action in range(3):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if info.get("reason") == "win":
                env._restore_state(snapshot)
                return action
            env._restore_state(snapshot)
            continue
        if first_nonterminal is None:
            first_nonterminal = action
        if first_tail_reachable is None and _head_can_reach_tail(env):
            first_tail_reachable = action
        if _cycle_ordered(env, cycle_index):
            env._restore_state(snapshot)
            return action
        env._restore_state(snapshot)
    if first_tail_reachable is not None:
        return first_tail_reachable
    if first_nonterminal is not None:
        return first_nonterminal
    return 1


def _cycle_shortcut_teacher_action(
    env: SnakeEnv,
    cycle: list[tuple[int, int]],
    cycle_index: dict[tuple[int, int], int],
) -> int:
    """Choose a food-directed shortcut only if it preserves cycle ordering."""
    n_cycle = len(cycle)
    head_idx = cycle_index[env.snake_head]
    food_idx = cycle_index[env.food_pos]
    fallback = _safe_fallback_action(env, cycle_index)
    best: tuple[float, int] | None = None
    snapshot = env._snapshot_state()
    score_before = int(env.score)

    for action in range(3):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if info.get("reason") == "win":
                env._restore_state(snapshot)
                return action
            env._restore_state(snapshot)
            continue

        if _cycle_ordered(env, cycle_index):
            target_idx = cycle_index[env.snake_head]
            food_dist = (target_idx - food_idx) % n_cycle
            ate = int(env.score) > score_before
            tail_idx = cycle_index[env.snake[-1]]
            slack = (tail_idx - target_idx) % n_cycle
            teacher_score = (100_000.0 if ate else 0.0) - 100.0 * food_dist + 0.01 * slack
            if best is None or teacher_score > best[0]:
                best = (teacher_score, action)
        env._restore_state(snapshot)

    return best[1] if best is not None else fallback


def _grid_shortest_teacher_action(
    env: SnakeEnv,
    cycle: list[tuple[int, int]],
    cycle_index: dict[tuple[int, int], int],
) -> int:
    """Follow shortest static food paths when that one-step move remains cycle-safe."""
    n_cycle = len(cycle)
    head_idx = cycle_index[env.snake_head]
    fallback = _cycle_shortcut_teacher_action(env, cycle, cycle_index)
    shortest_first_actions = _static_shortest_first_actions(env)
    safe_candidates: list[int] = []
    winning_candidates: list[int] = []
    for action in range(3):
        safe, wins = _action_preserves_cycle_order(env, action, cycle_index)
        if wins:
            winning_candidates.append(action)
        elif safe:
            safe_candidates.append(action)
    if winning_candidates:
        return winning_candidates[0]

    path_candidates = [action for action in safe_candidates if action in shortest_first_actions]
    if path_candidates:
        return min(path_candidates, key=lambda action: _static_food_distance_after_action(env, action))
    if safe_candidates:
        return min(
            safe_candidates,
            key=lambda action: (
                _static_food_distance_after_action(env, action),
                0 if action == fallback else 1,
            ),
        )
    return _safe_fallback_action(env, cycle_index)


def _grid_path_teacher_action(
    env: SnakeEnv,
    cycle: list[tuple[int, int]],
    cycle_index: dict[tuple[int, int], int],
    max_plan_nodes: int,
    max_plan_candidates: int,
) -> int:
    for path in _static_path_candidates(
        env,
        max_nodes=max_plan_nodes,
        max_candidates=max_plan_candidates,
    ):
        if path and _path_preserves_cycle_order(env, path, cycle_index):
            return int(path[0])
    return _cycle_shortcut_teacher_action(env, cycle, cycle_index)


def _tail_path_teacher_action(
    env: SnakeEnv,
    cycle: list[tuple[int, int]],
    cycle_index: dict[tuple[int, int], int],
    max_plan_nodes: int,
    max_plan_candidates: int,
) -> int:
    for path in _static_path_candidates(
        env,
        max_nodes=max_plan_nodes,
        max_candidates=max_plan_candidates,
    ):
        if path and _path_reaches_food_with_tail_escape(env, path):
            return int(path[0])
    return _cycle_shortcut_teacher_action(env, cycle, cycle_index)


def _teacher_action(
    env: SnakeEnv,
    cycle: list[tuple[int, int]],
    cycle_index: dict[tuple[int, int], int],
    teacher_mode: str,
    max_plan_nodes: int,
    max_plan_candidates: int,
    shortcut_score_max: int,
) -> int:
    if teacher_mode == "grid_path":
        return _grid_path_teacher_action(
            env,
            cycle,
            cycle_index,
            max_plan_nodes=max_plan_nodes,
            max_plan_candidates=max_plan_candidates,
        )
    if teacher_mode == "tail_path":
        if shortcut_score_max >= 0 and int(env.score) > shortcut_score_max:
            return _cycle_shortcut_teacher_action(env, cycle, cycle_index)
        return _tail_path_teacher_action(
            env,
            cycle,
            cycle_index,
            max_plan_nodes=max_plan_nodes,
            max_plan_candidates=max_plan_candidates,
        )
    if teacher_mode == "grid_shortest":
        return _grid_shortest_teacher_action(env, cycle, cycle_index)
    if teacher_mode == "cycle":
        return _cycle_shortcut_teacher_action(env, cycle, cycle_index)
    raise ValueError(f"unknown teacher mode: {teacher_mode}")


@torch.no_grad()
def _collect_teacher_seed(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seed: int,
    device: str,
    max_steps: int,
    sample_stride: int,
    correction_weight: float,
    teacher_weight: float,
    teacher_kl_weight: float,
    teacher_mode: str,
    max_plan_nodes: int,
    max_plan_candidates: int,
    shortcut_score_max: int,
    teacher_rollout_policy: str,
) -> dict[str, Any]:
    env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
    obs, _ = env.reset(seed=seed)
    cycle, _ = find_aligned_cycle(env)
    cycle_index = {pos: idx for idx, pos in enumerate(cycle)}
    hidden = policy.initial_state(1, device)
    features: list[torch.Tensor] = []
    labels: list[int] = []
    weights: list[float] = []
    base_logits: list[torch.Tensor] = []
    kl_weights: list[float] = []
    corrections = 0
    samples = 0
    info: dict[str, Any] = {}

    for step in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        encoded = policy.encoder(obs_t)
        hidden = policy.gru_cell(encoded, hidden)
        logits = policy.policy_head(hidden)
        base_action = int(torch.argmax(logits, dim=-1).item())
        teacher_action = _teacher_action(
            env,
            cycle,
            cycle_index,
            teacher_mode,
            max_plan_nodes=max_plan_nodes,
            max_plan_candidates=max_plan_candidates,
            shortcut_score_max=shortcut_score_max,
        )
        is_correction = teacher_action != base_action

        if is_correction or step % sample_stride == 0:
            features.append(hidden.squeeze(0).detach().cpu())
            labels.append(int(teacher_action))
            weights.append(float(correction_weight if is_correction else teacher_weight))
            base_logits.append(logits.squeeze(0).detach().cpu())
            kl_weights.append(float(teacher_kl_weight))
            corrections += int(is_correction)
            samples += 1

        rollout_action = base_action if teacher_rollout_policy == "base" else teacher_action
        obs, _, terminated, truncated, info = env.step(rollout_action)
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "features": features,
        "labels": labels,
        "weights": weights,
        "base_logits": base_logits,
        "kl_weights": kl_weights,
        "samples": samples,
        "corrections": corrections,
        "teacher_score": int(info.get("score", env.score)),
        "teacher_reason": info.get("reason"),
        "teacher_steps": int(info.get("steps", env.total_steps)),
        "teacher_mode": teacher_mode,
        "teacher_rollout_policy": teacher_rollout_policy,
    }


@torch.no_grad()
def _collect_anchor_seed(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seed: int,
    device: str,
    max_steps: int,
    anchor_stride: int,
    anchor_weight: float,
    anchor_kl_weight: float,
) -> dict[str, Any]:
    env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
    obs, _ = env.reset(seed=seed)
    hidden = policy.initial_state(1, device)
    features: list[torch.Tensor] = []
    labels: list[int] = []
    weights: list[float] = []
    base_logits: list[torch.Tensor] = []
    kl_weights: list[float] = []
    samples = 0
    info: dict[str, Any] = {}

    for step in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        encoded = policy.encoder(obs_t)
        hidden = policy.gru_cell(encoded, hidden)
        logits = policy.policy_head(hidden)
        action = int(torch.argmax(logits, dim=-1).item())

        if step % anchor_stride == 0:
            features.append(hidden.squeeze(0).detach().cpu())
            labels.append(action)
            weights.append(float(anchor_weight))
            base_logits.append(logits.squeeze(0).detach().cpu())
            kl_weights.append(float(anchor_kl_weight))
            samples += 1

        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "features": features,
        "labels": labels,
        "weights": weights,
        "base_logits": base_logits,
        "kl_weights": kl_weights,
        "samples": samples,
        "score": int(info.get("score", env.score)),
        "reason": info.get("reason"),
        "steps": int(info.get("steps", env.total_steps)),
    }


def _pop_samples(record: dict[str, Any], dataset: dict[str, list[Any]]) -> None:
    for key in ("features", "labels", "weights", "base_logits", "kl_weights"):
        dataset[key].extend(record.pop(key))


def _collect_dataset(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    teacher_seeds: list[int],
    anchor_seeds: list[int],
    device: str,
    max_steps: int,
    sample_stride: int,
    anchor_stride: int,
    correction_weight: float,
    teacher_weight: float,
    anchor_weight: float,
    teacher_kl_weight: float,
    anchor_kl_weight: float,
    teacher_mode: str,
    max_plan_nodes: int,
    max_plan_candidates: int,
    shortcut_score_max: int,
    teacher_rollout_policy: str,
) -> dict[str, Any]:
    dataset: dict[str, list[Any]] = {
        "features": [],
        "labels": [],
        "weights": [],
        "base_logits": [],
        "kl_weights": [],
    }
    seed_records = []
    for seed in teacher_seeds:
        record = _collect_teacher_seed(
            policy=policy,
            board_size=board_size,
            seed=seed,
            device=device,
            max_steps=max_steps,
            sample_stride=sample_stride,
            correction_weight=correction_weight,
            teacher_weight=teacher_weight,
            teacher_kl_weight=teacher_kl_weight,
            teacher_mode=teacher_mode,
            max_plan_nodes=max_plan_nodes,
            max_plan_candidates=max_plan_candidates,
            shortcut_score_max=shortcut_score_max,
            teacher_rollout_policy=teacher_rollout_policy,
        )
        _pop_samples(record, dataset)
        record["role"] = "teacher"
        seed_records.append(record)
        print({"collect": record}, flush=True)

    for seed in anchor_seeds:
        record = _collect_anchor_seed(
            policy=policy,
            board_size=board_size,
            seed=seed,
            device=device,
            max_steps=max_steps,
            anchor_stride=anchor_stride,
            anchor_weight=anchor_weight,
            anchor_kl_weight=anchor_kl_weight,
        )
        _pop_samples(record, dataset)
        record["role"] = "anchor"
        seed_records.append(record)
        print({"collect": record}, flush=True)

    if not dataset["features"]:
        raise RuntimeError("no samples collected")
    return {
        "features": torch.stack(dataset["features"], dim=0),
        "labels": torch.tensor(dataset["labels"], dtype=torch.long),
        "weights": torch.tensor(dataset["weights"], dtype=torch.float32),
        "base_logits": torch.stack(dataset["base_logits"], dim=0),
        "kl_weights": torch.tensor(dataset["kl_weights"], dtype=torch.float32),
        "seed_records": seed_records,
    }


def _train_head(
    *,
    policy: SnakeRNNPolicy,
    dataset: dict[str, Any],
    lr: float,
    epochs: int,
    batch_size: int,
    device: str,
    head_name: str = "policy_head",
) -> dict[str, Any]:
    if head_name not in {"policy_head", "residual_policy_head"}:
        raise ValueError(f"unsupported head_name={head_name}")
    if not hasattr(policy, head_name):
        raise ValueError(f"policy has no {head_name}")
    for name, param in policy.named_parameters():
        param.requires_grad = name.startswith(head_name)
    optimizer = torch.optim.Adam([param for param in policy.parameters() if param.requires_grad], lr=lr)
    features = dataset["features"].to(device)
    labels = dataset["labels"].to(device)
    weights = dataset["weights"].to(device)
    base_logits = dataset["base_logits"].to(device)
    base_probs = F.softmax(base_logits, dim=-1)
    kl_weights = dataset["kl_weights"].to(device)
    head = getattr(policy, head_name)
    n_samples = int(labels.shape[0])
    losses = []
    ce_losses = []
    kl_losses = []

    policy.train()
    for _ in range(epochs):
        order = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, batch_size):
            index = order[start : start + batch_size]
            logits = head(features[index])
            if head_name == "residual_policy_head":
                logits = base_logits[index] + logits
            ce = F.cross_entropy(logits, labels[index], reduction="none")
            weighted_ce = (ce * weights[index]).sum() / weights[index].sum().clamp_min(1e-6)
            kl_per_sample = F.kl_div(
                F.log_softmax(logits, dim=-1),
                base_probs[index],
                reduction="none",
            ).sum(dim=-1)
            weighted_kl = (kl_per_sample * kl_weights[index]).sum() / kl_weights[index].sum().clamp_min(1e-6)
            loss = weighted_ce + weighted_kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            ce_losses.append(float(weighted_ce.item()))
            kl_losses.append(float(weighted_kl.item()))

    return {
        "samples": n_samples,
        "epochs": epochs,
        "lr": lr,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "mean_ce": float(np.mean(ce_losses)) if ce_losses else 0.0,
        "mean_kl": float(np.mean(kl_losses)) if kl_losses else 0.0,
    }


@torch.no_grad()
def _eval_gate(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seeds: list[int],
    device: str,
    max_steps: int,
    fail_fast: bool,
) -> list[dict[str, Any]]:
    policy.eval()
    results = []
    for seed in seeds:
        result = eval_seed(
            policy=policy,
            board_size=board_size,
            seed=seed,
            device=device,
            max_steps=max_steps,
        )
        results.append(result)
        print({"eval": result}, flush=True)
        if fail_fast and not result["win"]:
            break
    return results


def _score_results(results: list[dict[str, Any]]) -> tuple[int, float, float]:
    wins = sum(int(result["win"]) for result in results)
    mean_score = float(np.mean([int(result["score"]) for result in results])) if results else 0.0
    win_steps = [int(result["steps"]) for result in results if result["win"] and result.get("steps") is not None]
    mean_win_steps = float(np.mean(win_steps)) if win_steps else float("inf")
    return wins, mean_score, -mean_win_steps


def main() -> int:
    parser = argparse.ArgumentParser(description="Search RNN policy-head patches for path-efficient wins")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--teacher-seeds", type=_parse_ints, required=True)
    parser.add_argument("--anchor-seeds", type=_parse_ints, required=True)
    parser.add_argument("--gate-seeds", type=_parse_ints, required=True)
    parser.add_argument("--lrs", type=_parse_floats, required=True)
    parser.add_argument("--epochs", type=_parse_ints, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--anchor-stride", type=int, default=8)
    parser.add_argument("--correction-weight", type=float, default=5.0)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--teacher-kl-weight", type=float, default=0.01)
    parser.add_argument("--anchor-kl-weight", type=float, default=0.1)
    parser.add_argument("--teacher-mode", choices=["cycle", "grid_shortest", "grid_path", "tail_path"], default="cycle")
    parser.add_argument("--max-plan-nodes", type=int, default=2000)
    parser.add_argument("--max-plan-candidates", type=int, default=64)
    parser.add_argument(
        "--shortcut-score-max",
        type=int,
        default=-1,
        help="For tail_path, use shortcut labels only through this score, then fall back to the cycle teacher (-1 = no cutoff).",
    )
    parser.add_argument(
        "--teacher-rollout-policy",
        choices=["teacher", "base"],
        default="teacher",
        help="Step teacher seeds with teacher actions, or keep rollouts on the base policy while labeling shortcut corrections.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--save-all", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_state = torch.load(args.base, map_location="cpu")
    base_policy = _make_policy(
        board_size=args.board_size,
        hidden_size=args.hidden_size,
        device=args.device,
        state=base_state,
    )
    base_policy.eval()
    dataset = _collect_dataset(
        policy=base_policy,
        board_size=args.board_size,
        teacher_seeds=args.teacher_seeds,
        anchor_seeds=args.anchor_seeds,
        device=args.device,
        max_steps=args.max_steps,
        sample_stride=max(1, args.sample_stride),
        anchor_stride=max(1, args.anchor_stride),
        correction_weight=args.correction_weight,
        teacher_weight=args.teacher_weight,
        anchor_weight=args.anchor_weight,
        teacher_kl_weight=args.teacher_kl_weight,
        anchor_kl_weight=args.anchor_kl_weight,
        teacher_mode=args.teacher_mode,
        max_plan_nodes=max(1, args.max_plan_nodes),
        max_plan_candidates=max(1, args.max_plan_candidates),
        shortcut_score_max=args.shortcut_score_max,
        teacher_rollout_policy=args.teacher_rollout_policy,
    )
    dataset_summary = {
        "samples": int(dataset["labels"].shape[0]),
        "teacher_seeds": args.teacher_seeds,
        "anchor_seeds": args.anchor_seeds,
        "gate_seeds": args.gate_seeds,
        "sample_stride": max(1, args.sample_stride),
        "anchor_stride": max(1, args.anchor_stride),
        "corrections": int(sum(record.get("corrections", 0) for record in dataset["seed_records"])),
        "teacher_mode": args.teacher_mode,
        "teacher_rollout_policy": args.teacher_rollout_policy,
        "shortcut_score_max": args.shortcut_score_max,
        "max_plan_nodes": max(1, args.max_plan_nodes),
        "max_plan_candidates": max(1, args.max_plan_candidates),
        "seed_records": dataset["seed_records"],
    }
    print({"dataset": dataset_summary}, flush=True)

    best_key: tuple[int, float, float] | None = None
    best_record: dict[str, Any] | None = None
    started = time.time()
    with (args.out_dir / "search.jsonl").open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"dataset": dataset_summary}, sort_keys=True) + "\n")
        log_file.flush()
        for lr in args.lrs:
            for epoch_count in args.epochs:
                candidate_started = time.time()
                policy = _make_policy(
                    board_size=args.board_size,
                    hidden_size=args.hidden_size,
                    device=args.device,
                    state=base_state,
                )
                train_stats = _train_head(
                    policy=policy,
                    dataset=dataset,
                    lr=lr,
                    epochs=epoch_count,
                    batch_size=args.batch_size,
                    device=args.device,
                )
                gate_results = _eval_gate(
                    policy=policy,
                    board_size=args.board_size,
                    seeds=args.gate_seeds,
                    device=args.device,
                    max_steps=args.max_steps,
                    fail_fast=args.fail_fast,
                )
                key = _score_results(gate_results)
                lr_label = f"{lr:.2e}".replace("+", "").replace(".", "p")
                candidate_path = args.out_dir / f"lr{lr_label}_ep{epoch_count}.pt"
                saved_path = None
                if args.save_all or best_key is None or key > best_key:
                    _save_atomic(policy.state_dict(), candidate_path)
                    saved_path = str(candidate_path)
                if best_key is None or key > best_key:
                    best_key = key
                    best_record = {
                        "checkpoint": str(candidate_path),
                        "key": key,
                        "gate_results": gate_results,
                    }
                win_steps = [
                    int(result["steps"])
                    for result in gate_results
                    if result["win"] and result.get("steps") is not None
                ]
                record = {
                    "lr": lr,
                    "epochs": epoch_count,
                    "train_stats": train_stats,
                    "gate_results": gate_results,
                    "wins": key[0],
                    "gate_count": len(gate_results),
                    "mean_score": float(np.mean([result["score"] for result in gate_results])),
                    "mean_win_steps": float(np.mean(win_steps)) if win_steps else None,
                    "saved_path": saved_path,
                    "elapsed_sec": round(time.time() - candidate_started, 1),
                    "total_elapsed_sec": round(time.time() - started, 1),
                }
                print(record, flush=True)
                log_file.write(json.dumps(record, sort_keys=True) + "\n")
                log_file.flush()

    summary = {
        "dataset": dataset_summary,
        "best": best_record,
        "elapsed_sec": round(time.time() - started, 1),
        "args": vars(args),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print({"summary": str(args.out_dir / "summary.json"), "best": best_record}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
