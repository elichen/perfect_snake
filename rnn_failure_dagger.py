"""Failure-focused on-policy DAgger for recurrent Snake checkpoints.

The policy remains inference-pure. This script uses a train-time Hamiltonian
teacher only to label states that the current policy actually visits, then
upweights policy/teacher disagreements instead of diluting them across an
entire 40k-step expert episode.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from distill.expert import expert_action, find_aligned_cycle
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


@torch.no_grad()
def _collect_rollout(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seed: int,
    device: str,
    max_steps: int,
    focus_window: int,
    window_weight: float,
    min_score: int,
    min_fill: float,
    max_disagreements_log: int,
) -> dict[str, Any]:
    env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
    obs, _ = env.reset(seed=seed)
    hidden = policy.initial_state(1, device)
    observations: list[np.ndarray] = []
    labels: list[int] = []
    actions: list[int] = []
    weights: list[float] = []
    disagreements: list[dict[str, Any]] = []
    disagreement_count = 0
    weighted_disagreement_count = 0
    info: dict[str, Any] = {}

    for step in range(max_steps):
        try:
            cycle, head_idx = find_aligned_cycle(env)
            label, _ = expert_action(env, cycle, head_idx)
        except Exception:
            break
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, hidden = policy.forward_step(obs_t, hidden)
        action = int(torch.argmax(logits, dim=-1).item())

        observations.append(obs.astype(np.float32, copy=True))
        labels.append(int(label))
        actions.append(action)
        is_disagreement = action != int(label)
        fill = float(env.snake_length) / float(board_size * board_size)
        in_focus = int(env.score) >= min_score and fill >= min_fill
        should_weight = is_disagreement and in_focus
        weights.append(1.0 if should_weight else 0.0)
        if is_disagreement:
            disagreement_count += 1
            if should_weight:
                weighted_disagreement_count += 1
            if len(disagreements) < max_disagreements_log:
                disagreements.append(
                    {
                        "step": step,
                        "score": int(env.score),
                        "length": int(env.snake_length),
                        "fill": round(fill, 4),
                        "weighted": bool(should_weight),
                        "head": tuple(env.snake_head),
                        "direction": int(env.direction),
                        "action": action,
                        "teacher": int(label),
                    }
                )
            if should_weight and focus_window > 0:
                start = max(0, len(weights) - focus_window)
                for index in range(start, len(weights) - 1):
                    weights[index] = max(weights[index], window_weight)

        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "observations": observations,
        "labels": labels,
        "actions": actions,
        "weights": weights,
        "disagreements": disagreements,
        "disagreement_count": disagreement_count,
        "weighted_disagreement_count": weighted_disagreement_count,
        "score": int(info.get("score", env.score)),
        "reason": info.get("reason"),
        "steps": info.get("steps"),
    }


def _train_weighted_sequence(
    *,
    policy: SnakeRNNPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, Any],
    seq_len: int,
    device: str,
    grad_clip: float,
) -> dict[str, Any]:
    observations: list[np.ndarray] = rollout["observations"]
    labels: list[int] = rollout["labels"]
    weights: list[float] = rollout["weights"]
    hidden = policy.initial_state(1, device)
    weighted_losses = []
    weight_total = 0.0
    chunks = 0

    for start in range(0, len(observations), seq_len):
        end = min(len(observations), start + seq_len)
        obs_t = torch.as_tensor(np.stack(observations[start:end]), dtype=torch.float32, device=device).unsqueeze(1)
        label_t = torch.as_tensor(labels[start:end], dtype=torch.long, device=device).unsqueeze(1)
        weight_t = torch.as_tensor(weights[start:end], dtype=torch.float32, device=device).unsqueeze(1)
        logits, next_hidden = policy.forward_sequence(obs_t, hidden=hidden)
        chunk_weight = float(weight_t.sum().item())
        if chunk_weight > 0:
            losses = F.cross_entropy(logits.reshape(-1, 3), label_t.reshape(-1), reduction="none").reshape_as(weight_t)
            loss = (losses * weight_t).sum() / weight_t.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()
            weighted_losses.append(float(loss.item()))
            weight_total += chunk_weight
        hidden = next_hidden.detach()
        chunks += 1

    return {
        "chunks": chunks,
        "weighted_steps": weight_total,
        "mean_weighted_loss": float(np.mean(weighted_losses)) if weighted_losses else 0.0,
        "disagreements": rollout["disagreements"],
        "disagreement_count": rollout["disagreement_count"],
        "weighted_disagreement_count": rollout["weighted_disagreement_count"],
        "score": rollout["score"],
        "reason": rollout["reason"],
        "steps": rollout["steps"],
    }


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
        if fail_fast and not result["win"]:
            break
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Failure-focused DAgger patch search")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--train-seeds", type=_parse_ints, required=True)
    parser.add_argument("--gate-seeds", type=_parse_ints, required=True)
    parser.add_argument("--lrs", type=_parse_floats, required=True)
    parser.add_argument("--round-counts", type=_parse_ints, required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--focus-window", type=int, default=32)
    parser.add_argument("--window-weight", type=float, default=0.05)
    parser.add_argument("--min-score", type=int, default=0)
    parser.add_argument("--min-fill", type=float, default=0.0)
    parser.add_argument("--max-disagreements-log", type=int, default=32)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--train-policy-head-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_state = torch.load(args.base, map_location="cpu")
    started = time.time()
    best_key: tuple[int, float, int] | None = None

    with (args.out_dir / "search.jsonl").open("a", encoding="utf-8") as log_file:
        for lr in args.lrs:
            for round_count in args.round_counts:
                candidate_started = time.time()
                policy = _make_policy(
                    board_size=args.board_size,
                    hidden_size=args.hidden_size,
                    device=args.device,
                    state=base_state,
                )
                if args.train_policy_head_only:
                    for name, param in policy.named_parameters():
                        param.requires_grad = name.startswith("policy_head")
                optimizer = torch.optim.Adam(
                    [param for param in policy.parameters() if param.requires_grad],
                    lr=lr,
                )
                train_events = []
                policy.train()
                for round_idx in range(round_count):
                    seed = args.train_seeds[round_idx % len(args.train_seeds)]
                    rollout = _collect_rollout(
                        policy=policy,
                        board_size=args.board_size,
                        seed=seed,
                        device=args.device,
                        max_steps=args.max_steps,
                        focus_window=args.focus_window,
                        window_weight=args.window_weight,
                        min_score=args.min_score,
                        min_fill=args.min_fill,
                        max_disagreements_log=args.max_disagreements_log,
                    )
                    event = _train_weighted_sequence(
                        policy=policy,
                        optimizer=optimizer,
                        rollout=rollout,
                        seq_len=args.seq_len,
                        device=args.device,
                        grad_clip=args.grad_clip,
                    )
                    event["seed"] = seed
                    event["round"] = round_idx + 1
                    train_events.append(event)

                gate_results = _eval_gate(
                    policy=policy,
                    board_size=args.board_size,
                    seeds=args.gate_seeds,
                    device=args.device,
                    max_steps=args.max_steps,
                    fail_fast=args.fail_fast,
                )
                wins = sum(int(result["win"]) for result in gate_results)
                mean_score = float(np.mean([result["score"] for result in gate_results]))
                key = (wins, mean_score, len(gate_results))
                lr_label = f"{lr:.2e}".replace("+", "").replace(".", "p")
                candidate_path = args.out_dir / f"lr{lr_label}_rounds{round_count}.pt"
                if best_key is None or key > best_key or wins == len(args.gate_seeds):
                    best_key = key
                    _save_atomic(policy.state_dict(), candidate_path)
                    saved_path = str(candidate_path)
                else:
                    saved_path = None

                record = {
                    "lr": lr,
                    "round_count": round_count,
                    "train_seeds": args.train_seeds,
                    "gate_seeds": args.gate_seeds,
                    "min_score": args.min_score,
                    "min_fill": args.min_fill,
                    "focus_window": args.focus_window,
                    "window_weight": args.window_weight,
                    "gate_results": gate_results,
                    "wins": wins,
                    "gate_count": len(gate_results),
                    "mean_score": mean_score,
                    "saved_path": saved_path,
                    "train_events": train_events,
                    "elapsed_sec": round(time.time() - candidate_started, 1),
                    "total_elapsed_sec": round(time.time() - started, 1),
                }
                print(record, flush=True)
                log_file.write(json.dumps(record, sort_keys=True) + "\n")
                log_file.flush()
                if wins == len(args.gate_seeds):
                    print({"gate_passed": True, "checkpoint": saved_path}, flush=True)
                    return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
