"""Search tiny recurrent-BC patches around a strong checkpoint.

This is a tooling script for the non-cheating RNN branch: inference still uses
only the standard Snake observation stream. The script trains from a fixed base
checkpoint on one or more expert trajectories, then evaluates exact seed gates
in fail-fast order so patches that fix one late failure but regress another are
discarded quickly.
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

from distill.rnn_model import SnakeRNNPolicy, load_rnn_policy_state
from rnn_online_cycle_bc import train_episode
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


def _make_policy(*, board_size: int, hidden_size: int, device: str, base_state: dict[str, Any]) -> SnakeRNNPolicy:
    policy = SnakeRNNPolicy(board_size=board_size, n_channels=5, hidden_size=hidden_size).to(device)
    load_rnn_policy_state(policy, base_state)
    return policy


@torch.no_grad()
def _eval_seed(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seed: int,
    device: str,
    max_steps: int,
) -> dict[str, Any]:
    env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
    obs, _ = env.reset(seed=seed)
    hidden = policy.initial_state(1, device)
    info: dict[str, Any] = {}
    for _ in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, hidden = policy.forward_step(obs_t, hidden)
        action = int(torch.argmax(logits, dim=-1).item())
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    score = int(info.get("score", 0))
    return {
        "seed": seed,
        "score": score,
        "reason": info.get("reason"),
        "steps": info.get("steps"),
        "win": score >= board_size * board_size - 3,
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
        result = _eval_seed(
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
    parser = argparse.ArgumentParser(description="Tiny BC patch search for recurrent Snake policies")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--train-seeds", type=_parse_ints, required=True)
    parser.add_argument("--gate-seeds", type=_parse_ints, required=True)
    parser.add_argument("--lrs", type=_parse_floats, required=True)
    parser.add_argument("--episode-counts", type=_parse_ints, required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=80_000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--train-policy-head-only", action="store_true")
    parser.add_argument("--save-all", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "search.jsonl"
    base_state = torch.load(args.base, map_location="cpu")
    started = time.time()
    best_key: tuple[int, float, int] | None = None
    best_path: Path | None = None

    with log_path.open("a", encoding="utf-8") as log_file:
        for lr in args.lrs:
            for episode_count in args.episode_counts:
                candidate_started = time.time()
                policy = _make_policy(
                    board_size=args.board_size,
                    hidden_size=args.hidden_size,
                    device=args.device,
                    base_state=base_state,
                )
                if args.train_policy_head_only:
                    for name, param in policy.named_parameters():
                        param.requires_grad = name.startswith("policy_head")
                optimizer = torch.optim.Adam(
                    [param for param in policy.parameters() if param.requires_grad],
                    lr=lr,
                )
                policy.train()
                train_events = []
                for ep_idx in range(episode_count):
                    train_seed = args.train_seeds[ep_idx % len(args.train_seeds)]
                    event = train_episode(
                        policy=policy,
                        anchor_policy=None,
                        rollout_policy=None,
                        optimizer=optimizer,
                        board_size=args.board_size,
                        flood_fill=False,
                        head_centered=False,
                        seed=train_seed,
                        device=args.device,
                        seq_len=args.seq_len,
                        max_steps=args.max_steps,
                        grad_clip=args.grad_clip,
                        teacher_mode="hamiltonian",
                        max_plan_nodes=1,
                        max_plan_candidates=1,
                        kl_anchor_coef=0.0,
                        shortcut_score_max=-1,
                        teacher_rollout_policy="teacher",
                        correction_weight=1.0,
                        teacher_weight=1.0,
                    )
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
                candidate_name = f"lr{lr_label}_ep{episode_count}"
                candidate_path = args.out_dir / f"{candidate_name}.pt"
                should_save = args.save_all or best_key is None or key > best_key or wins == len(args.gate_seeds)
                if should_save:
                    _save_atomic(policy.state_dict(), candidate_path)
                if best_key is None or key > best_key:
                    best_key = key
                    best_path = candidate_path if should_save else None
                record = {
                    "candidate": candidate_name,
                    "lr": lr,
                    "episode_count": episode_count,
                    "train_seeds": args.train_seeds,
                    "gate_seeds": args.gate_seeds,
                    "gate_results": gate_results,
                    "wins": wins,
                    "gate_count": len(gate_results),
                    "mean_score": mean_score,
                    "saved_path": str(candidate_path) if should_save else None,
                    "best_path": str(best_path) if best_path is not None else None,
                    "train_events": train_events,
                    "elapsed_sec": round(time.time() - candidate_started, 1),
                    "total_elapsed_sec": round(time.time() - started, 1),
                }
                print(record, flush=True)
                log_file.write(json.dumps(record, sort_keys=True) + "\n")
                log_file.flush()
                if wins == len(args.gate_seeds):
                    print({"gate_passed": True, "checkpoint": str(candidate_path)}, flush=True)
                    return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
