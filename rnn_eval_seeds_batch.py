"""Batched exact-seed evaluator for recurrent Snake checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distill.rnn_model import SnakeRNNPolicy, load_rnn_policy_state
from rnn_eval_seeds import summarize
from snake_env import SnakeEnv


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("expected at least one seed")
    return seeds


@torch.no_grad()
def eval_seed_batch(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seeds: list[int],
    device: str,
    max_steps: int,
    use_fill_values: bool,
    progress_every: int,
    stop_after_failures: int,
) -> list[dict[str, Any]]:
    envs = [SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed) for seed in seeds]
    observations = []
    for env, seed in zip(envs, seeds):
        obs, _ = env.reset(seed=seed)
        observations.append(obs)

    hidden = policy.initial_state(len(seeds), device)
    done = [False for _ in seeds]
    results: list[dict[str, Any] | None] = [None for _ in seeds]
    completed = 0
    failures = 0

    for _ in range(max_steps):
        active = [idx for idx, is_done in enumerate(done) if not is_done]
        if not active:
            break

        active_t = torch.as_tensor(active, dtype=torch.long, device=device)
        obs_t = torch.as_tensor(
            np.stack([observations[idx] for idx in active], axis=0),
            dtype=torch.float32,
            device=device,
        )
        fill_t = None
        if use_fill_values:
            fill_t = torch.as_tensor(
                [envs[idx].snake_length / float(board_size * board_size) for idx in active],
                dtype=torch.float32,
                device=device,
            )
        logits, active_hidden = policy.forward_step(
            obs_t,
            hidden.index_select(0, active_t),
            fill_values=fill_t,
        )
        hidden[active_t] = active_hidden
        actions = torch.argmax(logits, dim=-1).detach().cpu().tolist()

        for idx, action in zip(active, actions):
            obs, _, terminated, truncated, info = envs[idx].step(int(action))
            observations[idx] = obs
            if terminated or truncated:
                score = int(info.get("score", 0))
                result = {
                    "seed": seeds[idx],
                    "score": score,
                    "length": int(info.get("length", score + 3)),
                    "reason": str(info.get("reason", "unknown")),
                    "steps": info.get("steps"),
                    "win": score >= board_size * board_size - 3,
                }
                results[idx] = result
                done[idx] = True
                completed += 1
                if not result["win"]:
                    failures += 1
                    print({"failure": result, "idx": completed, "failures": failures}, flush=True)
                    if stop_after_failures and failures >= stop_after_failures:
                        return [result for result in results if result is not None]
                elif progress_every > 0 and completed % progress_every == 0:
                    print({"idx": completed, "wins": completed - failures, "failures": failures}, flush=True)

    for idx, result in enumerate(results):
        if result is None:
            result = {
                "seed": seeds[idx],
                "score": int(envs[idx].score),
                "length": int(envs[idx].snake_length),
                "reason": "max_steps",
                "steps": int(envs[idx].total_steps),
                "win": False,
            }
            results[idx] = result
            failures += 1
            print({"failure": result, "idx": idx + 1, "failures": failures}, flush=True)
            if stop_after_failures and failures >= stop_after_failures:
                break

    return [result for result in results if result is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RNN checkpoint on exact seeds with batched policy calls")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--early-head-max-fill", type=float, default=None)
    parser.add_argument("--residual-policy-head", action="store_true")
    parser.add_argument("--residual-min-fill", type=float, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=20001)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=_parse_seeds, default=None)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--stop-after-failures", type=int, default=0)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.seed, args.seed + args.episodes))
    policy = SnakeRNNPolicy(
        board_size=args.board_size,
        n_channels=5,
        hidden_size=args.hidden_size,
        early_head_max_fill=args.early_head_max_fill,
        residual_policy_head=args.residual_policy_head,
        residual_policy_min_fill=args.residual_min_fill,
    ).to(args.device)
    load_rnn_policy_state(policy, torch.load(args.checkpoint, map_location="cpu"))
    policy.eval()

    results = eval_seed_batch(
        policy=policy,
        board_size=args.board_size,
        seeds=seeds,
        device=args.device,
        max_steps=args.max_steps,
        use_fill_values=args.early_head_max_fill is not None or args.residual_min_fill is not None,
        progress_every=args.progress_every,
        stop_after_failures=args.stop_after_failures,
    )
    summary = summarize(results, board_size=args.board_size, checkpoint=str(args.checkpoint))
    print({"summary": summary}, flush=True)
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
