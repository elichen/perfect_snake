"""Progress-printing exact-seed evaluator for recurrent Snake checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distill.rnn_model import SnakeRNNPolicy, load_rnn_policy_state
from eval_metrics import summarize_phase_metrics
from snake_env import SnakeEnv


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("expected at least one seed")
    return seeds


@torch.no_grad()
def eval_seed(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seed: int,
    device: str,
    max_steps: int,
    use_fill_values: bool,
) -> dict[str, Any]:
    env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
    obs, _ = env.reset(seed=seed)
    hidden = policy.initial_state(1, device)
    info: dict[str, Any] = {}
    for _ in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        fill_t = None
        if use_fill_values:
            fill_t = torch.as_tensor(
                [env.snake_length / float(board_size * board_size)],
                dtype=torch.float32,
                device=device,
            )
        logits, hidden = policy.forward_step(obs_t, hidden, fill_values=fill_t)
        action = int(torch.argmax(logits, dim=-1).item())
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    score = int(info.get("score", 0))
    return {
        "seed": seed,
        "score": score,
        "length": int(info.get("length", score + 3)),
        "reason": str(info.get("reason", "unknown")),
        "steps": info.get("steps"),
        "win": score >= board_size * board_size - 3,
    }


def summarize(results: list[dict[str, Any]], *, board_size: int, checkpoint: str) -> dict[str, Any]:
    perfect_score = board_size * board_size - 3
    scores = [int(result["score"]) for result in results]
    lengths = [int(result["length"]) for result in results]
    reasons = [str(result["reason"]) for result in results]
    wins = sum(int(score >= perfect_score) for score in scores)
    win_steps = [int(result["steps"]) for result in results if result["win"] and result.get("steps") is not None]
    summary = {
        "episodes": len(results),
        "wins": wins,
        "win_rate": float(wins / max(1, len(results))),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "median_score": float(np.median(scores)) if scores else 0.0,
        "min_score": int(min(scores)) if scores else 0,
        "max_score": int(max(scores)) if scores else 0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "failures": [result for result in results if not result["win"]],
        "checkpoint": checkpoint,
        "mean_win_steps": float(np.mean(win_steps)) if win_steps else None,
        "median_win_steps": float(np.median(win_steps)) if win_steps else None,
        "p95_win_steps": float(np.percentile(win_steps, 95)) if win_steps else None,
        "min_win_steps": int(min(win_steps)) if win_steps else None,
        "max_win_steps": int(max(win_steps)) if win_steps else None,
        "steps_per_food": float(np.mean(win_steps) / perfect_score) if win_steps else None,
    }
    summary.update(
        summarize_phase_metrics(
            scores=scores,
            terminal_lengths=lengths,
            reasons=reasons,
            perfect_score=perfect_score,
            episodes=len(results),
        )
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RNN checkpoint on exact seed ranges with progress")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--early-head-max-fill", type=float, default=None)
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
    ).to(args.device)
    load_rnn_policy_state(policy, torch.load(args.checkpoint, map_location="cpu"))
    policy.eval()

    results = []
    failures = 0
    for index, seed in enumerate(seeds, start=1):
        result = eval_seed(
            policy=policy,
            board_size=args.board_size,
            seed=seed,
            device=args.device,
            max_steps=args.max_steps,
            use_fill_values=args.early_head_max_fill is not None,
        )
        results.append(result)
        if not result["win"]:
            failures += 1
            print({"failure": result, "idx": index, "failures": failures}, flush=True)
            if args.stop_after_failures and failures >= args.stop_after_failures:
                break
        elif args.progress_every > 0 and index % args.progress_every == 0:
            print({"idx": index, "wins": index - failures, "failures": failures}, flush=True)

    print({"summary": summarize(results, board_size=args.board_size, checkpoint=str(args.checkpoint))}, flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
