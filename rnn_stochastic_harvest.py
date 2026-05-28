"""Harvest shorter winning trajectories near a recurrent Snake checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distill.rnn_model import SnakeRNNPolicy, load_rnn_policy_state
from eval_metrics import summarize_phase_metrics
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


def _make_policy(*, checkpoint: Path, board_size: int, hidden_size: int, device: str) -> SnakeRNNPolicy:
    policy = SnakeRNNPolicy(board_size=board_size, n_channels=5, hidden_size=hidden_size).to(device)
    load_rnn_policy_state(policy, torch.load(checkpoint, map_location="cpu"))
    policy.eval()
    return policy


@torch.no_grad()
def rollout(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seed: int,
    device: str,
    max_steps: int,
    temperature: float,
    epsilon: float,
    explore_score_max: int,
    rng: random.Random,
    record_actions: bool,
) -> dict[str, Any]:
    env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
    obs, _ = env.reset(seed=seed)
    hidden = policy.initial_state(1, device)
    actions: list[int] = []
    info: dict[str, Any] = {}

    for _ in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, hidden = policy.forward_step(obs_t, hidden)
        logits_np = logits[0].detach().cpu().numpy().astype(np.float64)
        greedy = int(np.argmax(logits_np))
        can_explore = explore_score_max < 0 or int(env.score) <= explore_score_max
        if can_explore and epsilon > 0.0 and rng.random() < epsilon:
            choices = [action for action in (0, 1, 2) if action != greedy]
            action = int(rng.choice(choices))
        elif can_explore and temperature > 0.0:
            shifted = logits_np / max(1e-6, temperature)
            shifted = shifted - np.max(shifted)
            probs = np.exp(shifted)
            probs = probs / probs.sum()
            action = int(rng.choices((0, 1, 2), weights=probs, k=1)[0])
        else:
            action = greedy
        if record_actions:
            actions.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    score = int(info.get("score", env.score))
    result = {
        "seed": seed,
        "score": score,
        "length": int(info.get("length", score + 3)),
        "reason": str(info.get("reason", "timeout")),
        "steps": int(info.get("steps", env.total_steps)),
        "win": score >= board_size * board_size - 3,
    }
    if record_actions:
        result["actions"] = actions
    return result


def summarize(results: list[dict[str, Any]], *, board_size: int) -> dict[str, Any]:
    perfect_score = board_size * board_size - 3
    scores = [int(result["score"]) for result in results]
    lengths = [int(result["length"]) for result in results]
    reasons = [str(result["reason"]) for result in results]
    wins = [result for result in results if result["win"]]
    win_steps = [int(result["steps"]) for result in wins]
    summary = {
        "episodes": len(results),
        "wins": len(wins),
        "win_rate": float(len(wins) / max(1, len(results))),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "median_score": float(np.median(scores)) if scores else 0.0,
        "min_score": int(min(scores)) if scores else 0,
        "max_score": int(max(scores)) if scores else 0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "mean_win_steps": float(np.mean(win_steps)) if win_steps else None,
        "median_win_steps": float(np.median(win_steps)) if win_steps else None,
        "p95_win_steps": float(np.percentile(win_steps, 95)) if win_steps else None,
        "steps_per_food": float(np.mean(win_steps) / perfect_score) if win_steps else None,
        "failures": [
            {key: value for key, value in result.items() if key != "actions"}
            for result in results
            if not result["win"]
        ],
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
    parser = argparse.ArgumentParser(description="Sample stochastic rollouts around a recurrent checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seeds", type=_parse_ints, required=True)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--temperatures", type=_parse_floats, default=[0.0])
    parser.add_argument("--epsilons", type=_parse_floats, default=[0.0])
    parser.add_argument("--explore-score-max", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-improvement", type=int, default=1)
    args = parser.parse_args()

    if args.attempts < 1:
        raise SystemExit("--attempts must be >= 1")
    policy = _make_policy(
        checkpoint=args.checkpoint,
        board_size=args.board_size,
        hidden_size=args.hidden_size,
        device=args.device,
    )
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    baseline_by_seed: dict[int, dict[str, Any]] = {}
    all_results: list[dict[str, Any]] = []
    improved: list[dict[str, Any]] = []
    with args.out.open("w", encoding="utf-8") as f:
        for seed in args.seeds:
            baseline = rollout(
                policy=policy,
                board_size=args.board_size,
                seed=seed,
                device=args.device,
                max_steps=args.max_steps,
                temperature=0.0,
                epsilon=0.0,
                explore_score_max=-1,
                rng=rng,
                record_actions=False,
            )
            baseline_by_seed[seed] = baseline
            print({"baseline": baseline}, flush=True)
            for temperature in args.temperatures:
                for epsilon in args.epsilons:
                    for attempt in range(1, args.attempts + 1):
                        result = rollout(
                            policy=policy,
                            board_size=args.board_size,
                            seed=seed,
                            device=args.device,
                            max_steps=args.max_steps,
                            temperature=temperature,
                            epsilon=epsilon,
                            explore_score_max=args.explore_score_max,
                            rng=rng,
                            record_actions=True,
                        )
                        result.update(
                            {
                                "attempt": attempt,
                                "temperature": temperature,
                                "epsilon": epsilon,
                                "explore_score_max": args.explore_score_max,
                                "baseline_steps": baseline["steps"],
                                "improvement": int(baseline["steps"]) - int(result["steps"])
                                if result["win"] and baseline["win"]
                                else None,
                            }
                        )
                        all_results.append(result)
                        if (
                            result["win"]
                            and baseline["win"]
                            and int(result["steps"]) <= int(baseline["steps"]) - args.min_improvement
                        ):
                            improved.append(result)
                            f.write(json.dumps(result, sort_keys=True) + "\n")
                            f.flush()
                            print({"improved": {key: value for key, value in result.items() if key != "actions"}}, flush=True)
                        else:
                            print({"sample": {key: value for key, value in result.items() if key != "actions"}}, flush=True)

    best_improved = min(improved, key=lambda item: int(item["steps"])) if improved else None
    if best_improved is not None:
        best_improved = {key: value for key, value in best_improved.items() if key != "actions"}
    summary = {
        "checkpoint": str(args.checkpoint),
        "baseline": baseline_by_seed,
        "sample_summary": summarize(all_results, board_size=args.board_size),
        "improved_count": len(improved),
        "best_improved": best_improved,
        "out": str(args.out),
    }
    print(json.dumps({"summary": summary}, default=str, sort_keys=True), flush=True)
    summary_path = args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
