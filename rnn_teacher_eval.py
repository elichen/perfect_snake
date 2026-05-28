"""Evaluate train-time RNN shortcut teachers without a neural policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from distill.expert import expert_action, find_aligned_cycle
from eval_metrics import summarize_phase_metrics
from rnn_cycle_shortcut_patch import _teacher_action
from snake_env import SnakeEnv


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("expected at least one seed")
    return seeds


def eval_teacher_seed(
    *,
    board_size: int,
    seed: int,
    teacher_mode: str,
    max_steps: int,
    max_plan_nodes: int,
    max_plan_candidates: int,
    shortcut_score_max: int,
) -> dict[str, Any]:
    env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
    env.reset(seed=seed)
    cycle, head_idx = find_aligned_cycle(env)
    cycle_index = {pos: idx for idx, pos in enumerate(cycle)}
    info: dict[str, Any] = {}

    for _ in range(max_steps):
        if teacher_mode == "hamiltonian":
            action, head_idx = expert_action(env, cycle, head_idx)
        else:
            action = _teacher_action(
                env,
                cycle,
                cycle_index,
                teacher_mode,
                max_plan_nodes=max_plan_nodes,
                max_plan_candidates=max_plan_candidates,
                shortcut_score_max=shortcut_score_max,
            )
            new_dir = (env.direction + {0: -1, 1: 0, 2: 1}[int(action)]) % 4
            dr, dc = env.DIRECTIONS[new_dir]
            hr, hc = env.snake_head
            head_idx = cycle_index.get((hr + dr, hc + dc), head_idx)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    score = int(info.get("score", env.score))
    return {
        "seed": seed,
        "score": score,
        "length": int(info.get("length", score + 3)),
        "reason": str(info.get("reason", "timeout")),
        "steps": int(info.get("steps", env.total_steps)),
        "win": score >= board_size * board_size - 3,
    }


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
        "failures": [result for result in results if not result["win"]],
        "mean_win_steps": float(np.mean(win_steps)) if win_steps else None,
        "median_win_steps": float(np.median(win_steps)) if win_steps else None,
        "p95_win_steps": float(np.percentile(win_steps, 95)) if win_steps else None,
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
    parser = argparse.ArgumentParser(description="Evaluate train-time RNN shortcut teachers")
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--teacher-mode", choices=["hamiltonian", "cycle", "grid_shortest", "grid_path", "tail_path"], required=True)
    parser.add_argument("--seed", type=int, default=20001)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=_parse_seeds, default=None)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--max-plan-nodes", type=int, default=2000)
    parser.add_argument("--max-plan-candidates", type=int, default=64)
    parser.add_argument("--shortcut-score-max", type=int, default=-1)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.seed, args.seed + args.episodes))
    results = []
    for index, seed in enumerate(seeds, start=1):
        result = eval_teacher_seed(
            board_size=args.board_size,
            seed=seed,
            teacher_mode=args.teacher_mode,
            max_steps=args.max_steps,
            max_plan_nodes=max(1, args.max_plan_nodes),
            max_plan_candidates=max(1, args.max_plan_candidates),
            shortcut_score_max=args.shortcut_score_max,
        )
        results.append(result)
        if args.progress_every > 0 and index % args.progress_every == 0:
            print({"idx": index, "result": result}, flush=True)

    payload = {
        "args": vars(args),
        "results": results,
        "summary": summarize(results, board_size=args.board_size),
    }
    print(json.dumps({"summary": payload["summary"]}, sort_keys=True), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if payload["summary"]["wins"] == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
