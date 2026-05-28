"""Batched scan of forced initial actions for recurrent Snake checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distill.rnn_model import SnakeRNNPolicy, load_rnn_policy_state
from snake_env import SnakeEnv


def _parse_ints(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _parse_seed_ranges(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid seed range: {item}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(item))
    if not seeds:
        raise argparse.ArgumentTypeError("expected at least one seed or seed range")
    return seeds


def _make_policy(
    *,
    checkpoint: Path,
    board_size: int,
    hidden_size: int,
    early_head_max_fill: float | None,
    device: str,
) -> SnakeRNNPolicy:
    policy = SnakeRNNPolicy(
        board_size=board_size,
        n_channels=5,
        hidden_size=hidden_size,
        early_head_max_fill=early_head_max_fill,
    ).to(device)
    load_rnn_policy_state(policy, torch.load(checkpoint, map_location="cpu"))
    policy.eval()
    return policy


@torch.no_grad()
def scan_initial_actions(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seeds: list[int],
    device: str,
    max_steps: int,
    use_fill_values: bool,
    min_improvement: int,
    progress_every: int,
) -> dict[str, Any]:
    seed_envs = [SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed) for seed in seeds]
    seed_obs = []
    for env, seed in zip(seed_envs, seeds):
        obs, _ = env.reset(seed=seed)
        seed_obs.append(obs)

    initial_hidden = policy.initial_state(len(seeds), device)
    obs_t = torch.as_tensor(np.stack(seed_obs, axis=0), dtype=torch.float32, device=device)
    fill_t = None
    if use_fill_values:
        fill_t = torch.as_tensor(
            [env.snake_length / float(board_size * board_size) for env in seed_envs],
            dtype=torch.float32,
            device=device,
        )
    logits, hidden_after_obs = policy.forward_step(obs_t, initial_hidden, fill_values=fill_t)
    base_actions = torch.argmax(logits, dim=-1).detach().cpu().tolist()

    jobs: list[dict[str, Any]] = []
    envs: list[SnakeEnv] = []
    observations: list[np.ndarray | None] = []
    hidden_rows: list[torch.Tensor] = []
    done: list[bool] = []
    action_logs: list[bytearray] = []
    results: list[dict[str, Any] | None] = []

    for seed_index, seed in enumerate(seeds):
        for forced_action in (0, 1, 2):
            env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
            env.reset(seed=seed)
            obs, _, terminated, truncated, info = env.step(forced_action)
            envs.append(env)
            jobs.append(
                {
                    "seed": seed,
                    "seed_index": seed_index,
                    "forced_action": forced_action,
                    "base_action": int(base_actions[seed_index]),
                }
            )
            hidden_rows.append(hidden_after_obs[seed_index].detach())
            action_logs.append(bytearray([forced_action]))
            if terminated or truncated:
                score = int(info.get("score", env.score))
                results.append(
                    {
                        "score": score,
                        "length": int(info.get("length", score + 3)),
                        "reason": str(info.get("reason", "unknown")),
                        "steps": int(info.get("steps", env.total_steps)),
                        "win": score >= board_size * board_size - 3,
                    }
                )
                observations.append(None)
                done.append(True)
            else:
                results.append(None)
                observations.append(obs)
                done.append(False)

    hidden = torch.stack(hidden_rows, dim=0).to(device)
    completed = sum(int(is_done) for is_done in done)
    for _ in range(max_steps):
        active = [idx for idx, is_done in enumerate(done) if not is_done]
        if not active:
            break

        active_t = torch.as_tensor(active, dtype=torch.long, device=device)
        obs_batch = torch.as_tensor(
            np.stack([observations[idx] for idx in active if observations[idx] is not None], axis=0),
            dtype=torch.float32,
            device=device,
        )
        fill_batch = None
        if use_fill_values:
            fill_batch = torch.as_tensor(
                [envs[idx].snake_length / float(board_size * board_size) for idx in active],
                dtype=torch.float32,
                device=device,
            )
        logits, active_hidden = policy.forward_step(
            obs_batch,
            hidden.index_select(0, active_t),
            fill_values=fill_batch,
        )
        hidden[active_t] = active_hidden
        actions = torch.argmax(logits, dim=-1).detach().cpu().tolist()

        for idx, action in zip(active, actions):
            action_logs[idx].append(int(action))
            obs, _, terminated, truncated, info = envs[idx].step(int(action))
            observations[idx] = obs
            if terminated or truncated:
                score = int(info.get("score", envs[idx].score))
                results[idx] = {
                    "score": score,
                    "length": int(info.get("length", score + 3)),
                    "reason": str(info.get("reason", "unknown")),
                    "steps": int(info.get("steps", envs[idx].total_steps)),
                    "win": score >= board_size * board_size - 3,
                }
                done[idx] = True
                completed += 1
                if progress_every > 0 and completed % progress_every == 0:
                    print({"completed": completed, "jobs": len(jobs)}, flush=True)

    for idx, result in enumerate(results):
        if result is None:
            result = {
                "score": int(envs[idx].score),
                "length": int(envs[idx].snake_length),
                "reason": "max_steps",
                "steps": int(envs[idx].total_steps),
                "win": False,
            }
            results[idx] = result

    by_seed: dict[int, dict[str, Any]] = {}
    for job, result, actions in zip(jobs, results, action_logs):
        seed = int(job["seed"])
        seed_result = by_seed.setdefault(
            seed,
            {
                "seed": seed,
                "base_action": int(job["base_action"]),
                "baseline": None,
                "tested": 0,
                "improved": [],
            },
        )
        if int(job["forced_action"]) == int(job["base_action"]):
            seed_result["baseline"] = result
        else:
            seed_result["tested"] += 1

    for job, result, actions in zip(jobs, results, action_logs):
        seed = int(job["seed"])
        seed_result = by_seed[seed]
        baseline = seed_result["baseline"]
        if baseline is None or int(job["forced_action"]) == int(job["base_action"]):
            continue
        improvement = int(baseline["steps"]) - int(result["steps"])
        if result["win"] and improvement >= min_improvement:
            seed_result["improved"].append(
                {
                    "seed": seed,
                    "deviation_step": 0,
                    "deviation_score": 0,
                    "base_action": int(job["base_action"]),
                    "alt_action": int(job["forced_action"]),
                    "result": result,
                    "actions": list(actions),
                    "baseline_steps": int(baseline["steps"]),
                    "improvement": improvement,
                }
            )

    results_by_seed = [by_seed[seed] for seed in seeds]
    improved_records = [
        record
        for seed_result in results_by_seed
        for record in seed_result["improved"]
    ]
    return {
        "results": results_by_seed,
        "summary": {
            "seeds": seeds,
            "tested": sum(int(seed_result["tested"]) for seed_result in results_by_seed),
            "improved_count": len(improved_records),
            "improved_seed_count": sum(int(bool(seed_result["improved"])) for seed_result in results_by_seed),
            "best_improvement": max((record["improvement"] for record in improved_records), default=None),
            "mean_baseline_steps": float(
                np.mean([int(seed_result["baseline"]["steps"]) for seed_result in results_by_seed])
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-scan forced first actions from an RNN checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--early-head-max-fill", type=float, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=20001)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=_parse_ints, default=None)
    parser.add_argument("--seed-ranges", type=_parse_seed_ranges, default=None)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--min-improvement", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed_ranges is not None:
        seeds = args.seed_ranges
    else:
        seeds = list(range(args.seed, args.seed + args.episodes))

    policy = _make_policy(
        checkpoint=args.checkpoint,
        board_size=args.board_size,
        hidden_size=args.hidden_size,
        early_head_max_fill=args.early_head_max_fill,
        device=args.device,
    )
    result = scan_initial_actions(
        policy=policy,
        board_size=args.board_size,
        seeds=seeds,
        device=args.device,
        max_steps=args.max_steps,
        use_fill_values=args.early_head_max_fill is not None,
        min_improvement=args.min_improvement,
        progress_every=args.progress_every,
    )
    result["summary"]["out"] = str(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["summary"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
