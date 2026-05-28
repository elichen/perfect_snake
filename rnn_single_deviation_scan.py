"""Scan one-action deviations from a recurrent Snake checkpoint trajectory."""

from __future__ import annotations

import argparse
import json
import copy
from pathlib import Path
from typing import Any

import torch

from distill.rnn_model import SnakeRNNPolicy, load_rnn_policy_state
from snake_env import SnakeEnv


def _parse_ints(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _make_policy(
    *,
    checkpoint: Path,
    board_size: int,
    hidden_size: int,
    early_head_max_fill: float | None,
    residual_policy_head: bool,
    device: str,
) -> SnakeRNNPolicy:
    policy = SnakeRNNPolicy(
        board_size=board_size,
        n_channels=5,
        hidden_size=hidden_size,
        early_head_max_fill=early_head_max_fill,
        residual_policy_head=residual_policy_head,
    ).to(device)
    load_rnn_policy_state(policy, torch.load(checkpoint, map_location="cpu"))
    policy.eval()
    return policy


@torch.no_grad()
def _greedy_action(
    policy: SnakeRNNPolicy,
    obs,
    hidden,
    device: str,
    fill_value: float | None,
) -> tuple[int, torch.Tensor]:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    fill_t = None
    if fill_value is not None:
        fill_t = torch.as_tensor([fill_value], dtype=torch.float32, device=device)
    logits, next_hidden = policy.forward_step(obs_t, hidden, fill_values=fill_t)
    return int(torch.argmax(logits, dim=-1).item()), next_hidden


@torch.no_grad()
def _finish_rollout(
    *,
    policy: SnakeRNNPolicy,
    env: SnakeEnv,
    obs,
    hidden: torch.Tensor,
    device: str,
    max_steps: int,
    actions: list[int] | None,
    use_fill_values: bool,
) -> dict[str, Any]:
    info: dict[str, Any] = {}
    while env.total_steps < max_steps:
        fill_value = None
        if use_fill_values:
            fill_value = env.snake_length / float(env.n * env.n)
        action, hidden = _greedy_action(policy, obs, hidden, device, fill_value=fill_value)
        if actions is not None:
            actions.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    score = int(info.get("score", env.score))
    return {
        "score": score,
        "length": int(info.get("length", score + 3)),
        "reason": str(info.get("reason", "timeout")),
        "steps": int(info.get("steps", env.total_steps)),
        "win": score >= env.n * env.n - 3,
    }


@torch.no_grad()
def scan_seed(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seed: int,
    device: str,
    max_steps: int,
    use_fill_values: bool,
    scan_step_min: int,
    scan_step_max: int,
    scan_score_min: int,
    scan_score_max: int,
    stride: int,
    min_improvement: int,
    stop_after_improved: int,
) -> dict[str, Any]:
    baseline_env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
    baseline_obs, _ = baseline_env.reset(seed=seed)
    baseline_hidden = policy.initial_state(1, device)
    baseline = _finish_rollout(
        policy=policy,
        env=baseline_env,
        obs=baseline_obs,
        hidden=baseline_hidden,
        device=device,
        max_steps=max_steps,
        actions=None,
        use_fill_values=use_fill_values,
    )
    baseline["seed"] = seed

    env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
    obs, _ = env.reset(seed=seed)
    hidden = policy.initial_state(1, device)
    baseline_actions: list[int] = []
    improved: list[dict[str, Any]] = []
    tested = 0

    while env.total_steps < max_steps:
        step = int(env.total_steps)
        score = int(env.score)
        fill_value = None
        if use_fill_values:
            fill_value = env.snake_length / float(board_size * board_size)
        action, next_hidden = _greedy_action(policy, obs, hidden, device, fill_value=fill_value)

        should_scan = (
            step >= scan_step_min
            and step <= scan_step_max
            and score >= scan_score_min
            and (scan_score_max < 0 or score <= scan_score_max)
            and step % stride == 0
        )
        if should_scan:
            snapshot = env._snapshot_state()
            hidden_after_obs = next_hidden.detach().clone()
            prefix_actions = list(baseline_actions)
            for alt_action in (0, 1, 2):
                if alt_action == action:
                    continue
                branch_env = env
                branch_env._restore_state(copy.deepcopy(snapshot))
                branch_hidden = hidden_after_obs.detach().clone()
                branch_actions = prefix_actions + [alt_action]
                branch_obs, _, terminated, truncated, branch_info = branch_env.step(alt_action)
                if terminated or truncated:
                    score_after = int(branch_info.get("score", branch_env.score))
                    result = {
                        "score": score_after,
                        "length": int(branch_info.get("length", score_after + 3)),
                        "reason": str(branch_info.get("reason", "timeout")),
                        "steps": int(branch_info.get("steps", branch_env.total_steps)),
                        "win": score_after >= board_size * board_size - 3,
                    }
                else:
                    result = _finish_rollout(
                        policy=policy,
                        env=branch_env,
                        obs=branch_obs,
                        hidden=branch_hidden,
                        device=device,
                        max_steps=max_steps,
                        actions=branch_actions,
                        use_fill_values=use_fill_values,
                    )
                tested += 1
                branch_env._restore_state(copy.deepcopy(snapshot))
                if result["win"]:
                    record = {
                        "seed": seed,
                        "deviation_step": step,
                        "deviation_score": score,
                        "base_action": action,
                        "alt_action": alt_action,
                        "result": result,
                        "actions": branch_actions,
                    }
                    improved.append(record)
                    print(
                        {
                            "winning_deviation": {
                                key: value for key, value in record.items() if key != "actions"
                            }
                        },
                        flush=True,
                    )
                    if stop_after_improved > 0 and len(improved) >= stop_after_improved:
                        break
            env._restore_state(snapshot)
            if stop_after_improved > 0 and len(improved) >= stop_after_improved:
                break

        baseline_actions.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        hidden = next_hidden.detach()
        if terminated or truncated:
            break

    for record in improved:
        record["baseline_steps"] = baseline["steps"]
        record["improvement"] = int(baseline["steps"]) - int(record["result"]["steps"])
    improved = [
        record
        for record in improved
        if record["improvement"] >= min_improvement
    ]
    return {
        "seed": seed,
        "baseline": baseline,
        "tested": tested,
        "improved": improved,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan safe one-action deviations from a recurrent checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--early-head-max-fill", type=float, default=None)
    parser.add_argument("--residual-policy-head", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=_parse_ints, default=None)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--scan-step-min", type=int, default=0)
    parser.add_argument("--scan-step-max", type=int, default=5000)
    parser.add_argument("--scan-score-min", type=int, default=0)
    parser.add_argument("--scan-score-max", type=int, default=80)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--min-improvement", type=int, default=1)
    parser.add_argument("--stop-after-improved", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    seeds = args.seeds if args.seeds is not None else ([args.seed] if args.seed is not None else None)
    if not seeds:
        raise SystemExit("provide --seed or --seeds")
    policy = _make_policy(
        checkpoint=args.checkpoint,
        board_size=args.board_size,
        hidden_size=args.hidden_size,
        early_head_max_fill=args.early_head_max_fill,
        residual_policy_head=args.residual_policy_head,
        device=args.device,
    )
    results = [
        scan_seed(
            policy=policy,
            board_size=args.board_size,
            seed=seed,
            device=args.device,
            max_steps=args.max_steps,
            use_fill_values=args.early_head_max_fill is not None,
            scan_step_min=args.scan_step_min,
            scan_step_max=args.scan_step_max,
            scan_score_min=args.scan_score_min,
            scan_score_max=args.scan_score_max,
            stride=args.stride,
            min_improvement=args.min_improvement,
            stop_after_improved=args.stop_after_improved,
        )
        for seed in seeds
    ]
    result = results[0] if len(results) == 1 else {"results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    improved_records = [
        record
        for seed_result in results
        for record in seed_result["improved"]
    ]
    print(
        json.dumps(
            {
                "summary": {
                    "seeds": seeds,
                    "tested": sum(int(seed_result["tested"]) for seed_result in results),
                    "improved_count": len(improved_records),
                    "best_improvement": max(
                        (record["improvement"] for record in improved_records),
                        default=None,
                    ),
                    "out": str(args.out),
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
