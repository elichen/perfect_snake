"""Scan a cycle-conditioned checkpoint for Hamiltonian expert disagreements."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cycle_bc_exhaustive import _direction_from_body
from distill.conditioning import augment_observation, conditioning_channels
from distill.expert import expert_action
from distill.model import SnakePolicy, load_policy_state
from snake_env import SnakeEnv


def _make_policy(
    *,
    checkpoint: Path,
    board_size: int,
    network_scale: int,
    head_centered: bool,
    device: str,
    cycle_conditioning: bool,
) -> tuple[SnakePolicy, SnakeEnv]:
    env = SnakeEnv(n=board_size, head_centered=head_centered)
    extra_channels = conditioning_channels(env) if cycle_conditioning else 0
    n_channels = env.observation_space.shape[0] + extra_channels
    policy = SnakePolicy(
        board_size=board_size,
        scale=network_scale,
        n_channels=n_channels,
        aux_flood_fill=False,
        head_centered=head_centered,
    ).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    load_policy_state(policy, state, aux_flood_fill=False, late_head_min_fill=None)
    policy.eval()
    return policy, env


def _sample_state(
    *,
    env: SnakeEnv,
    rng: np.random.Generator,
    min_len: int,
    max_len: int,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    cycle_idx = int(rng.integers(len(env._curriculum_cycles)))
    cycle = env._curriculum_cycles[cycle_idx]
    head_idx = int(rng.integers(len(cycle)))
    snake_len = int(rng.integers(min_len, max_len + 1))
    snake = [cycle[(head_idx + offset) % len(cycle)] for offset in range(snake_len)]

    occupied = set(snake)
    empty = [cell for cell in env._curriculum_cycles[0] if cell not in occupied]
    food = empty[int(rng.integers(len(empty)))] if empty else (-1, -1)

    env.snake = snake
    env.direction = _direction_from_body(snake[0], snake[1])
    env.food_pos = food
    env.score = snake_len - 3
    env.steps_since_food = 0
    env.total_steps = 0
    env._curriculum_cycle = None
    env._curriculum_head_idx = None
    env._obs_history_frames.clear()
    env._action_history.clear()
    env.prev_phi = env._compute_phi()

    action, _ = expert_action(env, cycle, head_idx)
    obs = augment_observation(env._get_observation(), env, cycle_idx)
    metadata = {
        "cycle_idx": cycle_idx,
        "head_idx": head_idx,
        "length": snake_len,
        "score": snake_len - 3,
        "head": list(snake[0]),
        "direction": int(env.direction),
        "food": list(food),
        "expert": int(action),
    }
    return obs, int(action), metadata


@torch.no_grad()
def scan(
    *,
    checkpoint: Path,
    board_size: int,
    network_scale: int,
    head_centered: bool,
    device: str,
    samples: int,
    batch_size: int,
    seed: int,
    min_fill: float,
    max_fill: float,
    max_errors: int,
    progress_every: int,
    cycle_conditioning: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy, env = _make_policy(
        checkpoint=checkpoint,
        board_size=board_size,
        network_scale=network_scale,
        head_centered=head_centered,
        device=device,
        cycle_conditioning=cycle_conditioning,
    )
    board_area = board_size * board_size
    min_len = max(3, int(np.floor(min_fill * board_area)))
    max_len = min(board_area - 1, int(np.ceil(max_fill * board_area)))
    rng = np.random.default_rng(seed)
    errors: list[dict[str, Any]] = []
    correct = 0
    total = 0
    loss_sum = 0.0
    length_error_counts: dict[int, int] = {}
    started_at = time.time()

    while total < samples:
        current_batch = min(batch_size, samples - total)
        obs_batch = []
        target_batch = []
        meta_batch = []
        for _ in range(current_batch):
            obs, target, meta = _sample_state(
                env=env,
                rng=rng,
                min_len=min_len,
                max_len=max_len,
            )
            if not cycle_conditioning:
                obs = obs[: env.observation_space.shape[0]]
            obs_batch.append(obs)
            target_batch.append(target)
            meta_batch.append(meta)

        obs_t = torch.as_tensor(np.stack(obs_batch), dtype=torch.float32, device=device)
        target_t = torch.as_tensor(target_batch, dtype=torch.long, device=device)
        logits, _ = policy(obs_t)
        loss = torch.nn.functional.cross_entropy(logits, target_t, reduction="sum")
        pred = torch.argmax(logits, dim=-1)
        matches = pred == target_t

        correct += int(matches.sum().item())
        loss_sum += float(loss.item())
        pred_cpu = pred.detach().cpu().numpy()
        probs_cpu = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        matches_cpu = matches.detach().cpu().numpy()
        for idx, matched in enumerate(matches_cpu):
            if matched:
                continue
            meta = dict(meta_batch[idx])
            meta["pred"] = int(pred_cpu[idx])
            meta["probs"] = [round(float(p), 8) for p in probs_cpu[idx].tolist()]
            length_error_counts[int(meta["length"])] = length_error_counts.get(int(meta["length"]), 0) + 1
            if len(errors) < max_errors:
                errors.append(meta)

        total += current_batch
        if progress_every > 0 and total % progress_every == 0:
            print(
                {
                    "samples": total,
                    "accuracy": round(correct / max(1, total), 8),
                    "errors": total - correct,
                    "elapsed_sec": round(time.time() - started_at, 1),
                },
                flush=True,
            )

    summary = {
        "checkpoint": str(checkpoint),
        "board_size": board_size,
        "samples": samples,
        "seed": seed,
        "min_fill": min_fill,
        "max_fill": max_fill,
        "cycle_conditioning": cycle_conditioning,
        "min_len": min_len,
        "max_len": max_len,
        "correct": correct,
        "errors": samples - correct,
        "accuracy": float(correct / max(1, samples)),
        "mean_ce_loss": float(loss_sum / max(1, samples)),
        "elapsed_sec": float(time.time() - started_at),
        "length_error_counts": {str(k): v for k, v in sorted(length_error_counts.items())},
    }
    return summary, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan checkpoint against synthetic cycle expert states")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--network-scale", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--head-centered", action="store_true")
    parser.add_argument("--no-cycle-conditioning", action="store_false", dest="cycle_conditioning")
    parser.add_argument("--cycle-conditioning", action="store_true", dest="cycle_conditioning")
    parser.set_defaults(cycle_conditioning=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--min-fill", type=float, default=0.0)
    parser.add_argument("--max-fill", type=float, default=1.0)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--errors-output", type=Path, default=None)
    args = parser.parse_args()

    summary, errors = scan(
        checkpoint=args.checkpoint,
        board_size=args.board_size,
        network_scale=args.network_scale,
        head_centered=args.head_centered,
        device=args.device,
        samples=args.samples,
        batch_size=args.batch_size,
        seed=args.seed,
        min_fill=args.min_fill,
        max_fill=args.max_fill,
        max_errors=args.max_errors,
        progress_every=args.progress_every,
        cycle_conditioning=args.cycle_conditioning,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.errors_output is not None:
        args.errors_output.parent.mkdir(parents=True, exist_ok=True)
        with args.errors_output.open("w", encoding="utf-8") as f:
            for error in errors:
                f.write(json.dumps(error, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        print(json.dumps({"example_errors": errors[: min(5, len(errors))]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
