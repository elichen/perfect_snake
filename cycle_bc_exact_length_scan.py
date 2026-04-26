"""Exhaustively scan one snake length for cycle-conditioned expert agreement."""

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


def _build_policy(
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
    policy = SnakePolicy(
        board_size=board_size,
        scale=network_scale,
        n_channels=env.observation_space.shape[0] + extra_channels,
        aux_flood_fill=False,
        head_centered=head_centered,
    ).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    load_policy_state(policy, state, aux_flood_fill=False, late_head_min_fill=None)
    policy.eval()
    return policy, env


def _set_cycle_state(
    *,
    env: SnakeEnv,
    cycle: list[tuple[int, int]],
    cycle_idx: int,
    head_idx: int,
    length: int,
    food: tuple[int, int],
    cycle_conditioning: bool,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    snake = [cycle[(head_idx + offset) % len(cycle)] for offset in range(length)]
    env.snake = snake
    env.direction = _direction_from_body(snake[0], snake[1])
    env.food_pos = food
    env.score = length - 3
    env.steps_since_food = 0
    env.total_steps = 0
    env._curriculum_cycle = None
    env._curriculum_head_idx = None
    env._obs_history_frames.clear()
    env._action_history.clear()
    env.prev_phi = env._compute_phi()
    action, _ = expert_action(env, cycle, head_idx)
    obs = env._get_observation()
    if cycle_conditioning:
        obs = augment_observation(obs, env, cycle_idx)
    meta = {
        "cycle_idx": cycle_idx,
        "head_idx": head_idx,
        "length": length,
        "head": list(snake[0]),
        "direction": int(env.direction),
        "food": list(food),
        "expert": int(action),
    }
    return obs, int(action), meta


@torch.no_grad()
def scan_exact_length(
    *,
    checkpoint: Path,
    board_size: int,
    network_scale: int,
    head_centered: bool,
    device: str,
    length: int,
    batch_size: int,
    max_errors: int,
    progress_every: int,
    cycle_conditioning: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy, env = _build_policy(
        checkpoint=checkpoint,
        board_size=board_size,
        network_scale=network_scale,
        head_centered=head_centered,
        device=device,
        cycle_conditioning=cycle_conditioning,
    )
    all_cells = list(env._curriculum_cycles[0])
    total = 0
    correct = 0
    loss_sum = 0.0
    errors: list[dict[str, Any]] = []
    started_at = time.time()

    obs_batch: list[np.ndarray] = []
    target_batch: list[int] = []
    meta_batch: list[dict[str, Any]] = []

    def flush_batch() -> None:
        nonlocal correct, loss_sum, total, obs_batch, target_batch, meta_batch, errors
        if not obs_batch:
            return
        obs_t = torch.as_tensor(np.stack(obs_batch), dtype=torch.float32, device=device)
        target_t = torch.as_tensor(target_batch, dtype=torch.long, device=device)
        logits, _ = policy(obs_t)
        loss_sum += float(torch.nn.functional.cross_entropy(logits, target_t, reduction="sum").item())
        pred = torch.argmax(logits, dim=-1)
        matches = pred == target_t
        correct += int(matches.sum().item())
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred_cpu = pred.detach().cpu().numpy()
        matches_cpu = matches.detach().cpu().numpy()
        for idx, matched in enumerate(matches_cpu):
            if matched:
                continue
            meta = dict(meta_batch[idx])
            meta["pred"] = int(pred_cpu[idx])
            meta["probs"] = [round(float(p), 8) for p in probs[idx].tolist()]
            if len(errors) < max_errors:
                errors.append(meta)
        total += len(obs_batch)
        if progress_every > 0 and total % progress_every < len(obs_batch):
            print(
                {
                    "samples": total,
                    "accuracy": round(correct / max(1, total), 8),
                    "errors": total - correct,
                    "elapsed_sec": round(time.time() - started_at, 1),
                },
                flush=True,
            )
        obs_batch = []
        target_batch = []
        meta_batch = []

    for cycle_idx, cycle in enumerate(env._curriculum_cycles):
        for head_idx in range(len(cycle)):
            snake = [cycle[(head_idx + offset) % len(cycle)] for offset in range(length)]
            occupied = set(snake)
            for food in all_cells:
                if food in occupied:
                    continue
                obs, target, meta = _set_cycle_state(
                    env=env,
                    cycle=cycle,
                    cycle_idx=cycle_idx,
                    head_idx=head_idx,
                    length=length,
                    food=food,
                    cycle_conditioning=cycle_conditioning,
                )
                obs_batch.append(obs)
                target_batch.append(target)
                meta_batch.append(meta)
                if len(obs_batch) >= batch_size:
                    flush_batch()
    flush_batch()

    summary = {
        "checkpoint": str(checkpoint),
        "board_size": board_size,
        "length": length,
        "cycle_conditioning": cycle_conditioning,
        "samples": total,
        "correct": correct,
        "errors": total - correct,
        "accuracy": float(correct / max(1, total)),
        "mean_ce_loss": float(loss_sum / max(1, total)),
        "elapsed_sec": float(time.time() - started_at),
    }
    return summary, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Exact scan of one on-cycle snake length")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--network-scale", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--head-centered", action="store_true")
    parser.add_argument("--no-cycle-conditioning", action="store_false", dest="cycle_conditioning")
    parser.add_argument("--cycle-conditioning", action="store_true", dest="cycle_conditioning")
    parser.set_defaults(cycle_conditioning=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=250_000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--errors-output", type=Path, default=None)
    args = parser.parse_args()

    summary, errors = scan_exact_length(
        checkpoint=args.checkpoint,
        board_size=args.board_size,
        network_scale=args.network_scale,
        head_centered=args.head_centered,
        device=args.device,
        length=args.length,
        batch_size=args.batch_size,
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
