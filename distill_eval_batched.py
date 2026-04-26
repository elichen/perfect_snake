"""Batched evaluator for cycle-conditioned distillation checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from distill.conditioning import augment_observation, conditioning_channels, find_cycle_condition
from distill.model import SnakePolicy, load_policy_state
from eval_metrics import summarize_phase_metrics
from snake_env import SnakeEnv


@torch.no_grad()
def evaluate_batched(
    *,
    checkpoint: Path,
    board_size: int,
    episodes: int,
    seed: int,
    deterministic: bool,
    device: str,
    network_scale: int,
    flood_fill: bool,
    head_centered: bool,
    cycle_conditioning: bool,
    progress_every_steps: int,
) -> dict:
    probe_env = SnakeEnv(
        n=board_size,
        gamma=0.999,
        alpha=0.2,
        seed=seed,
        flood_fill_obs=flood_fill,
        head_centered=head_centered,
    )
    n_channels = probe_env.observation_space.shape[0]
    if cycle_conditioning:
        n_channels += conditioning_channels(probe_env)

    policy = SnakePolicy(
        board_size=board_size,
        scale=network_scale,
        n_channels=n_channels,
        aux_flood_fill=False,
        head_centered=head_centered,
    ).to(device)
    state_dict = torch.load(checkpoint, map_location="cpu")
    load_policy_state(policy, state_dict, aux_flood_fill=False, late_head_min_fill=None)
    policy.eval()

    envs: list[SnakeEnv] = []
    observations: list[np.ndarray | None] = []
    cycle_indices: list[int | None] = []
    done = np.zeros(episodes, dtype=bool)
    scores = np.zeros(episodes, dtype=np.int32)
    lengths = np.zeros(episodes, dtype=np.int32)
    reasons = ["unknown"] * episodes
    steps = np.zeros(episodes, dtype=np.int32)

    for ep in range(episodes):
        env = SnakeEnv(
            n=board_size,
            gamma=0.999,
            alpha=0.2,
            seed=seed + ep,
            flood_fill_obs=flood_fill,
            head_centered=head_centered,
        )
        obs, _ = env.reset(seed=seed + ep)
        envs.append(env)
        observations.append(obs)
        if cycle_conditioning:
            cycle_idx, _ = find_cycle_condition(env)
            cycle_indices.append(cycle_idx)
        else:
            cycle_indices.append(None)

    started_at = time.time()
    total_steps = 0
    last_progress = 0
    while not bool(np.all(done)):
        active_indices = [idx for idx in range(episodes) if not done[idx]]
        batch = []
        for idx in active_indices:
            obs = observations[idx]
            if obs is None:
                raise RuntimeError(f"missing observation for active episode {idx}")
            if cycle_conditioning:
                obs = augment_observation(obs, envs[idx], cycle_indices[idx])
            batch.append(obs)

        obs_t = torch.as_tensor(np.stack(batch), dtype=torch.float32, device=device)
        logits, _ = policy(obs_t)
        if deterministic:
            actions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        else:
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1).detach().cpu().numpy()

        for batch_idx, env_idx in enumerate(active_indices):
            obs, _, terminated, truncated, info = envs[env_idx].step(int(actions[batch_idx]))
            steps[env_idx] += 1
            total_steps += 1
            if terminated or truncated:
                done[env_idx] = True
                observations[env_idx] = None
                scores[env_idx] = int(info.get("score", 0))
                lengths[env_idx] = int(info.get("length", int(scores[env_idx]) + 3))
                reasons[env_idx] = str(info.get("reason", "unknown"))
            else:
                observations[env_idx] = obs

        if (
            progress_every_steps > 0
            and total_steps - last_progress >= progress_every_steps
        ):
            last_progress = total_steps
            completed = int(np.sum(done))
            wins_so_far = int(np.sum(scores >= board_size * board_size - 3))
            print(
                {
                    "completed": completed,
                    "episodes": episodes,
                    "active": episodes - completed,
                    "total_steps": int(total_steps),
                    "wins_so_far": wins_so_far,
                    "elapsed_sec": round(time.time() - started_at, 1),
                },
                flush=True,
            )

    perfect_score = board_size * board_size - 3
    wins = int(np.sum(scores >= perfect_score))
    stats = {
        "checkpoint": str(checkpoint),
        "board_size": board_size,
        "episodes": episodes,
        "deterministic": deterministic,
        "seed": seed,
        "wins": wins,
        "win_rate": float(wins / max(1, episodes)),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "min_score": int(np.min(scores)) if episodes else 0,
        "max_score": int(np.max(scores)) if episodes else 0,
        "std_score": float(np.std(scores)),
        "mean_length": float(np.mean(lengths)),
        "mean_steps": float(np.mean(steps)),
        "max_steps": int(np.max(steps)) if episodes else 0,
        "elapsed_sec": float(time.time() - started_at),
        "scores": [int(v) for v in scores.tolist()],
        "reasons": reasons,
        "steps": [int(v) for v in steps.tolist()],
    }
    stats.update(
        summarize_phase_metrics(
            scores=[int(v) for v in scores.tolist()],
            terminal_lengths=[int(v) for v in lengths.tolist()],
            reasons=reasons,
            perfect_score=perfect_score,
            episodes=episodes,
        )
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Batched distillation checkpoint evaluation")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--network-scale", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--flood-fill", action="store_true")
    parser.add_argument("--head-centered", action="store_true")
    parser.add_argument("--cycle-conditioning", action="store_true")
    parser.add_argument("--progress-every-steps", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    stats = evaluate_batched(
        checkpoint=args.checkpoint,
        board_size=args.board_size,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
        network_scale=args.network_scale,
        flood_fill=args.flood_fill,
        head_centered=args.head_centered,
        cycle_conditioning=args.cycle_conditioning,
        progress_every_steps=args.progress_every_steps,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
