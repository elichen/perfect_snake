"""Standalone evaluator for distillation checkpoints."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from eval_metrics import summarize_phase_metrics
from snake_env import SnakeEnv

from .conditioning import augment_observation, conditioning_channels, find_cycle_condition
from .model import SnakePolicy, load_policy_state


@torch.no_grad()
def evaluate_policy(
    policy: SnakePolicy,
    *,
    board_size: int,
    episodes: int,
    seed: int,
    deterministic: bool,
    flood_fill: bool,
    head_centered: bool,
    device: str,
    cycle_conditioning: bool = False,
) -> dict:
    perfect_score = board_size * board_size - 3
    scores: list[int] = []
    lengths: list[int] = []
    reasons: list[str] = []

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
        cycle_idx = None
        if cycle_conditioning:
            cycle_idx, _ = find_cycle_condition(env)
        done = False
        info = {}
        while not done:
            obs_eval = obs
            if cycle_conditioning:
                obs_eval = augment_observation(obs, env, cycle_idx)
            obs_t = torch.as_tensor(obs_eval, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = policy(obs_t)
            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                action = int(torch.multinomial(probs[0], num_samples=1).item())
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        score = int(info.get("score", 0))
        length = int(info.get("length", score + 3))
        reason = str(info.get("reason", "unknown"))
        scores.append(score)
        lengths.append(length)
        reasons.append(reason)

    wins = sum(int(score >= perfect_score) for score in scores)
    stats = {
        "episodes": episodes,
        "deterministic": deterministic,
        "wins": wins,
        "win_rate": float(wins / max(1, episodes)),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "min_score": int(min(scores)) if scores else 0,
        "max_score": int(max(scores)) if scores else 0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
    }
    stats.update(
        summarize_phase_metrics(
            scores=scores,
            terminal_lengths=lengths,
            reasons=reasons,
            perfect_score=perfect_score,
            episodes=episodes,
        )
    )
    return stats


@torch.no_grad()
def evaluate_checkpoint(
    *,
    checkpoint_path: str,
    board_size: int,
    episodes: int,
    seed: int,
    deterministic: bool,
    device: str,
    network_scale: int,
    flood_fill: bool,
    aux_flood_fill: bool,
    head_centered: bool,
    late_head_min_fill: float | None,
    cycle_conditioning: bool = False,
) -> dict:
    base_channels = 5 + int(flood_fill)
    n_channels = base_channels
    if cycle_conditioning:
        probe_env = SnakeEnv(
            n=board_size,
            gamma=0.999,
            alpha=0.2,
            seed=seed,
            flood_fill_obs=flood_fill,
            head_centered=head_centered,
        )
        n_channels += conditioning_channels(probe_env)
    policy = SnakePolicy(
        board_size=board_size,
        scale=network_scale,
        n_channels=n_channels,
        aux_flood_fill=aux_flood_fill,
        head_centered=head_centered,
        late_head_min_fill=late_head_min_fill,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    load_policy_state(
        policy,
        state_dict,
        aux_flood_fill=aux_flood_fill,
        late_head_min_fill=late_head_min_fill,
    )
    policy.eval()
    stats = evaluate_policy(
        policy,
        board_size=board_size,
        episodes=episodes,
        seed=seed,
        deterministic=deterministic,
        flood_fill=flood_fill,
        head_centered=head_centered,
        device=device,
        cycle_conditioning=cycle_conditioning,
    )
    stats["checkpoint"] = checkpoint_path
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a distillation checkpoint")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--network-scale", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--flood-fill", action="store_true")
    parser.add_argument("--aux-flood-fill", action="store_true")
    parser.add_argument("--head-centered", action="store_true")
    parser.add_argument("--late-head-min-fill", type=float, default=None)
    parser.add_argument("--cycle-conditioning", action="store_true")
    args = parser.parse_args()

    stats = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        board_size=args.board_size,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
        network_scale=args.network_scale,
        flood_fill=args.flood_fill,
        aux_flood_fill=args.aux_flood_fill,
        head_centered=args.head_centered,
        late_head_min_fill=args.late_head_min_fill,
        cycle_conditioning=args.cycle_conditioning,
    )

    print()
    print("=" * 60)
    print(f"Results for {args.checkpoint}")
    print("=" * 60)
    print(f"  Board size:    {args.board_size}x{args.board_size}")
    print(f"  Perfect score: {args.board_size * args.board_size - 3}")
    print(f"  Episodes:      {stats['episodes']}")
    print(f"  Deterministic: {stats['deterministic']}")
    print()
    print(f"  Win rate:      {stats['win_rate']*100:.1f}% ({stats['wins']}/{stats['episodes']})")
    print(f"  Mean score:    {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
    print(f"  Median score:  {stats['median_score']:.0f}")
    print(f"  Score range:   [{stats['min_score']}, {stats['max_score']}]")
    print(f"  Mean length:   {stats['mean_length']:.1f} steps")
    print()
    print("  --- Phase Buckets ---")
    print(f"  <20%        : {stats['phase_lt20_count']:3d} ({stats['phase_lt20_rate']*100:4.1f}%)")
    print(f"  20-80%      : {stats['phase_20_80_count']:3d} ({stats['phase_20_80_rate']*100:4.1f}%)")
    print(f"  80-95%      : {stats['phase_80_95_count']:3d} ({stats['phase_80_95_rate']*100:4.1f}%)")
    print(f"  95-100%     : {stats['phase_gte95_count']:3d} ({stats['phase_gte95_rate']*100:4.1f}%)")
    print(f"  win         : {stats['win_count']:3d} ({stats['win_rate']*100:4.1f}%)")
    print()
    print("  Death reasons:")
    for reason in ("self", "wall", "stall", "other"):
        count = stats.get(f"death_{reason}_count", 0)
        print(f"    {reason:8s}: {count:3d} ({count/max(1, stats['episodes'])*100:4.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
