"""Evaluate a trained Snake policy checkpoint."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snake_env import SnakeEnv


class SnakePolicy(nn.Module):
    """FC policy for Snake (must match train.py architecture)."""

    def __init__(self, board_size: int, scale: int = 1, n_channels: int = 5,
                 aux_flood_fill: bool = False, head_centered: bool = False):
        super().__init__()

        total_channels = n_channels
        n_actions = 3

        self.aux_flood_fill = aux_flood_fill
        self.head_centered = head_centered
        if aux_flood_fill:
            self.encoder_channels = total_channels - 1
        else:
            self.encoder_channels = total_channels

        if head_centered:
            obs_n = 2 * (board_size - 1) + 1  # 39 for board_size=20
        else:
            obs_n = board_size + 2
        obs_shape = (self.encoder_channels, obs_n, obs_n)
        n_input = int(np.prod(obs_shape))
        self.board_size = board_size

        # Scale network width
        w = [1024, 512, 256, 128]
        if scale == 2:
            w = [2048, 1024, 512, 256]
        elif scale == 4:
            w = [4096, 2048, 1024, 512]

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input, w[0]),
            nn.LayerNorm(w[0]),
            nn.ReLU(),
            nn.Linear(w[0], w[1]),
            nn.LayerNorm(w[1]),
            nn.ReLU(),
            nn.Linear(w[1], w[2]),
            nn.LayerNorm(w[2]),
            nn.ReLU(),
            nn.Linear(w[2], w[3]),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(w[3], w[3] // 2),
            nn.ReLU(),
            nn.Linear(w[3] // 2, n_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(w[3], w[3]),
            nn.ReLU(),
            nn.Linear(w[3], w[3] // 2),
            nn.ReLU(),
            nn.Linear(w[3] // 2, 1),
        )

        if aux_flood_fill:
            if head_centered:
                flood_target_n = obs_n
            else:
                flood_target_n = board_size
            self.flood_decoder = nn.Sequential(
                nn.Linear(w[3], w[2]),
                nn.ReLU(),
                nn.Linear(w[2], flood_target_n * flood_target_n),
            )

    def forward(self, observations, state=None):
        obs_input = observations[:, :self.encoder_channels]
        features = self.features(obs_input)
        logits = self.policy_head(features)
        values = self.value_head(features)
        return logits, values


class IterativeBlock(nn.Module):
    """Weight-tied convolutional block (must match train.py)."""

    def __init__(self, channels: int):
        super().__init__()
        n_groups = min(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(n_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(n_groups, channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return torch.relu(out + residual)


class SnakeIterativeCNNPolicy(nn.Module):
    """Weight-tied iterative CNN (must match train.py)."""

    def __init__(self, board_size: int, scale: int = 1, n_channels: int = 5,
                 n_iterations: int = 12, aux_flood_fill: bool = False):
        super().__init__()

        total_channels = n_channels
        n_actions = 3

        if scale >= 4:
            channels = 128
            hidden = 512
        elif scale >= 2:
            channels = 64
            hidden = 256
        else:
            channels = 32
            hidden = 128

        self.n_iterations = n_iterations
        self.aux_flood_fill = aux_flood_fill
        self.channels = channels
        self.encoder_channels = total_channels - 1 if aux_flood_fill else total_channels

        n_groups = min(8, channels)

        self.input_conv = nn.Sequential(
            nn.Conv2d(self.encoder_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(n_groups, channels),
            nn.ReLU(),
        )
        self.iter_block = IterativeBlock(channels)
        self.post_norm = nn.GroupNorm(n_groups, channels)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.policy_head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        if aux_flood_fill:
            self.flood_decoder = nn.Sequential(
                nn.Conv2d(channels, channels // 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels // 2, 1, 1),
            )

    def forward_spatial(self, observations):
        x = self.input_conv(observations)
        for _ in range(self.n_iterations):
            x = self.iter_block(x)
        x = torch.relu(self.post_norm(x))
        return x

    def forward(self, observations, state=None):
        obs_input = observations[:, :self.encoder_channels]
        spatial = self.forward_spatial(obs_input)
        features = self.gap(spatial).flatten(1)
        logits = self.policy_head(features)
        values = self.value_head(features)
        return logits, values


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    board_size: int,
    episodes: int,
    seed: int,
    deterministic: bool,
    device: str,
    network_scale: int = 1,
    verbose: bool = False,
    flood_fill: bool = False,
    iterative_cnn: bool = False,
    n_iterations: int = 12,
    aux_flood_fill: bool = False,
    head_centered: bool = False,
) -> dict:
    """Load checkpoint and evaluate."""

    # Load policy
    n_channels = 6 if flood_fill else 5
    state_dict = torch.load(checkpoint_path, map_location=device)

    if iterative_cnn:
        policy = SnakeIterativeCNNPolicy(
            board_size=board_size, scale=network_scale, n_channels=n_channels,
            n_iterations=n_iterations, aux_flood_fill=aux_flood_fill,
        ).to(device)
        # Filter out flood_decoder keys if loading without aux
        if not aux_flood_fill:
            state_dict = {k: v for k, v in state_dict.items()
                          if not k.startswith('flood_decoder')}
        policy.load_state_dict(state_dict, strict=True)
    else:
        policy = SnakePolicy(board_size, scale=network_scale, n_channels=n_channels,
                             aux_flood_fill=aux_flood_fill, head_centered=head_centered).to(device)
        # Filter out flood_decoder keys if loading without aux
        if not aux_flood_fill:
            state_dict = {k: v for k, v in state_dict.items()
                          if not k.startswith('flood_decoder')}
        policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    # Create env
    env = SnakeEnv(n=board_size, gamma=0.99, alpha=0.2, seed=seed, flood_fill_obs=flood_fill, head_centered=head_centered)
    perfect_score = board_size * board_size - 3

    scores = []
    wins = 0
    lengths = []
    death_lengths = []
    death_reasons = {}

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        last_info = info
        steps = 0

        while not done:
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            logits, _ = policy(obs_t)
            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                action = int(torch.distributions.Categorical(logits=logits).sample().item())
            obs, _, terminated, truncated, last_info = env.step(action)
            done = terminated or truncated
            steps += 1

        score = int(last_info.get("score", 0))
        snake_len = int(last_info.get("length", score + 3))
        reason = last_info.get("reason", "unknown")
        scores.append(score)
        lengths.append(steps)
        death_lengths.append(snake_len)
        death_reasons[reason] = death_reasons.get(reason, 0) + 1

        if score >= perfect_score:
            wins += 1

        if verbose:
            fill_pct = snake_len / (board_size * board_size) * 100
            win_str = "WIN" if score >= perfect_score else ""
            print(f"  Ep {ep+1:3d}: score={score:3d}/{perfect_score}  len={snake_len:3d}  fill={fill_pct:4.1f}%  steps={steps:5d}  {reason:6s} {win_str}")

    board_area = board_size * board_size
    bucket_size = board_area // 10
    bucket_counts = [0] * 10
    for dl in death_lengths:
        bucket = min((dl - 1) // bucket_size, 9)
        bucket_counts[bucket] += 1

    return {
        "checkpoint": checkpoint_path,
        "board_size": board_size,
        "perfect_score": perfect_score,
        "episodes": episodes,
        "deterministic": deterministic,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": int(np.min(scores)),
        "max_score": int(np.max(scores)),
        "win_rate": float(wins / episodes),
        "wins": wins,
        "mean_length": float(np.mean(lengths)),
        "death_lengths": death_lengths,
        "mean_death_length": float(np.mean(death_lengths)),
        "median_death_length": float(np.median(death_lengths)),
        "death_reasons": death_reasons,
        "death_fill_buckets": bucket_counts,
        "board_area": board_area,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Snake checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint file")
    parser.add_argument("--board-size", type=int, default=10, help="Board size (default: 10)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes (default: 100)")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for evaluation")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--network-scale", type=int, default=1, choices=[1, 2, 4], help="Network width multiplier (must match training)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-episode results")
    parser.add_argument("--flood-fill", action="store_true", help="Use flood-fill observation channel")
    parser.add_argument("--iterative-cnn", action="store_true", help="Use iterative CNN policy")
    parser.add_argument("--n-iterations", type=int, default=12, help="Iterations for iterative CNN")
    parser.add_argument("--aux-flood-fill", action="store_true", help="Model was trained with aux flood-fill decoder")
    parser.add_argument("--head-centered", action="store_true", help="Head-centered observation (39x39 for 20x20 board)")
    args = parser.parse_args()

    print(f"Evaluating: {args.checkpoint}")
    print(f"  board_size={args.board_size}, episodes={args.episodes}, deterministic={args.deterministic}")
    print()

    try:
        stats = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            board_size=args.board_size,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=args.deterministic,
            device=args.device,
            network_scale=args.network_scale,
            verbose=args.verbose,
            flood_fill=args.flood_fill,
            iterative_cnn=args.iterative_cnn,
            n_iterations=args.n_iterations,
            aux_flood_fill=args.aux_flood_fill,
            head_centered=args.head_centered,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 60)
    print(f"Results for {args.checkpoint}")
    print("=" * 60)
    print(f"  Board size:    {stats['board_size']}x{stats['board_size']}")
    print(f"  Perfect score: {stats['perfect_score']}")
    print(f"  Episodes:      {stats['episodes']}")
    print(f"  Deterministic: {stats['deterministic']}")
    print()
    print(f"  Win rate:      {stats['win_rate']*100:.1f}% ({stats['wins']}/{stats['episodes']})")
    print(f"  Mean score:    {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
    print(f"  Score range:   [{stats['min_score']}, {stats['max_score']}]")
    print(f"  Mean length:   {stats['mean_length']:.1f} steps")
    print()

    # Death analysis
    board_area = stats["board_area"]
    print(f"  --- Death Analysis ---")
    print(f"  Mean death length:   {stats['mean_death_length']:.1f}/{board_area} ({stats['mean_death_length']/board_area*100:.1f}% fill)")
    print(f"  Median death length: {stats['median_death_length']:.0f}/{board_area} ({stats['median_death_length']/board_area*100:.1f}% fill)")
    print()

    # Death reasons
    print(f"  Death reasons:")
    for reason, count in sorted(stats["death_reasons"].items(), key=lambda x: -x[1]):
        print(f"    {reason:8s}: {count:3d} ({count/stats['episodes']*100:.1f}%)")
    print()

    # Histogram
    bucket_size = board_area // 10
    buckets = stats["death_fill_buckets"]
    max_count = max(buckets) if max(buckets) > 0 else 1
    print(f"  Death length distribution:")
    for i, count in enumerate(buckets):
        lo = i * bucket_size + 1
        hi = (i + 1) * bucket_size
        if i == 9:
            hi = board_area
        bar = "#" * int(count / max_count * 30)
        pct = count / stats["episodes"] * 100
        print(f"    {lo:3d}-{hi:3d} ({lo/board_area*100:4.0f}-{hi/board_area*100:3.0f}%): {bar:30s} {count:3d} ({pct:4.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    main()
