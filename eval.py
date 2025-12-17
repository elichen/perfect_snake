"""Evaluate a trained Snake policy checkpoint."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn

from snake_env import SnakeEnv


class SnakePolicy(nn.Module):
    """FC policy for Snake (must match train.py architecture)."""

    def __init__(self, board_size: int, scale: int = 1):
        super().__init__()

        n_channels = 5
        obs_shape = (n_channels, board_size + 2, board_size + 2)
        n_input = int(np.prod(obs_shape))
        n_actions = 3

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

    def forward(self, observations, state=None):
        features = self.features(observations)
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
) -> dict:
    """Load checkpoint and evaluate."""

    # Load policy
    policy = SnakePolicy(board_size, scale=network_scale).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    # Create env
    env = SnakeEnv(n=board_size, gamma=0.99, alpha=0.2, seed=seed)
    perfect_score = board_size * board_size - 3

    scores = []
    wins = 0
    lengths = []

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
        reason = last_info.get("reason", "unknown")
        scores.append(score)
        lengths.append(steps)

        if score >= perfect_score:
            wins += 1

        if verbose:
            win_str = "WIN" if score >= perfect_score else ""
            print(f"  Episode {ep+1:3d}: score={score:3d}/{perfect_score} steps={steps:5d} reason={reason:10s} {win_str}")

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
    print("=" * 60)


if __name__ == "__main__":
    main()
