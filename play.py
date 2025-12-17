"""Watch a trained Snake agent play in the terminal."""

from __future__ import annotations

import argparse
import os
import sys
import time

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


# Direction arrows for display
DIR_ARROWS = {0: "▲", 1: "▶", 2: "▼", 3: "◀"}

# ANSI escape codes
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR_SCREEN = "\033[2J"
HOME = "\033[H"
CLEAR_LINE = "\033[K"


def setup_terminal():
    """Setup terminal for smooth animation."""
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.write(CLEAR_SCREEN)
    sys.stdout.flush()


def cleanup_terminal():
    """Restore terminal state."""
    sys.stdout.write(SHOW_CURSOR)
    sys.stdout.flush()


def draw_frame(content: str, height: int):
    """Draw frame without flickering using cursor positioning."""
    # Move to home position and draw
    sys.stdout.write(HOME)
    lines = content.split("\n")
    for i, line in enumerate(lines):
        sys.stdout.write(line + CLEAR_LINE + "\n")
    # Clear any remaining lines from previous frame
    for _ in range(height - len(lines)):
        sys.stdout.write(CLEAR_LINE + "\n")
    sys.stdout.flush()


def render_game(env: SnakeEnv, step: int, action: int | None = None, value: float | None = None):
    """Render game state to terminal."""
    n = env.n
    snake_set = set(env.snake)
    head = env.snake_head

    # Build display
    lines = []

    # Header
    perfect = n * n - 3
    pct = env.score / perfect * 100 if perfect > 0 else 0
    lines.append(f"  Score: {env.score}/{perfect} ({pct:.1f}%)  Length: {env.snake_length}  Steps: {step}")
    if action is not None:
        action_names = ["←", "↑", "→"]
        lines.append(f"  Action: {action_names[action]}  Direction: {DIR_ARROWS[env.direction]}  Value: {value:.2f}" if value else f"  Action: {action_names[action]}  Direction: {DIR_ARROWS[env.direction]}")
    lines.append("")

    # Top border
    lines.append("  ╔" + "══" * n + "╗")

    # Grid
    for r in range(n):
        row = "  ║"
        for c in range(n):
            pos = (r, c)
            if pos == head:
                # Head with direction indicator
                row += DIR_ARROWS[env.direction] + " "
            elif pos in snake_set:
                row += "██"
            elif pos == env.food_pos:
                row += "◆ "
            else:
                row += "  "
        row += "║"
        lines.append(row)

    # Bottom border
    lines.append("  ╚" + "══" * n + "╝")

    return "\n".join(lines)


@torch.no_grad()
def play_game(
    checkpoint_path: str,
    board_size: int,
    seed: int,
    device: str,
    network_scale: int,
    delay: float,
    deterministic: bool,
) -> dict:
    """Play one game with visualization."""

    # Load policy
    policy = SnakePolicy(board_size, scale=network_scale).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    # Create env
    env = SnakeEnv(n=board_size, gamma=0.99, alpha=0.2, seed=seed)
    perfect_score = board_size * board_size - 3

    obs, info = env.reset(seed=seed)
    done = False
    step = 0

    # Calculate frame height for clearing
    frame_height = board_size + 8  # grid + borders + header + footer

    setup_terminal()

    try:
        # Initial frame
        content = render_game(env, step) + "\n\n  Press Ctrl+C to quit"
        draw_frame(content, frame_height)
        time.sleep(delay * 2)

        while not done:
            # Get action
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            logits, values = policy(obs_t)

            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                action = int(torch.distributions.Categorical(logits=logits).sample().item())

            value = values.item()

            # Step
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            # Render
            if done:
                reason = info.get("reason", "unknown")
                score = info.get("score", 0)
                win = score >= perfect_score

                footer = "\n"
                if win:
                    footer += "  🎉 PERFECT GAME! 🎉\n"
                else:
                    footer += f"  Game Over: {reason}\n"
                footer += f"  Final Score: {score}/{perfect_score}"
                content = render_game(env, step, action, value) + footer
            else:
                content = render_game(env, step, action, value) + "\n\n  Press Ctrl+C to quit"

            draw_frame(content, frame_height)
            time.sleep(delay)

    except KeyboardInterrupt:
        cleanup_terminal()
        print("\n  Interrupted by user")
        return {"interrupted": True}

    cleanup_terminal()
    # Reprint final state so it stays visible
    print(content)

    return {
        "score": int(info.get("score", 0)),
        "steps": step,
        "win": info.get("score", 0) >= perfect_score,
        "reason": info.get("reason", "unknown"),
    }


def main():
    parser = argparse.ArgumentParser(description="Watch a trained Snake agent play")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint file")
    parser.add_argument("--board-size", type=int, default=10, help="Board size (default: 10)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--network-scale", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between frames in seconds")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy (default: deterministic)")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(time.time() * 1000) % 100000

    # Ensure cursor is restored on exit
    import atexit
    atexit.register(cleanup_terminal)

    wins = 0
    total_score = 0
    games_played = 0

    try:
        for game in range(args.games):
            seed = args.seed + game

            if args.games > 1 and game > 0:
                print(f"\n  Next game in 2 seconds... (Game {game + 1}/{args.games})")
                time.sleep(2)

            result = play_game(
                checkpoint_path=args.checkpoint,
                board_size=args.board_size,
                seed=seed,
                device=args.device,
                network_scale=args.network_scale,
                delay=args.delay,
                deterministic=not args.stochastic,
            )

            if result.get("interrupted"):
                break

            games_played += 1
            if result.get("win"):
                wins += 1
            total_score += result.get("score", 0)

    finally:
        cleanup_terminal()

    if games_played > 1:
        print(f"\n{'='*40}")
        print(f"  Games: {games_played}")
        print(f"  Wins: {wins} ({wins/games_played*100:.1f}%)")
        print(f"  Avg Score: {total_score/games_played:.1f}")
        print(f"{'='*40}")


if __name__ == "__main__":
    main()
