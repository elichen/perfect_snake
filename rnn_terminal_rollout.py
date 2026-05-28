"""Print sparse terminal snapshots from a greedy RNN Snake rollout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from distill.rnn_model import SnakeRNNPolicy, load_rnn_policy_state
from play import cleanup_terminal, draw_frame, render_game, setup_terminal
from snake_env import SnakeEnv


def _parse_steps(value: str) -> list[int]:
    steps = [int(part.strip()) for part in value.split(",") if part.strip()]
    if any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError("snapshot steps must be non-negative")
    return sorted(set(steps))


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--early-head-max-fill", type=float, default=None)
    parser.add_argument("--residual-policy-head", action="store_true")
    parser.add_argument("--residual-min-fill", type=float, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=20001)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--snapshot-steps", type=_parse_steps, default=_parse_steps("0,1000,5000,20000,40000"))
    parser.add_argument("--animate", action="store_true", help="render every inference step live in the terminal")
    parser.add_argument("--delay", type=float, default=0.02, help="seconds to sleep between animated frames")
    args = parser.parse_args()

    policy = SnakeRNNPolicy(
        board_size=args.board_size,
        n_channels=5,
        hidden_size=args.hidden_size,
        early_head_max_fill=args.early_head_max_fill,
        residual_policy_head=args.residual_policy_head,
        residual_policy_min_fill=args.residual_min_fill,
    ).to(args.device)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    load_rnn_policy_state(policy, state_dict)
    policy.eval()

    env = SnakeEnv(n=args.board_size, gamma=0.999, alpha=0.2, seed=args.seed)
    perfect_score = args.board_size * args.board_size - 3
    obs, _ = env.reset(seed=args.seed)
    hidden = policy.initial_state(1, args.device)

    snapshot_steps = set(args.snapshot_steps)
    snapshots: dict[int, str] = {}
    if 0 in snapshot_steps:
        snapshots[0] = render_game(env, 0)

    done = False
    info: dict = {}
    step = 0
    action: int | None = None
    frame_height = args.board_size + 8
    if args.animate:
        setup_terminal()
        draw_frame(render_game(env, 0) + "\n\n  Press Ctrl+C to quit", frame_height)
        time.sleep(args.delay * 2)
    try:
        while not done and step < args.max_steps:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            fill_t = None
            if args.early_head_max_fill is not None or args.residual_min_fill is not None:
                fill_t = torch.as_tensor(
                    [env.snake_length / float(args.board_size * args.board_size)],
                    dtype=torch.float32,
                    device=args.device,
                )
            logits, hidden = policy.forward_step(obs_t, hidden, fill_values=fill_t)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            if step in snapshot_steps:
                snapshots[step] = render_game(env, step, action)
            if args.animate:
                footer = "\n\n  Press Ctrl+C to quit"
                if done:
                    footer = f"\n\n  Game Over: {info.get('reason', 'unknown')}\n  Final Score: {info.get('score', env.score)}/{perfect_score}"
                draw_frame(render_game(env, step, action) + footer, frame_height)
                time.sleep(args.delay)
    except KeyboardInterrupt:
        done = True
        info = {
            "reason": "interrupted",
            "score": env.score,
            "length": env.snake_length,
            "steps": step,
        }
    finally:
        if args.animate:
            cleanup_terminal()
            print()

    if not done:
        info = {
            "reason": "max_steps",
            "score": env.score,
            "length": env.snake_length,
            "steps": step,
        }

    score = int(info.get("score", env.score))
    summary = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "board_size": args.board_size,
        "score": score,
        "perfect_score": perfect_score,
        "win": score >= perfect_score,
        "reason": info.get("reason", "unknown"),
        "length": int(info.get("length", env.snake_length)),
        "steps": int(info.get("steps", step)),
        "inference": "greedy RNN policy only; no planner or action override",
        "animated": args.animate,
    }

    print(json.dumps(summary, sort_keys=True))
    if args.animate:
        print(render_game(env, step, action))
        return 0 if summary["win"] else 1

    for snapshot_step in args.snapshot_steps:
        frame = snapshots.get(snapshot_step)
        if frame is None:
            continue
        print()
        print(f"--- terminal snapshot: step {snapshot_step} ---")
        print(frame)
    if step not in snapshots:
        print()
        print(f"--- terminal snapshot: final step {step} ---")
        print(render_game(env, step, action))

    return 0 if summary["win"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
