"""Behavior-clone a cycle-following expert on curriculum-aligned Snake states."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from distill.model import SnakePolicy, freeze_except_late_heads, load_policy_state
from distill.evaluate import evaluate_policy
from snake_env import SnakeEnv


DIR_TO_ACTION = {
    0: 1,  # up -> straight
    1: 2,  # right -> right
    2: 0,  # down is invalid for forward cycle following
    3: 0,  # left -> left
}


def expert_cycle_action(env: SnakeEnv) -> int:
    if env._curriculum_cycle is None or env._curriculum_head_idx is None:
        raise RuntimeError("expert_cycle_action requires a curriculum-aligned state")

    next_idx = (env._curriculum_head_idx - 1) % len(env._curriculum_cycle)
    target = env._curriculum_cycle[next_idx]
    hr, hc = env.snake_head
    tr, tc = target
    dr, dc = tr - hr, tc - hc
    try:
        new_dir = next(
            direction
            for direction, delta in env.DIRECTIONS.items()
            if delta == (dr, dc)
        )
    except StopIteration as exc:
        raise RuntimeError(
            f"expert target is not adjacent: head={env.snake_head} target={target}"
        ) from exc
    delta = (new_dir - env.direction) % 4
    if delta == 3:
        return 0
    if delta == 0:
        return 1
    if delta == 1:
        return 2
    raise RuntimeError(f"expert requires reverse move: head={env.snake_head} target={target}")


def make_curriculum_env(
    *,
    board_size: int,
    flood_fill: bool,
    head_centered: bool,
    min_fill: float,
    max_fill: float,
    seed: int,
) -> SnakeEnv:
    return SnakeEnv(
        n=board_size,
        gamma=0.999,
        alpha=0.2,
        seed=seed,
        flood_fill_obs=flood_fill,
        curriculum_prob=1.0,
        curriculum_min_fill=min_fill,
        curriculum_max_fill=max_fill,
        head_centered=head_centered,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain Snake by cloning a cycle expert")
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--steps", type=int, default=2000, help="Gradient steps")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--network-scale", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--flood-fill", action="store_true")
    parser.add_argument("--aux-flood-fill", action="store_true")
    parser.add_argument("--head-centered", action="store_true")
    parser.add_argument("--min-fill", type=float, default=0.05)
    parser.add_argument("--max-fill", type=float, default=0.98)
    parser.add_argument("--late-head-min-fill", type=float, default=None)
    parser.add_argument("--train-late-head-only", action="store_true")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.train_late_head_only and args.late_head_min_fill is None:
        raise SystemExit("--train-late-head-only requires --late-head-min-fill")

    env = make_curriculum_env(
        board_size=args.board_size,
        flood_fill=args.flood_fill,
        head_centered=args.head_centered,
        min_fill=args.min_fill,
        max_fill=args.max_fill,
        seed=args.seed,
    )
    policy = SnakePolicy(
        board_size=args.board_size,
        scale=args.network_scale,
        n_channels=env.observation_space.shape[0],
        aux_flood_fill=args.aux_flood_fill,
        head_centered=args.head_centered,
        late_head_min_fill=args.late_head_min_fill,
    ).to(args.device)

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        load_policy_state(
            policy,
            state,
            aux_flood_fill=args.aux_flood_fill,
            late_head_min_fill=args.late_head_min_fill,
        )

    if args.train_late_head_only:
        freeze_except_late_heads(policy)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    obs, _ = env.reset(seed=args.seed)
    start = time.time()
    best_mean = float("-inf")

    for step in range(1, args.steps + 1):
        obs_batch = np.zeros(
            (args.batch_size,) + env.observation_space.shape,
            dtype=np.float32,
        )
        act_batch = np.zeros((args.batch_size,), dtype=np.int64)

        filled = 0
        while filled < args.batch_size:
            try:
                action = expert_cycle_action(env)
            except RuntimeError:
                obs, _ = env.reset()
                continue

            obs_batch[filled] = obs
            act_batch[filled] = action
            filled += 1

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=args.device)
        act_t = torch.as_tensor(act_batch, dtype=torch.long, device=args.device)
        logits, _ = policy(obs_t)
        loss = F.cross_entropy(logits, act_t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            stats = evaluate_policy(
                policy,
                board_size=args.board_size,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
                deterministic=True,
                flood_fill=args.flood_fill,
                head_centered=args.head_centered,
                device=args.device,
            )
            elapsed = time.time() - start
            print(
                {
                    "step": step,
                    "loss": round(float(loss.item()), 6),
                    "mean_score": round(stats["mean_score"], 3),
                    "median_score": round(stats["median_score"], 3),
                    "win_rate": round(stats["win_rate"], 4),
                    "elapsed_sec": round(elapsed, 1),
                }
            )
            if stats["mean_score"] >= best_mean:
                best_mean = stats["mean_score"]
                os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
                torch.save(policy.state_dict(), args.save_path)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(policy.state_dict(), args.save_path)
    print({"saved": args.save_path, "best_mean_score": round(best_mean, 3)})


if __name__ == "__main__":
    main()
