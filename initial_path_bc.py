"""Fine-tune a no-cycle policy on standard-reset Hamiltonian first-lap states."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cycle_bc_exhaustive import _direction_from_body, _save_atomic
from distill.conditioning import find_cycle_condition
from distill.expert import expert_action
from distill.model import SnakePolicy, load_policy_state
from distill.evaluate import evaluate_policy
from snake_env import SnakeEnv


class InitialPathSampler:
    def __init__(
        self,
        *,
        board_size: int,
        head_centered: bool,
        flood_fill: bool,
        seed: int,
        device: str,
    ) -> None:
        self.env = SnakeEnv(
            n=board_size,
            gamma=0.999,
            alpha=0.2,
            seed=seed,
            flood_fill_obs=flood_fill,
            head_centered=head_centered,
        )
        self.env.reset(seed=seed)
        self.board_size = board_size
        self.board_area = board_size * board_size
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.obs_shape = self.env.observation_space.shape
        self.all_cells = [(r, c) for r in range(board_size) for c in range(board_size)]
        self.starts: list[tuple[int, int]] = []
        center = board_size // 2
        for direction in range(4):
            dr, dc = self.env.DIRECTIONS[direction]
            self.env.snake = [(center - idx * dr, center - idx * dc) for idx in range(3)]
            self.env.direction = direction
            cycle_idx, head_idx = find_cycle_condition(self.env)
            if cycle_idx is None or head_idx is None:
                raise RuntimeError(f"standard reset direction {direction} is not cycle-aligned")
            self.starts.append((int(cycle_idx), int(head_idx)))

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        obs_batch = np.empty((batch_size, *self.obs_shape), dtype=np.float32)
        act_batch = np.empty((batch_size,), dtype=np.int64)

        for batch_idx in range(batch_size):
            cycle_idx, start_head_idx = self.starts[int(self.rng.integers(len(self.starts)))]
            cycle = self.env._curriculum_cycles[cycle_idx]
            offset = int(self.rng.integers(len(cycle)))
            head_idx = (start_head_idx - offset) % len(cycle)
            snake = [cycle[(head_idx + body_idx) % len(cycle)] for body_idx in range(3)]
            occupied = set(snake)
            while True:
                food = self.all_cells[int(self.rng.integers(len(self.all_cells)))]
                if food not in occupied:
                    break

            self.env.snake = snake
            self.env.direction = _direction_from_body(snake[0], snake[1])
            self.env.food_pos = food
            self.env.score = 0
            self.env.steps_since_food = 0
            self.env.total_steps = 0
            self.env._curriculum_cycle = None
            self.env._curriculum_head_idx = None
            self.env._obs_history_frames.clear()
            self.env._action_history.clear()
            self.env.prev_phi = self.env._compute_phi()

            action, _ = expert_action(self.env, cycle, head_idx)
            obs_batch[batch_idx] = self.env._get_observation()
            act_batch[batch_idx] = int(action)

        return (
            torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device),
            torch.as_tensor(act_batch, dtype=torch.long, device=self.device),
        )


@torch.no_grad()
def estimate_accuracy(
    *,
    policy: SnakePolicy,
    sampler: InitialPathSampler,
    batches: int,
    batch_size: int,
) -> dict[str, float]:
    total = 0
    correct = 0
    losses: list[float] = []
    for _ in range(batches):
        obs, target = sampler.sample_batch(batch_size)
        logits, _ = policy(obs)
        losses.append(float(F.cross_entropy(logits, target).item()))
        pred = torch.argmax(logits, dim=-1)
        correct += int((pred == target).sum().item())
        total += int(target.numel())
    return {
        "accuracy": float(correct / max(1, total)),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "samples": float(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Initial first-lap BC for no-cycle student")
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10001)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--network-scale", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--head-centered", action="store_true")
    parser.add_argument("--flood-fill", action="store_true")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_sampler = InitialPathSampler(
        board_size=args.board_size,
        head_centered=args.head_centered,
        flood_fill=args.flood_fill,
        seed=args.seed,
        device=args.device,
    )
    eval_sampler = InitialPathSampler(
        board_size=args.board_size,
        head_centered=args.head_centered,
        flood_fill=args.flood_fill,
        seed=args.seed + 1_000_000,
        device=args.device,
    )

    policy = SnakePolicy(
        board_size=args.board_size,
        scale=args.network_scale,
        n_channels=train_sampler.obs_shape[0],
        aux_flood_fill=False,
        head_centered=args.head_centered,
    ).to(args.device)
    state = torch.load(args.resume, map_location="cpu")
    load_policy_state(policy, state, aux_flood_fill=False, late_head_min_fill=None)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    best_accuracy = -1.0
    start = time.time()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    (args.save_path.parent / "run.json").write_text(
        json.dumps({"args": vars(args), "obs_shape": train_sampler.obs_shape}, default=str, indent=2) + "\n"
    )

    for step in range(1, args.steps + 1):
        policy.train()
        obs, target = train_sampler.sample_batch(args.batch_size)
        logits, _ = policy(obs)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            policy.eval()
            acc = estimate_accuracy(
                policy=policy,
                sampler=eval_sampler,
                batches=args.eval_batches,
                batch_size=args.eval_batch_size,
            )
            event: dict[str, Any] = {
                "step": step,
                "train_loss": round(float(loss.item()), 6),
                "eval_loss": round(acc["loss"], 6),
                "eval_accuracy": round(acc["accuracy"], 8),
                "eval_samples": int(acc["samples"]),
                "elapsed_sec": round(time.time() - start, 1),
            }
            if args.eval_episodes > 0:
                stats = evaluate_policy(
                    policy,
                    board_size=args.board_size,
                    episodes=args.eval_episodes,
                    seed=args.eval_seed,
                    deterministic=True,
                    flood_fill=args.flood_fill,
                    head_centered=args.head_centered,
                    device=args.device,
                    cycle_conditioning=False,
                )
                event.update(
                    {
                        "episode_mean": round(float(stats["mean_score"]), 3),
                        "episode_win_rate": round(float(stats["win_rate"]), 6),
                        "episode_max": int(stats["max_score"]),
                    }
                )
            print(event, flush=True)
            if acc["accuracy"] > best_accuracy:
                best_accuracy = acc["accuracy"]
                _save_atomic(policy.state_dict(), args.save_path)
                best_path = args.save_path.with_name(f"{args.save_path.stem}.best_eval.pt")
                _save_atomic(policy.state_dict(), best_path)
                print({"best_checkpoint": str(best_path), "eval_accuracy": round(best_accuracy, 8)}, flush=True)

    _save_atomic(policy.state_dict(), args.save_path)
    print({"saved": str(args.save_path), "best_accuracy": round(best_accuracy, 8)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
