"""Standalone behavior-cloning trainer for expert-distillation experiments."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import numpy as np

from snake_env import SnakeEnv

from .conditioning import augment_observation, conditioning_channels
from .evaluate import evaluate_policy
from .expert import expert_action, find_aligned_cycle
from .model import SnakePolicy, freeze_except_late_heads, load_policy_state


def _save_atomic(payload, path: str) -> None:
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _selector_tuple(stats: dict) -> tuple[float, float, float, float]:
    return (
        float(stats["win_rate"]),
        float(stats["mean_score"]),
        float(stats["phase_gte95_rate"]),
        -float(stats["phase_lt20_rate"]),
    )


def make_env(*, board_size: int, flood_fill: bool, head_centered: bool, seed: int) -> SnakeEnv:
    return SnakeEnv(
        n=board_size,
        gamma=0.999,
        alpha=0.2,
        seed=seed,
        flood_fill_obs=flood_fill,
        head_centered=head_centered,
    )


@dataclass
class DaggerReplayBuffer:
    max_size: int
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    fills: list[float] = field(default_factory=list)
    cursor: int = 0

    def __len__(self) -> int:
        return len(self.actions)

    def add(self, obs: np.ndarray, action: int, fill_fraction: float) -> None:
        obs_value = obs.astype(np.float16, copy=True)
        action_value = int(action)
        fill_value = float(fill_fraction)
        if len(self.actions) < self.max_size:
            self.observations.append(obs_value)
            self.actions.append(action_value)
            self.fills.append(fill_value)
            return
        self.observations[self.cursor] = obs_value
        self.actions[self.cursor] = action_value
        self.fills[self.cursor] = fill_value
        self.cursor = (self.cursor + 1) % self.max_size

    def sample(
        self,
        count: int,
        *,
        priority_fill: float,
        priority_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if count <= 0 or len(self) == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")

        count = min(count, len(self))
        all_indices = list(range(len(self)))
        priority_indices = [idx for idx, fill in enumerate(self.fills) if fill <= priority_fill]
        priority_target = min(count, int(round(count * priority_ratio)))

        chosen: list[int] = []
        if priority_target > 0 and priority_indices:
            priority_count = min(priority_target, len(priority_indices))
            chosen.extend(np.random.choice(priority_indices, size=priority_count, replace=False).tolist())

        remaining = count - len(chosen)
        if remaining > 0:
            chosen_set = set(chosen)
            pool = [idx for idx in all_indices if idx not in chosen_set]
            if len(pool) >= remaining:
                chosen.extend(np.random.choice(pool, size=remaining, replace=False).tolist())
            elif pool:
                chosen.extend(pool)
            elif not chosen:
                chosen.extend(np.random.choice(all_indices, size=remaining, replace=True).tolist())

        chosen = chosen[:count]
        obs_batch = np.stack([self.observations[idx].astype(np.float32) for idx in chosen])
        act_batch = np.asarray([self.actions[idx] for idx in chosen], dtype=np.int64)
        priority_count = sum(int(self.fills[idx] <= priority_fill) for idx in chosen)
        return obs_batch, act_batch, priority_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pure-NN policy by cloning the perfect Hamiltonian expert")
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--network-scale", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--flood-fill", action="store_true")
    parser.add_argument("--aux-flood-fill", action="store_true")
    parser.add_argument("--aux-flood-fill-coef", type=float, default=1.0)
    parser.add_argument("--head-centered", action="store_true")
    parser.add_argument("--cycle-conditioning", action="store_true")
    parser.add_argument("--min-fill", type=float, default=0.0)
    parser.add_argument("--max-fill", type=float, default=1.0)
    parser.add_argument("--late-head-min-fill", type=float, default=None)
    parser.add_argument("--train-late-head-only", action="store_true")
    parser.add_argument("--policy-rollin-prob", type=float, default=0.0)
    parser.add_argument("--policy-rollin-ramp-steps", type=int, default=0)
    parser.add_argument("--policy-rollin-min-fill", type=float, default=0.0)
    parser.add_argument("--policy-rollin-max-fill", type=float, default=1.0)
    parser.add_argument("--dagger-buffer-size", type=int, default=0)
    parser.add_argument("--dagger-mix-ratio", type=float, default=0.0)
    parser.add_argument("--dagger-priority-fill", type=float, default=0.2)
    parser.add_argument("--dagger-priority-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    if args.train_late_head_only and args.late_head_min_fill is None:
        raise SystemExit("--train-late-head-only requires --late-head-min-fill")
    if not (0.0 <= args.min_fill <= args.max_fill <= 1.0):
        raise SystemExit("--min-fill/--max-fill must satisfy 0 <= min <= max <= 1")
    if not (0.0 <= args.policy_rollin_prob <= 1.0):
        raise SystemExit("--policy-rollin-prob must be in [0, 1]")
    if args.policy_rollin_ramp_steps < 0:
        raise SystemExit("--policy-rollin-ramp-steps must be >= 0")
    if not (0.0 <= args.policy_rollin_min_fill <= 1.0):
        raise SystemExit("--policy-rollin-min-fill must be in [0, 1]")
    if not (0.0 <= args.policy_rollin_max_fill <= 1.0):
        raise SystemExit("--policy-rollin-max-fill must be in [0, 1]")
    if args.policy_rollin_min_fill > args.policy_rollin_max_fill:
        raise SystemExit("--policy-rollin-min-fill must be <= --policy-rollin-max-fill")
    if args.dagger_buffer_size < 0:
        raise SystemExit("--dagger-buffer-size must be >= 0")
    if not (0.0 <= args.dagger_mix_ratio <= 1.0):
        raise SystemExit("--dagger-mix-ratio must be in [0, 1]")
    if not (0.0 <= args.dagger_priority_fill <= 1.0):
        raise SystemExit("--dagger-priority-fill must be in [0, 1]")
    if not (0.0 <= args.dagger_priority_ratio <= 1.0):
        raise SystemExit("--dagger-priority-ratio must be in [0, 1]")
    if args.dagger_mix_ratio > 0.0 and args.dagger_buffer_size < 1:
        raise SystemExit("--dagger-mix-ratio requires --dagger-buffer-size >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(
        board_size=args.board_size,
        flood_fill=args.flood_fill,
        head_centered=args.head_centered,
        seed=args.seed,
    )
    n_channels = env.observation_space.shape[0]
    if args.cycle_conditioning:
        n_channels += conditioning_channels(env)
    policy = SnakePolicy(
        board_size=args.board_size,
        scale=args.network_scale,
        n_channels=n_channels,
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
    cycle, head_idx = find_aligned_cycle(env)
    cycle_idx = env._curriculum_cycles.index(cycle) if args.cycle_conditioning else None
    start = time.time()
    best_selector = None
    replay_buffer = DaggerReplayBuffer(max_size=args.dagger_buffer_size) if args.dagger_buffer_size > 0 else None

    for step in range(1, args.steps + 1):
        rollin_scale = 1.0
        if args.policy_rollin_ramp_steps > 0:
            rollin_scale = min(1.0, step / float(args.policy_rollin_ramp_steps))
        step_rollin_prob = args.policy_rollin_prob * rollin_scale
        replay_count = 0
        if replay_buffer is not None and len(replay_buffer) > 0 and args.dagger_mix_ratio > 0.0:
            replay_count = min(int(round(args.batch_size * args.dagger_mix_ratio)), len(replay_buffer))
        fresh_target = args.batch_size - replay_count
        obs_shape = (n_channels,) + env.observation_space.shape[1:]
        obs_batch = np.zeros((args.batch_size,) + obs_shape, dtype=np.float32)
        act_batch = np.zeros((args.batch_size,), dtype=np.int64)
        replay_priority_count = 0

        filled = 0
        while filled < fresh_target:
            fill_fraction = env.snake_length / float(args.board_size * args.board_size)
            action, next_head_idx = expert_action(env, cycle, head_idx)
            obs_roll = obs
            if args.cycle_conditioning:
                obs_roll = augment_observation(obs, env, cycle_idx)
            if args.min_fill <= fill_fraction <= args.max_fill:
                obs_batch[filled] = obs_roll
                act_batch[filled] = action
                filled += 1
                if replay_buffer is not None:
                    replay_buffer.add(obs_roll, action, fill_fraction)

            rollout_action = action
            if (
                step_rollin_prob > 0.0
                and fill_fraction >= args.policy_rollin_min_fill
                and fill_fraction <= args.policy_rollin_max_fill
                and np.random.random() < step_rollin_prob
            ):
                with torch.no_grad():
                    obs_roll_t = torch.as_tensor(obs_roll, dtype=torch.float32, device=args.device).unsqueeze(0)
                    logits, _ = policy(obs_roll_t)
                    rollout_action = int(torch.argmax(logits, dim=-1).item())

            obs, _, terminated, truncated, _ = env.step(rollout_action)
            head_idx = next_head_idx
            if rollout_action != action and not (terminated or truncated):
                try:
                    cycle, head_idx = find_aligned_cycle(env)
                    cycle_idx = env._curriculum_cycles.index(cycle) if args.cycle_conditioning else None
                except RuntimeError:
                    terminated = True
            if terminated or truncated:
                obs, _ = env.reset()
                cycle, head_idx = find_aligned_cycle(env)
                cycle_idx = env._curriculum_cycles.index(cycle) if args.cycle_conditioning else None

        if replay_count > 0 and replay_buffer is not None:
            replay_obs, replay_actions, replay_priority_count = replay_buffer.sample(
                replay_count,
                priority_fill=args.dagger_priority_fill,
                priority_ratio=args.dagger_priority_ratio,
            )
            obs_batch[filled:filled + replay_count] = replay_obs
            act_batch[filled:filled + replay_count] = replay_actions
            filled += replay_count

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=args.device)
        act_t = torch.as_tensor(act_batch, dtype=torch.long, device=args.device)
        logits, _ = policy(obs_t)
        ce_loss = F.cross_entropy(logits, act_t)

        total_loss = ce_loss
        aux_flood_loss = None
        if args.aux_flood_fill:
            enc_input = obs_t[:, :policy.encoder_channels]
            flood_target = obs_t[:, policy.encoder_channels:policy.encoder_channels + 1]
            if not args.head_centered:
                flood_target = flood_target[:, :, 1:-1, 1:-1]
            flood_pred = policy.forward_flood_predict(enc_input)
            aux_flood_loss = F.binary_cross_entropy_with_logits(flood_pred, flood_target)
            total_loss = total_loss + args.aux_flood_fill_coef * aux_flood_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
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
                cycle_conditioning=args.cycle_conditioning,
            )
            selector = _selector_tuple(stats)
            elapsed = time.time() - start
            print(
                {
                    "step": step,
                    "loss": round(float(ce_loss.item()), 6),
                    "aux_flood_loss": None if aux_flood_loss is None else round(float(aux_flood_loss.item()), 6),
                    "mean_score": round(stats["mean_score"], 3),
                    "median_score": round(stats["median_score"], 3),
                    "win_rate": round(stats["win_rate"], 4),
                    "phase_lt20_rate": round(stats["phase_lt20_rate"], 4),
                    "phase_gte95_rate": round(stats["phase_gte95_rate"], 4),
                    "policy_rollin_prob": round(step_rollin_prob, 4),
                    "dagger_buffer_size": 0 if replay_buffer is None else len(replay_buffer),
                    "dagger_replay_count": replay_count,
                    "dagger_priority_replay_count": replay_priority_count,
                    "elapsed_sec": round(elapsed, 1),
                }
            )
            if best_selector is None or selector > best_selector:
                best_selector = selector
                os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
                _save_atomic(policy.state_dict(), args.save_path)
                save_dir = os.path.dirname(args.save_path) or "."
                save_name = os.path.splitext(os.path.basename(args.save_path))[0]
                best_eval_path = os.path.join(save_dir, f"{save_name}.best_eval.pt")
                _save_atomic(policy.state_dict(), best_eval_path)
                print(
                    {
                        "best_eval_checkpoint": best_eval_path,
                        "mean_score": round(stats["mean_score"], 3),
                        "win_rate": round(stats["win_rate"], 4),
                    }
                )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    _save_atomic(policy.state_dict(), args.save_path)
    print({"saved": args.save_path})


if __name__ == "__main__":
    main()
