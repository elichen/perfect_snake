"""Standalone recurrent DAgger trainer for the perfect Hamiltonian expert."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from snake_env import SnakeEnv

from .conditioning import augment_observation, conditioning_channels
from .evaluate_rnn import evaluate_policy
from .expert import expert_action, find_aligned_cycle
from .rnn_model import SnakeRNNPolicy, freeze_except_early_head, load_rnn_policy_state
from rnn_cycle_shortcut_patch import _teacher_action


START_ACTION_TOKEN = 3


def _parse_seed_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("seed list cannot be empty")
    return seeds


def _save_atomic(payload, path: str) -> None:
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _save_json_atomic(payload, path: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _selector_tuple(stats: dict) -> tuple[float, float, float, float]:
    return (
        float(stats["win_rate"]),
        float(stats["mean_score"]),
        float(stats["phase_gte95_rate"]),
        -float(stats["phase_lt20_rate"]),
    )


@dataclass
class EpisodeTrajectory:
    observations: np.ndarray
    actions: np.ndarray
    prev_actions: np.ndarray
    fill_fractions: np.ndarray
    safe_scores: np.ndarray
    score: int
    length: int
    reason: str

    def as_predeath_window(self, window: int) -> "EpisodeTrajectory | None":
        if window < 1 or len(self.actions) < 1:
            return None
        start = max(0, len(self.actions) - window)
        return EpisodeTrajectory(
            observations=self.observations[start:].copy(),
            actions=self.actions[start:].copy(),
            prev_actions=self.prev_actions[start:].copy(),
            fill_fractions=self.fill_fractions[start:].copy(),
            safe_scores=self.safe_scores[start:].copy(),
            score=self.score,
            length=self.length,
            reason=self.reason,
        )


@dataclass(frozen=True)
class PerturbConfig:
    probability: float
    min_fill: float
    max_fill: float
    min_steps: int
    max_steps: int


class TrajectoryPool:
    def __init__(self) -> None:
        self.episodes: list[EpisodeTrajectory] = []
        self.transitions = 0

    def __len__(self) -> int:
        return len(self.episodes)

    def add(self, episode: EpisodeTrajectory | None) -> None:
        if episode is None or len(episode.actions) < 1:
            return
        self.episodes.append(episode)
        self.transitions += int(len(episode.actions))

    def sample_windows(
        self,
        *,
        count: int,
        seq_len: int,
        burn_in: int,
        min_fill: float,
        max_fill: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if count < 1:
            raise RuntimeError("sample_windows requires count >= 1")
        if not self.episodes:
            raise RuntimeError("Cannot sample from an empty trajectory pool")

        sample_episode = self.episodes[0]
        obs_shape = sample_episode.observations.shape[1:]
        obs_batch = np.zeros((seq_len, count) + obs_shape, dtype=np.float32)
        act_batch = np.zeros((seq_len, count), dtype=np.int64)
        prev_batch = np.full((seq_len, count), START_ACTION_TOKEN, dtype=np.int64)
        fill_batch = np.zeros((seq_len, count), dtype=np.float32)
        safe_batch = np.zeros((seq_len, count, 3), dtype=np.float32)
        present_mask = np.zeros((seq_len, count), dtype=np.float32)
        loss_mask = np.zeros((seq_len, count), dtype=np.float32)

        for batch_idx in range(count):
            chosen = None
            for _ in range(8):
                episode = self.episodes[np.random.randint(0, len(self.episodes))]
                if len(episode.actions) < 1:
                    continue
                chosen = episode
                break
            if chosen is None:
                chosen = self.episodes[0]

            length = len(chosen.actions)
            valid_positions = np.where(
                (chosen.fill_fractions >= min_fill) & (chosen.fill_fractions <= max_fill)
            )[0]
            if valid_positions.size > 0:
                anchor = int(valid_positions[np.random.randint(0, valid_positions.size)])
                if length > seq_len:
                    min_start = max(0, anchor - seq_len + 1)
                    max_start = min(anchor, length - seq_len)
                    if min_start <= max_start:
                        start = int(np.random.randint(min_start, max_start + 1))
                    else:
                        start = max(0, min(anchor, length - seq_len))
                else:
                    start = 0
            elif length >= seq_len:
                start = np.random.randint(0, length - seq_len + 1)
            else:
                start = 0
            actual = min(seq_len, length - start)

            obs_batch[:actual, batch_idx] = chosen.observations[start:start + actual]
            act_batch[:actual, batch_idx] = chosen.actions[start:start + actual]
            prev_batch[:actual, batch_idx] = chosen.prev_actions[start:start + actual]
            fill_batch[:actual, batch_idx] = chosen.fill_fractions[start:start + actual]
            safe_batch[:actual, batch_idx] = chosen.safe_scores[start:start + actual]
            present_mask[:actual, batch_idx] = 1.0

            effective_burn = min(burn_in, max(0, actual - 1))
            if actual > 0:
                valid_fill = (
                    (fill_batch[:actual, batch_idx] >= min_fill)
                    & (fill_batch[:actual, batch_idx] <= max_fill)
                )
                loss_mask[effective_burn:actual, batch_idx] = valid_fill[effective_burn:actual].astype(np.float32)

        return obs_batch, act_batch, prev_batch, fill_batch, safe_batch, present_mask, loss_mask


def _make_env(*, board_size: int, flood_fill: bool, head_centered: bool, seed: int) -> SnakeEnv:
    return SnakeEnv(
        n=board_size,
        gamma=0.999,
        alpha=0.2,
        seed=seed,
        flood_fill_obs=flood_fill,
        head_centered=head_centered,
    )


def _prepare_observation(
    obs: np.ndarray,
    env: SnakeEnv,
    *,
    cycle_conditioning: bool,
    cycle_idx: int | None,
) -> np.ndarray:
    if cycle_conditioning:
        return augment_observation(obs, env, cycle_idx)
    return obs.astype(np.float32, copy=False)


def _sample_non_expert_action(expert_action_id: int) -> int:
    choices = [action for action in (0, 1, 2) if action != expert_action_id]
    return int(np.random.choice(choices))


def _collect_episode(
    *,
    policy: SnakeRNNPolicy | None,
    board_size: int,
    flood_fill: bool,
    head_centered: bool,
    device: str,
    seed: int,
    cycle_conditioning: bool,
    use_prev_action_input: bool,
    use_fill_input: bool,
    beta: float,
    teacher_mode: str,
    max_episode_steps: int,
    max_plan_nodes: int,
    max_plan_candidates: int,
    shortcut_score_max: int,
    perturb_config: PerturbConfig | None = None,
) -> EpisodeTrajectory:
    env = _make_env(
        board_size=board_size,
        flood_fill=flood_fill,
        head_centered=head_centered,
        seed=seed,
    )
    obs, _ = env.reset(seed=seed)
    cycle, head_idx = find_aligned_cycle(env)
    cycle_index = {pos: idx for idx, pos in enumerate(cycle)}
    cycle_idx = env._curriculum_cycles.index(cycle) if cycle_conditioning else None
    hidden = policy.initial_state(1, device) if policy is not None else None
    prev_action = START_ACTION_TOKEN

    observations: list[np.ndarray] = []
    actions: list[int] = []
    prev_actions: list[int] = []
    fill_fractions: list[float] = []
    safe_scores: list[np.ndarray] = []

    score = 0
    reason = "unknown"
    done = False
    max_steps = max_episode_steps
    steps = 0
    perturb_remaining = 0

    while not done and steps < max_steps:
        steps += 1
        fill_fraction = env.snake_length / float(board_size * board_size)
        if teacher_mode == "hamiltonian":
            expert, next_head_idx = expert_action(env, cycle, head_idx)
        else:
            expert = _teacher_action(
                env,
                cycle,
                cycle_index,
                teacher_mode,
                max_plan_nodes=max_plan_nodes,
                max_plan_candidates=max_plan_candidates,
                shortcut_score_max=shortcut_score_max,
            )
            next_head_idx = cycle_index.get(_target_head_after_action(env, expert), head_idx)
        obs_eval = _prepare_observation(
            obs,
            env,
            cycle_conditioning=cycle_conditioning,
            cycle_idx=cycle_idx,
        )

        observations.append(obs_eval.astype(np.float32, copy=True))
        actions.append(int(expert))
        prev_actions.append(int(prev_action))
        fill_fractions.append(float(fill_fraction))
        safe_scores.append(np.asarray(env.score_relative_actions(), dtype=np.float32))

        rollout_action = expert
        if (
            perturb_config is not None
            and perturb_remaining == 0
            and perturb_config.min_fill <= fill_fraction <= perturb_config.max_fill
            and np.random.random() < perturb_config.probability
        ):
            perturb_remaining = int(np.random.randint(perturb_config.min_steps, perturb_config.max_steps + 1))

        if perturb_remaining > 0:
            rollout_action = _sample_non_expert_action(expert)
            perturb_remaining -= 1
        elif policy is not None:
            obs_t = torch.as_tensor(obs_eval, dtype=torch.float32, device=device).unsqueeze(0)
            prev_t = None
            if use_prev_action_input:
                prev_t = torch.as_tensor([prev_action], dtype=torch.long, device=device)
            fill_t = None
            if use_fill_input or getattr(policy, "early_head_max_fill", None) is not None:
                fill_t = torch.as_tensor([fill_fraction], dtype=torch.float32, device=device)
            logits, hidden = policy.forward_step(
                obs_t,
                hidden,
                prev_actions=prev_t,
                fill_values=fill_t,
            )
            greedy_action = int(torch.argmax(logits, dim=-1).item())
            if np.random.random() >= beta:
                rollout_action = greedy_action

        obs, _, terminated, truncated, info = env.step(rollout_action)
        score = int(info.get("score", env.snake_length - 3))

        if terminated or truncated:
            reason = str(info.get("reason", "unknown"))
            done = True
        elif rollout_action == expert:
            head_idx = next_head_idx
        else:
            try:
                cycle, head_idx = find_aligned_cycle(env)
                cycle_index = {pos: idx for idx, pos in enumerate(cycle)}
                if cycle_conditioning:
                    cycle_idx = env._curriculum_cycles.index(cycle)
            except RuntimeError:
                reason = "offcycle"
                done = True

        prev_action = int(rollout_action)

    if steps >= max_steps and not done:
        reason = "timeout"

    return EpisodeTrajectory(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        prev_actions=np.asarray(prev_actions, dtype=np.int64),
        fill_fractions=np.asarray(fill_fractions, dtype=np.float32),
        safe_scores=np.asarray(safe_scores, dtype=np.float32),
        score=int(score),
        length=int(env.snake_length),
        reason=str(reason),
    )


def _target_head_after_action(env: SnakeEnv, action: int) -> tuple[int, int]:
    new_dir = (env.direction + {0: -1, 1: 0, 2: 1}[int(action)]) % 4
    dr, dc = env.DIRECTIONS[new_dir]
    hr, hc = env.snake_head
    return hr + dr, hc + dc


def _collect_episodes(
    *,
    target_pool: TrajectoryPool,
    predeath_pool: TrajectoryPool | None,
    episodes: int,
    seed_start: int,
    seed_sequence: list[int] | None,
    policy: SnakeRNNPolicy | None,
    board_size: int,
    flood_fill: bool,
    head_centered: bool,
    device: str,
    cycle_conditioning: bool,
    use_prev_action_input: bool,
    use_fill_input: bool,
    beta: float,
    predeath_window: int,
    teacher_mode: str,
    max_episode_steps: int,
    max_plan_nodes: int,
    max_plan_candidates: int,
    shortcut_score_max: int,
    perturb_config: PerturbConfig | None = None,
) -> int:
    seed_cursor = seed_start
    seeds = seed_sequence if seed_sequence is not None else list(range(seed_start, seed_start + episodes))
    for episode_seed in seeds:
        episode = _collect_episode(
            policy=policy,
            board_size=board_size,
            flood_fill=flood_fill,
            head_centered=head_centered,
            device=device,
            seed=episode_seed,
            cycle_conditioning=cycle_conditioning,
            use_prev_action_input=use_prev_action_input,
            use_fill_input=use_fill_input,
            beta=beta,
            teacher_mode=teacher_mode,
            max_episode_steps=max_episode_steps,
            max_plan_nodes=max_plan_nodes,
            max_plan_candidates=max_plan_candidates,
            shortcut_score_max=shortcut_score_max,
            perturb_config=perturb_config,
        )
        if seed_sequence is None:
            seed_cursor += 1
        target_pool.add(episode)
        if predeath_pool is not None and episode.score < (board_size * board_size - 3):
            predeath_pool.add(episode.as_predeath_window(predeath_window))
    return seed_cursor


def _sample_mixed_batch(
    *,
    expert_pool: TrajectoryPool,
    student_pool: TrajectoryPool,
    predeath_pool: TrajectoryPool,
    batch_size: int,
    seq_len: int,
    burn_in: int,
    min_fill: float,
    max_fill: float,
    student_mix_ratio: float,
    predeath_mix_ratio: float,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    predeath_count = int(round(batch_size * predeath_mix_ratio)) if len(predeath_pool) > 0 else 0
    student_count = int(round(batch_size * student_mix_ratio)) if len(student_pool) > 0 else 0
    if student_count + predeath_count > batch_size:
        overflow = student_count + predeath_count - batch_size
        if predeath_count >= overflow:
            predeath_count -= overflow
        else:
            overflow -= predeath_count
            predeath_count = 0
            student_count = max(0, student_count - overflow)
    expert_count = max(0, batch_size - student_count - predeath_count)
    if expert_count == 0 and len(expert_pool) > 0:
        expert_count = 1
        if student_count > predeath_count and student_count > 0:
            student_count -= 1
        elif predeath_count > 0:
            predeath_count -= 1

    pieces: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    counts = {
        "expert_windows": 0,
        "student_windows": 0,
        "predeath_windows": 0,
    }
    if expert_count > 0:
        pieces.append(
            expert_pool.sample_windows(
                count=expert_count,
                seq_len=seq_len,
                burn_in=burn_in,
                min_fill=min_fill,
                max_fill=max_fill,
            )
        )
        counts["expert_windows"] = expert_count
    if student_count > 0:
        pieces.append(
            student_pool.sample_windows(
                count=student_count,
                seq_len=seq_len,
                burn_in=burn_in,
                min_fill=min_fill,
                max_fill=max_fill,
            )
        )
        counts["student_windows"] = student_count
    if predeath_count > 0:
        pieces.append(
            predeath_pool.sample_windows(
                count=predeath_count,
                seq_len=seq_len,
                burn_in=burn_in,
                min_fill=min_fill,
                max_fill=max_fill,
            )
        )
        counts["predeath_windows"] = predeath_count

    if not pieces:
        raise RuntimeError("No trajectory pools had data for training")

    obs = np.concatenate([piece[0] for piece in pieces], axis=1)
    acts = np.concatenate([piece[1] for piece in pieces], axis=1)
    prevs = np.concatenate([piece[2] for piece in pieces], axis=1)
    fills = np.concatenate([piece[3] for piece in pieces], axis=1)
    safe_scores = np.concatenate([piece[4] for piece in pieces], axis=1)
    present_mask = np.concatenate([piece[5] for piece in pieces], axis=1)
    masks = np.concatenate([piece[6] for piece in pieces], axis=1)
    return {
        "observations": obs,
        "actions": acts,
        "prev_actions": prevs,
        "fill_values": fills,
        "safe_scores": safe_scores,
        "present_mask": present_mask,
        "loss_mask": masks,
    }, counts


def _train_round(
    *,
    policy: SnakeRNNPolicy,
    anchor_policy: SnakeRNNPolicy | None,
    optimizer: torch.optim.Optimizer,
    expert_pool: TrajectoryPool,
    student_pool: TrajectoryPool,
    predeath_pool: TrajectoryPool,
    steps: int,
    batch_size: int,
    seq_len: int,
    burn_in: int,
    min_fill: float,
    max_fill: float,
    student_mix_ratio: float,
    predeath_mix_ratio: float,
    device: str,
    log_every: int,
    safe_target_coef: float,
    safe_target_min_fill: float,
    safe_target_max_fill: float,
    safe_target_temperature: float,
    future_action_horizon: int,
    future_action_coef: float,
    kl_coef: float,
    expert_target_min_fill: float,
    expert_target_max_fill: float,
) -> tuple[dict[str, float], dict[str, int]]:
    last_metrics: dict[str, float] = {}
    last_counts: dict[str, int] = {}

    for step in range(1, steps + 1):
        batch, counts = _sample_mixed_batch(
            expert_pool=expert_pool,
            student_pool=student_pool,
            predeath_pool=predeath_pool,
            batch_size=batch_size,
            seq_len=seq_len,
            burn_in=burn_in,
            min_fill=min_fill,
            max_fill=max_fill,
            student_mix_ratio=student_mix_ratio,
            predeath_mix_ratio=predeath_mix_ratio,
        )
        obs_t = torch.as_tensor(batch["observations"], dtype=torch.float32, device=device)
        act_t = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
        prev_t = torch.as_tensor(batch["prev_actions"], dtype=torch.long, device=device)
        fill_t = torch.as_tensor(batch["fill_values"], dtype=torch.float32, device=device)
        safe_t = torch.as_tensor(batch["safe_scores"], dtype=torch.float32, device=device)
        present_t = torch.as_tensor(batch["present_mask"], dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(batch["loss_mask"], dtype=torch.float32, device=device)

        need_features = future_action_coef > 0.0 and future_action_horizon > 0
        needs_fill_values = policy.fill_input or getattr(policy, "early_head_max_fill", None) is not None
        if need_features:
            logits, _, hidden_states = policy.forward_sequence(
                obs_t,
                prev_actions=prev_t if policy.prev_action_input else None,
                fill_values=fill_t if needs_fill_values else None,
                return_features=True,
            )
        else:
            logits, _ = policy.forward_sequence(
                obs_t,
                prev_actions=prev_t if policy.prev_action_input else None,
                fill_values=fill_t if needs_fill_values else None,
            )
            hidden_states = None

        kl_loss = None
        if kl_coef > 0.0 and anchor_policy is not None:
            with torch.no_grad():
                anchor_logits, _ = anchor_policy.forward_sequence(
                    obs_t,
                    prev_actions=prev_t if anchor_policy.prev_action_input else None,
                    fill_values=fill_t if needs_fill_values else None,
                )
                anchor_log_probs = F.log_softmax(anchor_logits, dim=-1)
                anchor_probs = anchor_log_probs.exp()
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            act_t.reshape(-1),
            reduction="none",
        ).reshape_as(act_t)
        valid = mask_t > 0
        expert_valid = valid & (fill_t >= expert_target_min_fill) & (fill_t <= expert_target_max_fill)
        if torch.any(expert_valid):
            loss = ce[expert_valid].mean()
            accuracy = (torch.argmax(logits, dim=-1)[expert_valid] == act_t[expert_valid]).float().mean()
        elif torch.any(valid):
            loss = ce[valid].mean()
            accuracy = (torch.argmax(logits, dim=-1)[valid] == act_t[valid]).float().mean()
        else:
            loss = logits.sum() * 0.0
            accuracy = torch.zeros((), device=device)

        safe_loss = None
        future_loss = None
        if kl_loss is None and kl_coef > 0.0 and anchor_policy is not None and torch.any(valid):
            current_log_probs = F.log_softmax(logits, dim=-1)
            kl_values = (anchor_probs * (anchor_log_probs - current_log_probs)).sum(dim=-1)
            kl_loss = kl_values[valid].mean()
            loss = loss + kl_coef * kl_loss

        if safe_target_coef > 0.0:
            safe_target, safe_valid = _safe_target_distribution(
                safe_t,
                temperature=safe_target_temperature,
            )
            safe_fill = (fill_t >= safe_target_min_fill) & (fill_t <= safe_target_max_fill)
            safe_valid = safe_valid & safe_fill & valid
            if torch.any(safe_valid):
                log_probs = torch.log_softmax(logits, dim=-1)
                safe_loss = -(safe_target[safe_valid] * log_probs[safe_valid]).sum(dim=-1).mean()
                loss = loss + safe_target_coef * safe_loss

        if need_features and hidden_states is not None:
            future_logits = policy.forward_future_logits(hidden_states)
            future_loss = _future_action_loss(
                future_logits,
                act_t,
                present_t,
                horizon=future_action_horizon,
            )
            if future_loss is not None:
                loss = loss + future_action_coef * future_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        last_metrics = {
            "loss": float(loss.item()),
            "accuracy": float(accuracy.item()),
            "safe_loss": float(safe_loss.item()) if safe_loss is not None else None,
            "future_loss": float(future_loss.item()) if future_loss is not None else None,
            "kl_loss": float(kl_loss.item()) if kl_loss is not None else None,
        }
        last_counts = counts
        if step % log_every == 0 or step == 1 or step == steps:
            print(
                {
                    "train_step": step,
                    "loss": round(last_metrics["loss"], 6),
                    "accuracy": round(last_metrics["accuracy"], 4),
                    "safe_loss": None if last_metrics["safe_loss"] is None else round(last_metrics["safe_loss"], 6),
                    "future_loss": None if last_metrics["future_loss"] is None else round(last_metrics["future_loss"], 6),
                    "kl_loss": None if last_metrics["kl_loss"] is None else round(last_metrics["kl_loss"], 6),
                    **counts,
                }
            )

    if not last_metrics:
        last_metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "safe_loss": None,
            "future_loss": None,
            "kl_loss": None,
        }
    return last_metrics, last_counts


def _parse_beta_schedule(schedule: str, rounds: int) -> list[float]:
    if rounds < 1:
        return []
    if not schedule:
        return [0.0] * rounds
    values = [float(part.strip()) for part in schedule.split(",") if part.strip()]
    if not values:
        return [0.0] * rounds
    if len(values) == 1:
        return values * rounds
    if len(values) < rounds:
        values.extend([values[-1]] * (rounds - len(values)))
    return values[:rounds]


def _safe_target_distribution(
    safe_scores: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(safe_scores)
    valid = finite.any(dim=-1)
    target = torch.zeros_like(safe_scores)
    if not torch.any(valid):
        return target, valid

    inf_mask = torch.isinf(safe_scores) & (safe_scores > 0)
    inf_valid = inf_mask.any(dim=-1)
    if torch.any(inf_valid):
        inf_target = inf_mask[inf_valid].float()
        target[inf_valid] = inf_target / inf_target.sum(dim=-1, keepdim=True)

    finite_only = valid & ~inf_valid
    if torch.any(finite_only):
        finite_scores = safe_scores[finite_only]
        finite_mask = finite[finite_only]
        masked = torch.where(finite_mask, finite_scores, torch.full_like(finite_scores, -1e9))
        probs = torch.softmax(masked / temperature, dim=-1)
        probs = probs * finite_mask.float()
        probs = probs / probs.sum(dim=-1, keepdim=True)
        target[finite_only] = probs
    return target, valid


def _future_action_loss(
    future_logits: torch.Tensor,
    actions: torch.Tensor,
    present_mask: torch.Tensor,
    *,
    horizon: int,
) -> torch.Tensor | None:
    losses = []
    max_h = min(horizon, future_logits.shape[2], max(0, actions.shape[0] - 1))
    for step_ahead in range(1, max_h + 1):
        logits_h = future_logits[:-step_ahead, :, step_ahead - 1, :]
        target_h = actions[step_ahead:]
        valid = (present_mask[:-step_ahead] > 0) & (present_mask[step_ahead:] > 0)
        if torch.any(valid):
            ce = F.cross_entropy(logits_h[valid], target_h[valid], reduction="mean")
            losses.append(ce)
    if not losses:
        return None
    return torch.stack(losses).mean()


def _maybe_eval_and_save(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    short_eval_episodes: int,
    full_eval_episodes: int,
    seed: int,
    device: str,
    flood_fill: bool,
    head_centered: bool,
    cycle_conditioning: bool,
    save_path: str,
    best_selector: tuple[float, float, float, float] | None,
    save_best_eval: bool,
    summary_events: list[dict],
    label: str,
    use_prev_action_input: bool,
    use_fill_input: bool,
) -> tuple[tuple[float, float, float, float] | None, dict, dict | None]:
    short_stats = evaluate_policy(
        policy,
        board_size=board_size,
        episodes=short_eval_episodes,
        seed=seed,
        deterministic=True,
        device=device,
        flood_fill=flood_fill,
        head_centered=head_centered,
        cycle_conditioning=cycle_conditioning,
        use_prev_action_input=use_prev_action_input,
        use_fill_input=use_fill_input,
    )
    selector = _selector_tuple(short_stats)
    full_stats = None
    if save_best_eval and (best_selector is None or selector > best_selector):
        best_selector = selector
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        best_eval_path = os.path.splitext(save_path)[0] + ".best_eval.pt"
        _save_atomic(policy.state_dict(), best_eval_path)
        print(
            {
                "best_eval_checkpoint": best_eval_path,
                "label": label,
                "mean_score": round(short_stats["mean_score"], 3),
                "win_rate": round(short_stats["win_rate"], 4),
            }
        )

    if (
        short_stats["mean_score"] >= 100.0
        or short_stats["phase_lt20_rate"] <= 0.45
    ):
        full_stats = evaluate_policy(
            policy,
            board_size=board_size,
            episodes=full_eval_episodes,
            seed=seed,
            deterministic=True,
            device=device,
            flood_fill=flood_fill,
            head_centered=head_centered,
            cycle_conditioning=cycle_conditioning,
            use_prev_action_input=use_prev_action_input,
            use_fill_input=use_fill_input,
        )

    summary_events.append(
        {
            "label": label,
            "short_eval": short_stats,
            "full_eval": full_stats,
        }
    )
    return best_selector, short_stats, full_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a recurrent pure-NN policy by cloning the perfect Hamiltonian expert")
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000, help="Stage A optimization steps")
    parser.add_argument("--round-steps", type=int, default=750, help="Optimization steps per DAgger round")
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=16)
    parser.add_argument("--offline-episodes", type=int, default=32)
    parser.add_argument("--student-episodes", type=int, default=24, help="Student rollout episodes per DAgger round")
    parser.add_argument("--student-seeds", type=_parse_seed_list, default=None)
    parser.add_argument("--recovery-episodes", type=int, default=0, help="Perturb-and-relabel episodes per DAgger round")
    parser.add_argument("--predeath-window", type=int, default=16)
    parser.add_argument("--student-mix-ratio", type=float, default=0.30)
    parser.add_argument("--predeath-mix-ratio", type=float, default=0.20)
    parser.add_argument("--rollin-beta-schedule", type=str, default="0.2,0.1,0.05,0.0")
    parser.add_argument("--perturb-prob", type=float, default=0.0)
    parser.add_argument("--perturb-min-fill", type=float, default=0.0)
    parser.add_argument("--perturb-max-fill", type=float, default=1.0)
    parser.add_argument("--perturb-min-steps", type=int, default=1)
    parser.add_argument("--perturb-max-steps", type=int, default=1)
    parser.add_argument("--safe-target-coef", type=float, default=0.0)
    parser.add_argument("--safe-target-min-fill", type=float, default=0.0)
    parser.add_argument("--safe-target-max-fill", type=float, default=1.0)
    parser.add_argument("--safe-target-temperature", type=float, default=50.0)
    parser.add_argument(
        "--teacher-mode",
        choices=["hamiltonian", "cycle", "safe", "grid_shortest", "grid_path", "tail_path"],
        default="hamiltonian",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=0,
        help="Maximum rollout steps while collecting teacher episodes. 0 keeps the historical 2*board_area cap.",
    )
    parser.add_argument("--max-plan-nodes", type=int, default=2000)
    parser.add_argument("--max-plan-candidates", type=int, default=64)
    parser.add_argument("--shortcut-score-max", type=int, default=-1)
    parser.add_argument("--future-action-horizon", type=int, default=0)
    parser.add_argument("--future-action-coef", type=float, default=0.0)
    parser.add_argument("--kl-coef", type=float, default=0.0, help="Anchor policy KL penalty against --resume")
    parser.add_argument("--expert-target-min-fill", type=float, default=0.0)
    parser.add_argument("--expert-target-max-fill", type=float, default=1.0)
    parser.add_argument("--early-head-max-fill", type=float, default=None)
    parser.add_argument("--train-early-head-only", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--flood-fill", action="store_true")
    parser.add_argument("--head-centered", action="store_true")
    parser.add_argument("--cycle-conditioning", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--prev-action-input", action="store_true")
    parser.add_argument("--fill-input", action="store_true")
    parser.add_argument("--min-fill", type=float, default=0.0)
    parser.add_argument("--max-fill", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20, help="Full deterministic eval episodes")
    parser.add_argument("--short-eval-episodes", type=int, default=5)
    parser.add_argument("--save-best-eval", action="store_true")
    args = parser.parse_args()

    if not (0.0 <= args.min_fill <= args.max_fill <= 1.0):
        raise SystemExit("--min-fill/--max-fill must satisfy 0 <= min <= max <= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.seq_len < 1:
        raise SystemExit("--seq-len must be >= 1")
    if args.burn_in < 0 or args.burn_in >= args.seq_len:
        raise SystemExit("--burn-in must satisfy 0 <= burn-in < seq-len")
    if args.offline_episodes < 1:
        raise SystemExit("--offline-episodes must be >= 1")
    if args.student_episodes < 1 and args.dagger_rounds > 0:
        raise SystemExit("--student-episodes must be >= 1 when dagger-rounds > 0")
    if args.recovery_episodes < 0:
        raise SystemExit("--recovery-episodes must be >= 0")
    if not (0.0 <= args.student_mix_ratio <= 1.0):
        raise SystemExit("--student-mix-ratio must be in [0, 1]")
    if not (0.0 <= args.predeath_mix_ratio <= 1.0):
        raise SystemExit("--predeath-mix-ratio must be in [0, 1]")
    if args.student_mix_ratio + args.predeath_mix_ratio >= 1.0:
        raise SystemExit("--student-mix-ratio + --predeath-mix-ratio must be < 1")
    if not (0.0 <= args.perturb_prob <= 1.0):
        raise SystemExit("--perturb-prob must be in [0, 1]")
    if not (0.0 <= args.perturb_min_fill <= args.perturb_max_fill <= 1.0):
        raise SystemExit("--perturb-min-fill/--perturb-max-fill must satisfy 0 <= min <= max <= 1")
    if args.perturb_min_steps < 1 or args.perturb_max_steps < args.perturb_min_steps:
        raise SystemExit("--perturb-min-steps/--perturb-max-steps must satisfy 1 <= min <= max")
    if args.safe_target_coef < 0.0:
        raise SystemExit("--safe-target-coef must be >= 0")
    if not (0.0 <= args.safe_target_min_fill <= args.safe_target_max_fill <= 1.0):
        raise SystemExit("--safe-target-min-fill/--safe-target-max-fill must satisfy 0 <= min <= max <= 1")
    if args.safe_target_temperature <= 0.0:
        raise SystemExit("--safe-target-temperature must be > 0")
    if args.max_episode_steps < 0:
        raise SystemExit("--max-episode-steps must be >= 0")
    if args.future_action_horizon < 0:
        raise SystemExit("--future-action-horizon must be >= 0")
    if args.future_action_coef < 0.0:
        raise SystemExit("--future-action-coef must be >= 0")
    if args.kl_coef < 0.0:
        raise SystemExit("--kl-coef must be >= 0")
    if args.kl_coef > 0.0 and not args.resume:
        raise SystemExit("--kl-coef requires --resume")
    if not (0.0 <= args.expert_target_min_fill <= args.expert_target_max_fill <= 1.0):
        raise SystemExit("--expert-target-min-fill/--expert-target-max-fill must satisfy 0 <= min <= max <= 1")
    if args.early_head_max_fill is not None and not (0.0 <= args.early_head_max_fill <= 1.0):
        raise SystemExit("--early-head-max-fill must be in [0, 1]")
    if args.train_early_head_only and args.early_head_max_fill is None:
        raise SystemExit("--train-early-head-only requires --early-head-max-fill")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    probe_env = _make_env(
        board_size=args.board_size,
        flood_fill=args.flood_fill,
        head_centered=args.head_centered,
        seed=args.seed,
    )
    n_channels = probe_env.observation_space.shape[0]
    if args.cycle_conditioning:
        n_channels += conditioning_channels(probe_env)

    policy = SnakeRNNPolicy(
        board_size=args.board_size,
        n_channels=n_channels,
        flood_fill=args.flood_fill,
        head_centered=args.head_centered,
        hidden_size=args.hidden_size,
        prev_action_input=args.prev_action_input,
        fill_input=args.fill_input,
        future_action_horizon=args.future_action_horizon,
        early_head_max_fill=args.early_head_max_fill,
    ).to(args.device)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        load_rnn_policy_state(policy, state)
    anchor_policy = None
    if args.kl_coef > 0.0:
        anchor_policy = SnakeRNNPolicy(
            board_size=args.board_size,
            n_channels=n_channels,
            flood_fill=args.flood_fill,
            head_centered=args.head_centered,
            hidden_size=args.hidden_size,
            prev_action_input=args.prev_action_input,
            fill_input=args.fill_input,
            future_action_horizon=args.future_action_horizon,
            early_head_max_fill=args.early_head_max_fill,
        ).to(args.device)
        load_rnn_policy_state(anchor_policy, state)
        anchor_policy.eval()
        for param in anchor_policy.parameters():
            param.requires_grad_(False)
    if args.train_early_head_only:
        freeze_except_early_head(policy)

    optimizer = torch.optim.Adam((param for param in policy.parameters() if param.requires_grad), lr=args.lr)
    policy.train()

    best_selector: tuple[float, float, float, float] | None = None
    summary_events: list[dict] = []
    start = time.time()
    seed_cursor = args.seed
    max_episode_steps = args.max_episode_steps or (args.board_size * args.board_size * 2)

    expert_pool = TrajectoryPool()
    student_pool = TrajectoryPool()
    predeath_pool = TrajectoryPool()

    seed_cursor = _collect_episodes(
        target_pool=expert_pool,
        predeath_pool=None,
        episodes=args.offline_episodes,
        seed_start=seed_cursor,
        seed_sequence=None,
        policy=None,
        board_size=args.board_size,
        flood_fill=args.flood_fill,
        head_centered=args.head_centered,
        device=args.device,
        cycle_conditioning=args.cycle_conditioning,
        use_prev_action_input=args.prev_action_input,
        use_fill_input=args.fill_input,
        beta=1.0,
        predeath_window=args.predeath_window,
        teacher_mode=args.teacher_mode,
        max_episode_steps=max_episode_steps,
        max_plan_nodes=max(1, args.max_plan_nodes),
        max_plan_candidates=max(1, args.max_plan_candidates),
        shortcut_score_max=args.shortcut_score_max,
        perturb_config=None,
    )
    print(
        {
            "stage": "offline_collect",
            "expert_episodes": len(expert_pool),
            "expert_transitions": expert_pool.transitions,
        }
    )

    stage_metrics, stage_counts = _train_round(
        policy=policy,
        anchor_policy=anchor_policy,
        optimizer=optimizer,
        expert_pool=expert_pool,
        student_pool=student_pool,
        predeath_pool=predeath_pool,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        burn_in=args.burn_in,
        min_fill=args.min_fill,
        max_fill=args.max_fill,
        student_mix_ratio=args.student_mix_ratio,
        predeath_mix_ratio=args.predeath_mix_ratio,
        device=args.device,
        log_every=args.log_every,
        safe_target_coef=args.safe_target_coef,
        safe_target_min_fill=args.safe_target_min_fill,
        safe_target_max_fill=args.safe_target_max_fill,
        safe_target_temperature=args.safe_target_temperature,
        future_action_horizon=args.future_action_horizon,
        future_action_coef=args.future_action_coef,
        kl_coef=args.kl_coef,
        expert_target_min_fill=args.expert_target_min_fill,
        expert_target_max_fill=args.expert_target_max_fill,
    )
    best_selector, short_stats, full_stats = _maybe_eval_and_save(
        policy=policy,
        board_size=args.board_size,
        short_eval_episodes=args.short_eval_episodes,
        full_eval_episodes=args.eval_episodes,
        seed=args.seed + 10_000,
        device=args.device,
        flood_fill=args.flood_fill,
        head_centered=args.head_centered,
        cycle_conditioning=args.cycle_conditioning,
        save_path=args.save_path,
        best_selector=best_selector,
        save_best_eval=args.save_best_eval,
        summary_events=summary_events,
        label="stage_a",
        use_prev_action_input=args.prev_action_input,
        use_fill_input=args.fill_input,
    )
    print(
        {
            "stage": "stage_a",
            "loss": round(stage_metrics["loss"], 6),
            "accuracy": round(stage_metrics["accuracy"], 4),
            "short_mean_score": round(short_stats["mean_score"], 3),
            "short_phase_lt20_rate": round(short_stats["phase_lt20_rate"], 4),
            **stage_counts,
        }
    )

    beta_schedule = _parse_beta_schedule(args.rollin_beta_schedule, args.dagger_rounds)
    perturb_config = None
    if args.perturb_prob > 0.0:
        perturb_config = PerturbConfig(
            probability=args.perturb_prob,
            min_fill=args.perturb_min_fill,
            max_fill=args.perturb_max_fill,
            min_steps=args.perturb_min_steps,
            max_steps=args.perturb_max_steps,
        )
    hard_killed = False
    for round_idx, beta in enumerate(beta_schedule, start=1):
        policy.eval()
        recovery_before = len(student_pool)
        recovery_transitions_before = student_pool.transitions
        if args.recovery_episodes > 0 and perturb_config is not None:
            seed_cursor = _collect_episodes(
                target_pool=student_pool,
                predeath_pool=predeath_pool,
                episodes=args.recovery_episodes,
                seed_start=seed_cursor,
                seed_sequence=None,
                policy=None,
                board_size=args.board_size,
                flood_fill=args.flood_fill,
                head_centered=args.head_centered,
                device=args.device,
                cycle_conditioning=args.cycle_conditioning,
                use_prev_action_input=args.prev_action_input,
                use_fill_input=args.fill_input,
                beta=1.0,
                predeath_window=args.predeath_window,
                teacher_mode=args.teacher_mode,
                max_episode_steps=max_episode_steps,
                max_plan_nodes=max(1, args.max_plan_nodes),
                max_plan_candidates=max(1, args.max_plan_candidates),
                shortcut_score_max=args.shortcut_score_max,
                perturb_config=perturb_config,
            )
            print(
                {
                    "stage": f"recovery_collect_{round_idx}",
                    "recovery_episodes_added": len(student_pool) - recovery_before,
                    "recovery_transitions_added": student_pool.transitions - recovery_transitions_before,
                    "predeath_episodes": len(predeath_pool),
                    "predeath_transitions": predeath_pool.transitions,
                }
            )
        seed_cursor = _collect_episodes(
            target_pool=student_pool,
            predeath_pool=predeath_pool,
            episodes=len(args.student_seeds) if args.student_seeds is not None else args.student_episodes,
            seed_start=seed_cursor,
            seed_sequence=args.student_seeds,
            policy=policy,
            board_size=args.board_size,
            flood_fill=args.flood_fill,
            head_centered=args.head_centered,
            device=args.device,
            cycle_conditioning=args.cycle_conditioning,
            use_prev_action_input=args.prev_action_input,
            use_fill_input=args.fill_input,
            beta=beta,
            predeath_window=args.predeath_window,
            teacher_mode=args.teacher_mode,
            max_episode_steps=max_episode_steps,
            max_plan_nodes=max(1, args.max_plan_nodes),
            max_plan_candidates=max(1, args.max_plan_candidates),
            shortcut_score_max=args.shortcut_score_max,
            perturb_config=None,
        )
        print(
            {
                "stage": f"dagger_collect_{round_idx}",
                "beta": round(beta, 4),
                "student_episodes": len(student_pool),
                "student_transitions": student_pool.transitions,
                "predeath_episodes": len(predeath_pool),
                "predeath_transitions": predeath_pool.transitions,
            }
        )

        policy.train()
        round_metrics, round_counts = _train_round(
            policy=policy,
            anchor_policy=anchor_policy,
            optimizer=optimizer,
            expert_pool=expert_pool,
            student_pool=student_pool,
            predeath_pool=predeath_pool,
            steps=args.round_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            burn_in=args.burn_in,
            min_fill=args.min_fill,
            max_fill=args.max_fill,
            student_mix_ratio=args.student_mix_ratio,
            predeath_mix_ratio=args.predeath_mix_ratio,
            device=args.device,
            log_every=args.log_every,
            safe_target_coef=args.safe_target_coef,
            safe_target_min_fill=args.safe_target_min_fill,
            safe_target_max_fill=args.safe_target_max_fill,
            safe_target_temperature=args.safe_target_temperature,
            future_action_horizon=args.future_action_horizon,
            future_action_coef=args.future_action_coef,
            kl_coef=args.kl_coef,
            expert_target_min_fill=args.expert_target_min_fill,
            expert_target_max_fill=args.expert_target_max_fill,
        )
        best_selector, short_stats, full_stats = _maybe_eval_and_save(
            policy=policy,
            board_size=args.board_size,
            short_eval_episodes=args.short_eval_episodes,
            full_eval_episodes=args.eval_episodes,
            seed=args.seed + 10_000,
            device=args.device,
            flood_fill=args.flood_fill,
            head_centered=args.head_centered,
            cycle_conditioning=args.cycle_conditioning,
            save_path=args.save_path,
            best_selector=best_selector,
            save_best_eval=args.save_best_eval,
            summary_events=summary_events,
            label=f"round_{round_idx}",
            use_prev_action_input=args.prev_action_input,
            use_fill_input=args.fill_input,
        )
        print(
            {
                "stage": f"dagger_round_{round_idx}",
                "beta": round(beta, 4),
                "loss": round(round_metrics["loss"], 6),
                "accuracy": round(round_metrics["accuracy"], 4),
                "short_mean_score": round(short_stats["mean_score"], 3),
                "short_phase_lt20_rate": round(short_stats["phase_lt20_rate"], 4),
                **round_counts,
            }
        )

        if round_idx == 1 and short_stats["mean_score"] < 90.0 and short_stats["phase_lt20_rate"] > 0.60:
            hard_killed = True
            print(
                {
                    "hard_kill": True,
                    "round": round_idx,
                    "mean_score": round(short_stats["mean_score"], 3),
                    "phase_lt20_rate": round(short_stats["phase_lt20_rate"], 4),
                }
            )
            break

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    _save_atomic(policy.state_dict(), args.save_path)
    elapsed = time.time() - start
    summary = {
        "save_path": args.save_path,
        "best_eval_path": os.path.splitext(args.save_path)[0] + ".best_eval.pt",
        "elapsed_sec": elapsed,
        "expert_episodes": len(expert_pool),
        "expert_transitions": expert_pool.transitions,
        "student_episodes": len(student_pool),
        "student_transitions": student_pool.transitions,
        "predeath_episodes": len(predeath_pool),
        "predeath_transitions": predeath_pool.transitions,
        "hard_killed": hard_killed,
        "events": summary_events,
        "args": vars(args),
    }
    summary_path = os.path.splitext(args.save_path)[0] + ".summary.json"
    _save_json_atomic(summary, summary_path)
    print({"saved": args.save_path, "summary": summary_path, "elapsed_sec": round(elapsed, 1)})


if __name__ == "__main__":
    main()
