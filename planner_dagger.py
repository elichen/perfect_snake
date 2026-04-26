"""Planner-conditioned endgame correction with KL anchoring."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from deep_eval import _infer_eval_kwargs, _resolve_target
from eval import SnakePolicy, _load_policy_state, evaluate_checkpoint
from snake_env import SnakeEnv


@dataclass
class CollectedSample:
    episode_seed: int
    episode_index: int
    step: int
    score: int
    fill_ratio: float
    policy_action: int
    planner_action: int
    planner_margin: float
    reference_entropy: float


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_policy_from_checkpoint(
    *,
    checkpoint_path: Path,
    run_dir: Path,
    device: str,
) -> tuple[SnakePolicy, dict[str, Any]]:
    eval_kwargs = _infer_eval_kwargs(run_dir, device=device, num_envs=1)
    base_obs_channels = (
        5
        + int(eval_kwargs["flood_fill"])
        + int(eval_kwargs["body_age_obs"])
        + 3 * int(eval_kwargs["action_history_obs"])
    )
    n_channels = (
        base_obs_channels * int(eval_kwargs["obs_history"])
        + int(eval_kwargs["aux_cycle_target"])
        + int(eval_kwargs["aux_tail_target"])
        + int(eval_kwargs["aux_safe_action_target"])
        + 3 * int(eval_kwargs["aux_safe_action_soft_target"])
        + int(eval_kwargs["aux_body_age_target"])
    )

    policy = SnakePolicy(
        board_size=int(eval_kwargs["board_size"]),
        scale=int(eval_kwargs["network_scale"]),
        n_channels=n_channels,
        aux_flood_fill=bool(eval_kwargs["aux_flood_fill"]),
        aux_cycle_target=bool(eval_kwargs["aux_cycle_target"]),
        aux_tail_target=bool(eval_kwargs["aux_tail_target"]),
        aux_safe_action_target=bool(eval_kwargs["aux_safe_action_target"]),
        aux_safe_action_soft_target=bool(eval_kwargs["aux_safe_action_soft_target"]),
        aux_body_age_target=bool(eval_kwargs["aux_body_age_target"]),
        head_centered=bool(eval_kwargs["head_centered"]),
        late_head_min_fill=eval_kwargs["late_head_min_fill"],
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    _load_policy_state(
        policy,
        state_dict,
        aux_flood_fill=bool(eval_kwargs["aux_flood_fill"]),
        aux_cycle_target=bool(eval_kwargs["aux_cycle_target"]),
        aux_tail_target=bool(eval_kwargs["aux_tail_target"]),
        aux_body_age_target=bool(eval_kwargs["aux_body_age_target"]),
        late_head_min_fill=eval_kwargs["late_head_min_fill"],
    )
    policy.eval()
    return policy, eval_kwargs


def _make_env(eval_kwargs: dict[str, Any], *, seed: int) -> SnakeEnv:
    return SnakeEnv(
        n=int(eval_kwargs["board_size"]),
        gamma=0.99,
        alpha=0.2,
        seed=seed,
        flood_fill_obs=bool(eval_kwargs["flood_fill"]),
        body_age_obs=bool(eval_kwargs["body_age_obs"]),
        obs_history=int(eval_kwargs["obs_history"]),
        action_history_obs=int(eval_kwargs["action_history_obs"]),
        cycle_target_obs=bool(eval_kwargs["aux_cycle_target"]),
        tail_target_obs=bool(eval_kwargs["aux_tail_target"]),
        safe_action_target_obs=bool(eval_kwargs["aux_safe_action_target"]),
        safe_action_soft_target_obs=bool(eval_kwargs["aux_safe_action_soft_target"]),
        body_age_target_obs=bool(eval_kwargs["aux_body_age_target"]),
        cycle_target_min_fill=eval_kwargs["aux_cycle_target_min_fill"],
        safe_action_target_min_fill=float(eval_kwargs["aux_safe_action_target_min_fill"]),
        safe_action_soft_target_min_fill=float(eval_kwargs["aux_safe_action_soft_target_min_fill"]),
        body_age_obs_min_fill=float(eval_kwargs["body_age_obs_min_fill"]),
        body_age_target_min_fill=float(eval_kwargs["aux_body_age_target_min_fill"]),
        safe_action_fill_weight=float(eval_kwargs["aux_safe_action_fill_weight"]),
        safe_action_soft_temperature=float(eval_kwargs["aux_safe_action_soft_temperature"]),
        safe_action_bonus=float(eval_kwargs["safe_action_bonus"]),
        safe_action_bonus_min_fill=float(eval_kwargs["safe_action_bonus_min_fill"]),
        safe_action_bonus_fill_weight=float(eval_kwargs["safe_action_bonus_fill_weight"]),
        head_centered=bool(eval_kwargs["head_centered"]),
    )


def _score_actions(env: SnakeEnv, depth: int, fill_weight: float) -> list[float]:
    if depth <= 1:
        return env.score_relative_actions(fill_weight=fill_weight)

    scores: list[float] = []
    snapshot = env._snapshot_state()
    for action in range(3):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if info.get("reason") == "win":
                score = float("inf")
            else:
                score = float("-inf")
        else:
            score = max(_score_actions(env, depth - 1, fill_weight))
        scores.append(float(score))
        env._restore_state(snapshot)
    return scores


def collect_disagreement_dataset(
    *,
    policy: SnakePolicy,
    eval_kwargs: dict[str, Any],
    device: str,
    episodes: int,
    seed_start: int,
    min_fill: float,
    planner_depth: int,
    planner_fill_weight: float,
    planner_margin_min: float,
    max_samples: int,
    max_samples_per_episode: int,
    min_step_gap: int,
) -> tuple[torch.Tensor, torch.Tensor, list[CollectedSample], dict[str, Any]]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    samples: list[CollectedSample] = []
    total_high_fill_states = 0
    total_disagreements = 0
    score_counter: Counter[int] = Counter()
    margin_values: list[float] = []

    perfect_score = int(eval_kwargs["board_size"]) * int(eval_kwargs["board_size"]) - 3

    with torch.no_grad():
        for ep_idx in range(episodes):
            episode_seed = seed_start + ep_idx
            env = _make_env(eval_kwargs, seed=episode_seed)
            obs, _ = env.reset(seed=episode_seed)
            done = False
            step = 0
            episode_samples = 0
            last_collected_step = -10**9

            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = policy(obs_tensor)
                policy_logits = logits[0]
                policy_action = int(torch.argmax(policy_logits).item())
                fill_ratio = env.snake_length / float(env.n * env.n)

                if fill_ratio >= min_fill:
                    total_high_fill_states += 1
                    planner_scores = _score_actions(env, planner_depth, planner_fill_weight)
                    planner_scores_np = np.asarray(planner_scores, dtype=np.float32)
                    planner_action = int(np.argmax(planner_scores_np))
                    planner_best = float(planner_scores_np[planner_action])
                    if np.isfinite(planner_best):
                        second_best = float(
                            np.max(np.delete(planner_scores_np, planner_action))
                        ) if planner_scores_np.shape[0] > 1 else planner_best
                        planner_margin = planner_best - second_best
                        if planner_action != policy_action and planner_margin >= planner_margin_min:
                            if episode_samples >= max_samples_per_episode:
                                obs, _, terminated, truncated, _ = env.step(policy_action)
                                done = terminated or truncated
                                step += 1
                                continue
                            if step - last_collected_step < min_step_gap:
                                obs, _, terminated, truncated, _ = env.step(policy_action)
                                done = terminated or truncated
                                step += 1
                                continue
                            total_disagreements += 1
                            observations.append(obs.astype(np.float32, copy=True))
                            actions.append(planner_action)
                            score_counter[int(env.score)] += 1
                            margin_values.append(planner_margin)
                            probs = torch.softmax(policy_logits, dim=-1)
                            entropy = float(-(probs * torch.log(probs.clamp_min(1e-8))).sum().item())
                            samples.append(
                                CollectedSample(
                                    episode_seed=episode_seed,
                                    episode_index=ep_idx,
                                    step=step,
                                    score=int(env.score),
                                    fill_ratio=float(fill_ratio),
                                    policy_action=policy_action,
                                    planner_action=planner_action,
                                    planner_margin=float(planner_margin),
                                    reference_entropy=entropy,
                                )
                            )
                            episode_samples += 1
                            last_collected_step = step
                            if len(observations) >= max_samples:
                                done = True
                                break

                obs, _, terminated, truncated, _ = env.step(policy_action)
                done = terminated or truncated
                step += 1

            if len(observations) >= max_samples:
                break

    if not observations:
        empty_obs = torch.empty((0, 0), dtype=torch.float32)
        empty_actions = torch.empty((0,), dtype=torch.long)
    else:
        empty_obs = torch.from_numpy(np.stack(observations)).float()
        empty_actions = torch.as_tensor(actions, dtype=torch.long)

    summary = {
        "episodes_requested": episodes,
        "episodes_consumed": min(episodes, ep_idx + 1 if episodes > 0 else 0),
        "seed_start": seed_start,
        "min_fill": min_fill,
        "planner_depth": planner_depth,
        "planner_fill_weight": planner_fill_weight,
        "planner_margin_min": planner_margin_min,
        "max_samples": max_samples,
        "max_samples_per_episode": max_samples_per_episode,
        "min_step_gap": min_step_gap,
        "collected_samples": int(len(actions)),
        "perfect_score": perfect_score,
        "high_fill_states_seen": int(total_high_fill_states),
        "planner_disagreements_seen": int(total_disagreements),
        "mean_planner_margin": float(np.mean(margin_values)) if margin_values else None,
        "score_histogram": dict(sorted(score_counter.items())),
    }
    return empty_obs, empty_actions, samples, summary


def collect_predeath_trap_dataset(
    *,
    policy: SnakePolicy,
    eval_kwargs: dict[str, Any],
    device: str,
    episodes: int,
    seed_start: int,
    min_fill: float,
    planner_depth: int,
    planner_fill_weight: float,
    planner_margin_min: float,
    max_samples: int,
    step_back: int,
    terminal_score_min: int,
) -> tuple[torch.Tensor, torch.Tensor, list[CollectedSample], dict[str, Any]]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    samples: list[CollectedSample] = []
    score_counter: Counter[int] = Counter()
    terminal_reason_counter: Counter[str] = Counter()
    margin_values: list[float] = []
    qualified_failures = 0
    buffer_len = max(1, step_back + 1)

    perfect_score = int(eval_kwargs["board_size"]) * int(eval_kwargs["board_size"]) - 3

    with torch.no_grad():
        for ep_idx in range(episodes):
            episode_seed = seed_start + ep_idx
            env = _make_env(eval_kwargs, seed=episode_seed)
            obs, _ = env.reset(seed=episode_seed)
            done = False
            step = 0
            recent: deque[dict[str, Any]] = deque(maxlen=buffer_len)
            last_info: dict[str, Any] = {}

            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = policy(obs_tensor)
                policy_logits = logits[0]
                policy_action = int(torch.argmax(policy_logits).item())
                fill_ratio = env.snake_length / float(env.n * env.n)
                planner_action = None
                planner_margin = None

                if fill_ratio >= min_fill and env.score >= terminal_score_min:
                    planner_scores = _score_actions(env, planner_depth, planner_fill_weight)
                    planner_scores_np = np.asarray(planner_scores, dtype=np.float32)
                    best_idx = int(np.argmax(planner_scores_np))
                    best_score = float(planner_scores_np[best_idx])
                    if np.isfinite(best_score):
                        second_best = float(
                            np.max(np.delete(planner_scores_np, best_idx))
                        ) if planner_scores_np.shape[0] > 1 else best_score
                        planner_action = best_idx
                        planner_margin = best_score - second_best

                recent.append(
                    {
                        "obs": obs.astype(np.float32, copy=True),
                        "step": step,
                        "score": int(env.score),
                        "fill_ratio": float(fill_ratio),
                        "policy_action": policy_action,
                        "planner_action": planner_action,
                        "planner_margin": planner_margin,
                        "reference_entropy": float(
                            -(torch.softmax(policy_logits, dim=-1) * torch.log_softmax(policy_logits, dim=-1)).sum().item()
                        ),
                    }
                )

                obs, _, terminated, truncated, last_info = env.step(policy_action)
                done = terminated or truncated
                step += 1

            final_reason = str(last_info.get("reason", "unknown"))
            final_score = int(last_info.get("score", env.score))
            terminal_reason_counter[final_reason] += 1

            if final_reason not in {"self", "wall"}:
                continue
            if final_score < terminal_score_min:
                continue
            if len(recent) < buffer_len:
                continue

            qualified_failures += 1
            candidate = recent[-buffer_len]
            planner_action = candidate["planner_action"]
            policy_action = int(candidate["policy_action"])
            planner_margin = candidate["planner_margin"]
            if planner_action is None:
                continue
            if int(planner_action) == policy_action:
                continue
            if planner_margin is None or float(planner_margin) < planner_margin_min:
                continue

            observations.append(candidate["obs"])
            actions.append(int(planner_action))
            score_counter[int(candidate["score"])] += 1
            margin_values.append(float(planner_margin) if planner_margin is not None else 0.0)
            samples.append(
                CollectedSample(
                    episode_seed=episode_seed,
                    episode_index=ep_idx,
                    step=int(candidate["step"]),
                    score=int(candidate["score"]),
                    fill_ratio=float(candidate["fill_ratio"]),
                    policy_action=policy_action,
                    planner_action=int(planner_action),
                    planner_margin=float(planner_margin) if planner_margin is not None else 0.0,
                    reference_entropy=float(candidate["reference_entropy"]),
                )
            )
            if len(observations) >= max_samples:
                break

    if not observations:
        obs_tensor = torch.empty((0, 0), dtype=torch.float32)
        action_tensor = torch.empty((0,), dtype=torch.long)
    else:
        obs_tensor = torch.from_numpy(np.stack(observations)).float()
        action_tensor = torch.as_tensor(actions, dtype=torch.long)

    summary = {
        "episodes_requested": episodes,
        "episodes_consumed": min(episodes, ep_idx + 1 if episodes > 0 else 0),
        "seed_start": seed_start,
        "collect_mode": "predeath_trap",
        "min_fill": min_fill,
        "planner_depth": planner_depth,
        "planner_fill_weight": planner_fill_weight,
        "planner_margin_min": planner_margin_min,
        "max_samples": max_samples,
        "step_back": step_back,
        "terminal_score_min": terminal_score_min,
        "perfect_score": perfect_score,
        "qualified_failures": int(qualified_failures),
        "collected_samples": int(len(actions)),
        "mean_planner_margin": float(np.mean(margin_values)) if margin_values else None,
        "candidate_score_histogram": dict(sorted(score_counter.items())),
        "terminal_reason_counts": dict(sorted(terminal_reason_counter.items())),
    }
    return obs_tensor, action_tensor, samples, summary


def train_on_planner_corrections(
    *,
    student_policy: SnakePolicy,
    reference_policy: SnakePolicy,
    observations: torch.Tensor,
    planner_actions: torch.Tensor,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    ce_coef: float,
    kl_coef: float,
    grad_clip: float,
    seed: int,
) -> dict[str, Any]:
    if observations.shape[0] == 0:
        return {
            "epochs": 0,
            "steps": 0,
            "last_ce_loss": None,
            "last_kl_loss": None,
            "last_total_loss": None,
        }

    student_policy.train()
    reference_policy.eval()
    optimizer = torch.optim.Adam(student_policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = int(observations.shape[0])
    steps = 0
    last_ce = last_kl = last_total = None

    for epoch in range(epochs):
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            obs_batch = observations[batch_idx].to(device)
            action_batch = planner_actions[batch_idx].to(device)

            student_logits, _ = student_policy(obs_batch)
            with torch.no_grad():
                ref_logits, _ = reference_policy(obs_batch)

            ce_loss = F.cross_entropy(student_logits, action_batch)
            kl_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction="batchmean",
            )
            total_loss = ce_coef * ce_loss + kl_coef * kl_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_policy.parameters(), grad_clip)
            optimizer.step()

            last_ce = float(ce_loss.item())
            last_kl = float(kl_loss.item())
            last_total = float(total_loss.item())
            steps += 1

    student_policy.eval()
    return {
        "epochs": int(epochs),
        "steps": int(steps),
        "last_ce_loss": last_ce,
        "last_kl_loss": last_kl,
        "last_total_loss": last_total,
    }


def _set_trainable_parameters(policy: SnakePolicy, policy_head_only: bool) -> dict[str, int]:
    trainable = frozen = 0
    for name, param in policy.named_parameters():
        should_train = True
        if policy_head_only:
            should_train = name.startswith("policy_head") or name.startswith("late_policy_head")
        param.requires_grad_(should_train)
        if should_train:
            trainable += int(param.numel())
        else:
            frozen += int(param.numel())
    return {"trainable_params": trainable, "frozen_params": frozen}


def main() -> None:
    parser = argparse.ArgumentParser(description="Planner-conditioned endgame correction with KL anchoring")
    parser.add_argument("source", type=str, help="Run dir or checkpoint for incumbent policy")
    parser.add_argument("--checkpoint-name", type=str, default="best_eval.pt")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=183)
    parser.add_argument("--collect-episodes", type=int, default=400)
    parser.add_argument("--collect-seed-start", type=int, default=1)
    parser.add_argument(
        "--collect-mode",
        type=str,
        default="high_fill_disagreement",
        choices=["high_fill_disagreement", "predeath_trap"],
        help="How to collect correction states",
    )
    parser.add_argument("--min-fill", type=float, default=0.85)
    parser.add_argument("--planner-depth", type=int, default=3)
    parser.add_argument("--planner-fill-weight", type=float, default=500.0)
    parser.add_argument("--planner-margin-min", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--max-samples-per-episode", type=int, default=8)
    parser.add_argument("--min-step-gap", type=int, default=3)
    parser.add_argument("--predeath-step-back", type=int, default=3)
    parser.add_argument("--predeath-score-min", type=int, default=390)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--ce-coef", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--policy-head-only", action="store_true", help="Train only policy head parameters to reduce basin drift")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=10001)
    parser.add_argument("--exp-name", type=str, default=None)
    args = parser.parse_args()

    _set_seed(args.seed)

    source_run_dir, source_checkpoint = _resolve_target(args.source, args.checkpoint_name)
    reference_policy, eval_kwargs = _build_policy_from_checkpoint(
        checkpoint_path=source_checkpoint,
        run_dir=source_run_dir,
        device=args.device,
    )
    student_policy, _ = _build_policy_from_checkpoint(
        checkpoint_path=source_checkpoint,
        run_dir=source_run_dir,
        device=args.device,
    )

    timestamp = _timestamp()
    exp_name = args.exp_name or f"planner_dagger_{source_run_dir.name.split('_177')[0]}_d{args.planner_depth}_s{args.seed}"
    run_dir = Path("/Users/elichen/code/perfect_snake/experiments") / f"{exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_run_dir": str(source_run_dir),
        "source_checkpoint": str(source_checkpoint),
        "args": vars(args),
        "eval_kwargs": eval_kwargs,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[planner_dagger] run_dir={run_dir}", flush=True)
    print(f"[planner_dagger] source={source_checkpoint}", flush=True)

    if args.collect_mode == "high_fill_disagreement":
        observations, planner_actions, samples, collect_summary = collect_disagreement_dataset(
            policy=reference_policy,
            eval_kwargs=eval_kwargs,
            device=args.device,
            episodes=args.collect_episodes,
            seed_start=args.collect_seed_start,
            min_fill=args.min_fill,
            planner_depth=args.planner_depth,
            planner_fill_weight=args.planner_fill_weight,
            planner_margin_min=args.planner_margin_min,
            max_samples=args.max_samples,
            max_samples_per_episode=args.max_samples_per_episode,
            min_step_gap=args.min_step_gap,
        )
    else:
        observations, planner_actions, samples, collect_summary = collect_predeath_trap_dataset(
            policy=reference_policy,
            eval_kwargs=eval_kwargs,
            device=args.device,
            episodes=args.collect_episodes,
            seed_start=args.collect_seed_start,
            min_fill=args.min_fill,
            planner_depth=args.planner_depth,
            planner_fill_weight=args.planner_fill_weight,
            planner_margin_min=args.planner_margin_min,
            max_samples=args.max_samples,
            step_back=args.predeath_step_back,
            terminal_score_min=args.predeath_score_min,
        )
    (run_dir / "collection_summary.json").write_text(
        json.dumps(collect_summary, indent=2, sort_keys=True) + "\n"
    )
    with (run_dir / "collection_samples.jsonl").open("w") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), sort_keys=True) + "\n")

    summary_parts = [f"collected={collect_summary['collected_samples']}"]
    if "high_fill_states_seen" in collect_summary:
        summary_parts.append(f"high_fill_seen={collect_summary['high_fill_states_seen']}")
    if "planner_disagreements_seen" in collect_summary:
        summary_parts.append(f"planner_disagreements={collect_summary['planner_disagreements_seen']}")
    if "qualified_failures" in collect_summary:
        summary_parts.append(f"qualified_failures={collect_summary['qualified_failures']}")
    summary_parts.append(f"mean_margin={collect_summary['mean_planner_margin']}")
    print("[planner_dagger] " + " ".join(summary_parts), flush=True)

    if observations.shape[0] == 0:
        print("[planner_dagger] no disagreement samples collected; exiting", flush=True)
        return

    trainable_summary = _set_trainable_parameters(student_policy, args.policy_head_only)
    metadata["trainable"] = trainable_summary
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"[planner_dagger] trainable_params={trainable_summary['trainable_params']} "
        f"frozen_params={trainable_summary['frozen_params']}",
        flush=True,
    )

    train_summary = train_on_planner_corrections(
        student_policy=student_policy,
        reference_policy=reference_policy,
        observations=observations,
        planner_actions=planner_actions,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ce_coef=args.ce_coef,
        kl_coef=args.kl_coef,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )
    (run_dir / "train_summary.json").write_text(
        json.dumps(train_summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[planner_dagger] train steps={train_summary['steps']} "
        f"ce={train_summary['last_ce_loss']} kl={train_summary['last_kl_loss']} "
        f"total={train_summary['last_total_loss']}",
        flush=True,
    )

    checkpoint_path = run_dir / "best_eval.pt"
    torch.save(student_policy.state_dict(), checkpoint_path)
    print(f"[planner_dagger] saved checkpoint={checkpoint_path}", flush=True)

    eval_stats = evaluate_checkpoint(
        checkpoint_path=str(checkpoint_path),
        board_size=int(eval_kwargs["board_size"]),
        episodes=args.eval_episodes,
        seed=args.eval_seed_start,
        deterministic=True,
        device=args.device,
        network_scale=int(eval_kwargs["network_scale"]),
        verbose=False,
        flood_fill=bool(eval_kwargs["flood_fill"]),
        body_age_obs=bool(eval_kwargs["body_age_obs"]),
        obs_history=int(eval_kwargs["obs_history"]),
        action_history_obs=int(eval_kwargs["action_history_obs"]),
        iterative_cnn=bool(eval_kwargs["iterative_cnn"]),
        n_iterations=int(eval_kwargs["n_iterations"]),
        aux_flood_fill=bool(eval_kwargs["aux_flood_fill"]),
        aux_cycle_target=bool(eval_kwargs["aux_cycle_target"]),
        aux_tail_target=bool(eval_kwargs["aux_tail_target"]),
        aux_safe_action_target=bool(eval_kwargs["aux_safe_action_target"]),
        aux_safe_action_soft_target=bool(eval_kwargs["aux_safe_action_soft_target"]),
        aux_body_age_target=bool(eval_kwargs["aux_body_age_target"]),
        late_head_min_fill=eval_kwargs["late_head_min_fill"],
        aux_cycle_target_min_fill=eval_kwargs["aux_cycle_target_min_fill"],
        aux_safe_action_target_min_fill=float(eval_kwargs["aux_safe_action_target_min_fill"]),
        aux_safe_action_soft_target_min_fill=float(eval_kwargs["aux_safe_action_soft_target_min_fill"]),
        body_age_obs_min_fill=float(eval_kwargs["body_age_obs_min_fill"]),
        aux_body_age_target_min_fill=float(eval_kwargs["aux_body_age_target_min_fill"]),
        aux_safe_action_fill_weight=float(eval_kwargs["aux_safe_action_fill_weight"]),
        aux_safe_action_soft_temperature=float(eval_kwargs["aux_safe_action_soft_temperature"]),
        safe_action_bonus=float(eval_kwargs["safe_action_bonus"]),
        safe_action_bonus_min_fill=float(eval_kwargs["safe_action_bonus_min_fill"]),
        safe_action_bonus_fill_weight=float(eval_kwargs["safe_action_bonus_fill_weight"]),
        head_centered=bool(eval_kwargs["head_centered"]),
        num_envs=32,
    )
    (run_dir / "eval_summary.json").write_text(
        json.dumps(eval_stats, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[planner_dagger] eval episodes={args.eval_episodes} "
        f"mean={eval_stats['mean_score']:.2f} std={eval_stats['std_score']:.2f} "
        f"win={eval_stats['win_rate']*100:.2f}% max={eval_stats['max_score']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
