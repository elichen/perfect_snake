"""Offline correction on harvested near-terminal trap states."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deep_eval import _resolve_target
from eval import evaluate_checkpoint
from planner_dagger import (
    CollectedSample,
    _build_policy_from_checkpoint,
    _make_env,
    _score_actions,
    _set_trainable_parameters,
)


ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _as_cell(value: Any) -> tuple[int, int]:
    return (int(value[0]), int(value[1]))


def _expected_next_head(record: dict[str, Any]) -> tuple[int, int]:
    direction = int(record["direction"])
    action = int(record["action"])
    delta = {0: -1, 1: 0, 2: 1}[action]
    new_dir = (direction + delta) % 4
    dr, dc = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}[new_dir]
    hr, hc = _as_cell(record["head"])
    return (hr + dr, hc + dc)


def _reconstruct_history_snakes(row: dict[str, Any]) -> list[list[tuple[int, int]]] | None:
    history = row.get("history") or []
    if not history:
        return None

    snakes: list[list[tuple[int, int]] | None] = [None] * len(history)
    snakes[-1] = [_as_cell(cell) for cell in row["terminal_snake"]]

    last = history[-1]
    if snakes[-1][0] != _as_cell(last["head"]):
        return None
    if len(snakes[-1]) != int(last["length"]):
        return None

    for idx in range(len(history) - 2, -1, -1):
        current = history[idx]
        nxt = history[idx + 1]
        next_snake = snakes[idx + 1]
        if next_snake is None:
            return None

        current_len = int(current["length"])
        next_len = int(nxt["length"])
        expected_next_head = _expected_next_head(current)
        if expected_next_head != _as_cell(nxt["head"]):
            return None

        if next_len == current_len + 1:
            current_snake = list(next_snake[1:])
        elif next_len == current_len:
            current_snake = list(next_snake[1:]) + [_as_cell(current["tail"])]
        else:
            return None

        if len(current_snake) != current_len:
            return None
        if current_snake[0] != _as_cell(current["head"]):
            return None
        snakes[idx] = current_snake

    return [list(snake) for snake in snakes if snake is not None]


def _obs_for_reconstructed_state(
    *,
    eval_kwargs: dict[str, Any],
    history_record: dict[str, Any],
    snake: list[tuple[int, int]],
    seed: int,
) -> np.ndarray:
    env = _make_env(eval_kwargs, seed=seed)
    env.reset(seed=seed)
    env.snake = list(snake)
    env.direction = int(history_record["direction"])
    env.food_pos = _as_cell(history_record["food"])
    env.score = int(history_record["score"])
    env.steps_since_food = 0
    env.total_steps = 0
    env._curriculum_cycle = None
    env._curriculum_head_idx = None
    env._obs_history_frames.clear()
    env._action_history.clear()
    env.prev_phi = env._compute_phi()
    return env._get_observation().astype(np.float32, copy=True)


def _planner_label_for_state(
    *,
    eval_kwargs: dict[str, Any],
    history_record: dict[str, Any],
    snake: list[tuple[int, int]],
    seed: int,
    planner_depth: int,
    planner_fill_weight: float,
) -> tuple[int | None, float | None, list[float]]:
    env = _make_env(eval_kwargs, seed=seed)
    env.reset(seed=seed)
    env.snake = list(snake)
    env.direction = int(history_record["direction"])
    env.food_pos = _as_cell(history_record["food"])
    env.score = int(history_record["score"])
    env.steps_since_food = 0
    env.total_steps = 0
    env._curriculum_cycle = None
    env._curriculum_head_idx = None
    env._obs_history_frames.clear()
    env._action_history.clear()
    env.prev_phi = env._compute_phi()

    scores = _score_actions(env, planner_depth, planner_fill_weight)
    score_arr = np.asarray(scores, dtype=np.float32)
    finite_mask = np.isfinite(score_arr)
    if not np.any(finite_mask):
        return None, None, scores
    action = int(np.nanargmax(score_arr))
    best = float(score_arr[action])
    finite_scores = score_arr[finite_mask]
    if finite_scores.size <= 1:
        margin = float("inf")
    else:
        sorted_scores = np.sort(finite_scores)
        margin = float(sorted_scores[-1] - sorted_scores[-2])
    return action, margin, scores


def _safe_label_from_record(record: dict[str, Any]) -> tuple[int | None, float | None]:
    safe_best = record.get("safe_best_action")
    if safe_best is None:
        return None, None
    margin = record.get("safe_margin")
    if margin is not None:
        return int(safe_best), float(margin)

    scores = record.get("safe_scores") or []
    finite = [float(score) for score in scores if score is not None]
    if not finite:
        return int(safe_best), None
    if len(finite) == 1:
        return int(safe_best), float("inf")
    finite.sort()
    return int(safe_best), float(finite[-1] - finite[-2])


def collect_from_failures(
    *,
    failures_path: Path,
    reference_policy: torch.nn.Module,
    eval_kwargs: dict[str, Any],
    device: str,
    seed: int,
    step_back: int,
    terminal_score_min: int,
    terminal_score_max: int,
    candidate_score_min: int,
    target_source: str,
    planner_depth: int,
    planner_fill_weight: float,
    planner_margin_min: float,
    max_samples: int,
    preserve_score_min: int,
    max_preserve_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, list[CollectedSample], dict[str, Any]]:
    observations: list[np.ndarray] = []
    preserve_observations: list[np.ndarray] = []
    actions: list[int] = []
    samples: list[CollectedSample] = []
    reason_counts: Counter[str] = Counter()
    score_counts: Counter[int] = Counter()
    candidate_score_counts: Counter[int] = Counter()
    skipped: Counter[str] = Counter()
    margins: list[float] = []

    with failures_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    with torch.no_grad():
        for row_idx, row in enumerate(rows):
            terminal_score = int(row.get("score", 0))
            reason = str(row.get("reason", "unknown"))
            if reason not in {"self", "wall"}:
                skipped["reason"] += 1
                continue
            if terminal_score < terminal_score_min or terminal_score > terminal_score_max:
                skipped["terminal_score"] += 1
                continue

            history = row.get("history") or []
            target_idx = len(history) - 1 - step_back
            if target_idx < 0:
                skipped["history_too_short"] += 1
                continue

            snakes = _reconstruct_history_snakes(row)
            if snakes is None or len(snakes) != len(history):
                skipped["reconstruct"] += 1
                continue

            for hist_idx, hist_record in enumerate(history):
                if len(preserve_observations) >= max_preserve_samples:
                    break
                if int(hist_record["score"]) < preserve_score_min:
                    continue
                preserve_observations.append(
                    _obs_for_reconstructed_state(
                        eval_kwargs=eval_kwargs,
                        history_record=hist_record,
                        snake=snakes[hist_idx],
                        seed=seed + row_idx,
                    )
                )

            record = history[target_idx]
            if int(record["score"]) < candidate_score_min:
                skipped["candidate_score"] += 1
                continue

            if target_source == "safe":
                label_action, margin = _safe_label_from_record(record)
            else:
                label_action, margin, _ = _planner_label_for_state(
                    eval_kwargs=eval_kwargs,
                    history_record=record,
                    snake=snakes[target_idx],
                    seed=seed + row_idx,
                    planner_depth=planner_depth,
                    planner_fill_weight=planner_fill_weight,
                )
            if label_action is None or margin is None:
                skipped[f"{target_source}_no_label"] += 1
                continue
            if int(label_action) == int(record["action"]):
                skipped[f"{target_source}_matches_policy"] += 1
                continue
            if float(margin) < planner_margin_min:
                skipped["low_margin"] += 1
                continue

            obs = _obs_for_reconstructed_state(
                eval_kwargs=eval_kwargs,
                history_record=record,
                snake=snakes[target_idx],
                seed=seed + row_idx,
            )
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = reference_policy(obs_tensor)
            policy_logits = logits[0]
            policy_action = int(torch.argmax(policy_logits).item())
            if policy_action != int(record["action"]):
                skipped["policy_action_mismatch"] += 1
                continue
            probs = torch.softmax(policy_logits, dim=-1)
            entropy = float(-(probs * torch.log(probs.clamp_min(1e-8))).sum().item())

            observations.append(obs)
            actions.append(int(label_action))
            margins.append(float(margin))
            reason_counts[reason] += 1
            score_counts[terminal_score] += 1
            candidate_score_counts[int(record["score"])] += 1
            samples.append(
                CollectedSample(
                    episode_seed=int(row.get("episode", row_idx)),
                    episode_index=row_idx,
                    step=target_idx,
                    score=int(record["score"]),
                    fill_ratio=float(record["length"]) / float(int(eval_kwargs["board_size"]) ** 2),
                    policy_action=policy_action,
                    planner_action=int(label_action),
                    planner_margin=float(margin),
                    reference_entropy=entropy,
                )
            )
            if len(observations) >= max_samples:
                break

    if observations:
        obs_tensor = torch.from_numpy(np.stack(observations)).float()
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
    else:
        obs_tensor = torch.empty((0, 0), dtype=torch.float32)
        action_tensor = torch.empty((0,), dtype=torch.long)
    if preserve_observations:
        preserve_tensor = torch.from_numpy(np.stack(preserve_observations)).float()
    else:
        preserve_tensor = torch.empty((0, 0), dtype=torch.float32)

    summary = {
        "failures_path": str(failures_path),
        "rows_seen": len(rows),
        "collected_samples": len(actions),
        "step_back": step_back,
        "terminal_score_min": terminal_score_min,
        "terminal_score_max": terminal_score_max,
        "candidate_score_min": candidate_score_min,
        "target_source": target_source,
        "planner_depth": planner_depth,
        "planner_fill_weight": planner_fill_weight,
        "planner_margin_min": planner_margin_min,
        "preserve_score_min": preserve_score_min,
        "max_preserve_samples": max_preserve_samples,
        "preserve_samples": int(preserve_tensor.shape[0]),
        "mean_planner_margin": float(np.mean(margins)) if margins else None,
        "reason_counts": dict(sorted(reason_counts.items())),
        "terminal_score_histogram": {str(k): v for k, v in sorted(score_counts.items())},
        "candidate_score_histogram": {str(k): v for k, v in sorted(candidate_score_counts.items())},
        "skipped": dict(sorted(skipped.items())),
    }
    return obs_tensor, action_tensor, preserve_tensor, samples, summary


def train_with_preservation(
    *,
    student_policy: torch.nn.Module,
    reference_policy: torch.nn.Module,
    observations: torch.Tensor,
    planner_actions: torch.Tensor,
    preserve_observations: torch.Tensor,
    device: str,
    epochs: int,
    batch_size: int,
    preserve_batch_size: int,
    lr: float,
    ce_coef: float,
    kl_coef: float,
    preserve_kl_coef: float,
    grad_clip: float,
    seed: int,
) -> dict[str, Any]:
    student_policy.train()
    reference_policy.eval()
    optimizer = torch.optim.Adam(
        [param for param in student_policy.parameters() if param.requires_grad],
        lr=lr,
    )
    rng = np.random.default_rng(seed)
    n = int(observations.shape[0])
    n_preserve = int(preserve_observations.shape[0])
    steps = 0
    last_ce = last_kl = last_preserve_kl = last_total = None

    for _ in range(epochs):
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            obs_batch = observations[batch_idx].to(device)
            action_batch = planner_actions[batch_idx].to(device)

            student_logits, _ = student_policy(obs_batch)
            with torch.no_grad():
                ref_logits, _ = reference_policy(obs_batch)

            ce_loss = torch.nn.functional.cross_entropy(student_logits, action_batch)
            kl_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_logits, dim=-1),
                torch.nn.functional.softmax(ref_logits, dim=-1),
                reduction="batchmean",
            )
            total_loss = ce_coef * ce_loss + kl_coef * kl_loss
            preserve_kl_loss = None

            if n_preserve > 0 and preserve_kl_coef > 0.0:
                preserve_size = min(preserve_batch_size, n_preserve)
                preserve_idx = rng.choice(n_preserve, size=preserve_size, replace=False)
                preserve_batch = preserve_observations[preserve_idx].to(device)
                preserve_logits, _ = student_policy(preserve_batch)
                with torch.no_grad():
                    preserve_ref_logits, _ = reference_policy(preserve_batch)
                preserve_kl_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(preserve_logits, dim=-1),
                    torch.nn.functional.softmax(preserve_ref_logits, dim=-1),
                    reduction="batchmean",
                )
                total_loss = total_loss + preserve_kl_coef * preserve_kl_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_policy.parameters(), grad_clip)
            optimizer.step()

            last_ce = float(ce_loss.item())
            last_kl = float(kl_loss.item())
            last_preserve_kl = (
                float(preserve_kl_loss.item()) if preserve_kl_loss is not None else None
            )
            last_total = float(total_loss.item())
            steps += 1

    student_policy.eval()
    return {
        "epochs": int(epochs),
        "steps": int(steps),
        "preserve_samples": n_preserve,
        "last_ce_loss": last_ce,
        "last_kl_loss": last_kl,
        "last_preserve_kl_loss": last_preserve_kl,
        "last_total_loss": last_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline DAgger on harvested trap states")
    parser.add_argument("source", help="Source run dir or checkpoint")
    parser.add_argument("--checkpoint-name", default="best_eval.pt")
    parser.add_argument("--failures", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=183)
    parser.add_argument("--step-back", type=int, default=3)
    parser.add_argument("--terminal-score-min", type=int, default=393)
    parser.add_argument("--terminal-score-max", type=int, default=396)
    parser.add_argument("--candidate-score-min", type=int, default=390)
    parser.add_argument("--target-source", choices=["planner", "safe"], default="planner")
    parser.add_argument("--planner-depth", type=int, default=3)
    parser.add_argument("--planner-fill-weight", type=float, default=500.0)
    parser.add_argument("--planner-margin-min", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--preserve-score-min", type=int, default=390)
    parser.add_argument("--max-preserve-samples", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preserve-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--ce-coef", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=1.0)
    parser.add_argument("--preserve-kl-coef", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--policy-head-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=10001)
    parser.add_argument("--exp-name", default=None)
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

    exp_name = args.exp_name or f"offline_trap_dagger_{source_run_dir.name.split('_177')[0]}_s{args.seed}"
    run_dir = EXPERIMENTS_DIR / f"{exp_name}_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "offline_trap_dagger",
        "source_run_dir": str(source_run_dir),
        "source_checkpoint": str(source_checkpoint),
        "args": vars(args),
        "eval_kwargs": eval_kwargs,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[offline_trap] run_dir={run_dir}", flush=True)

    observations, planner_actions, preserve_observations, samples, collection_summary = collect_from_failures(
        failures_path=Path(args.failures).expanduser().resolve(),
        reference_policy=reference_policy,
        eval_kwargs=eval_kwargs,
        device=args.device,
        seed=args.seed,
        step_back=args.step_back,
        terminal_score_min=args.terminal_score_min,
        terminal_score_max=args.terminal_score_max,
        candidate_score_min=args.candidate_score_min,
        target_source=args.target_source,
        planner_depth=args.planner_depth,
        planner_fill_weight=args.planner_fill_weight,
        planner_margin_min=args.planner_margin_min,
        max_samples=args.max_samples,
        preserve_score_min=args.preserve_score_min,
        max_preserve_samples=args.max_preserve_samples,
    )
    (run_dir / "collection_summary.json").write_text(
        json.dumps(collection_summary, indent=2, sort_keys=True) + "\n"
    )
    with (run_dir / "collection_samples.jsonl").open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), sort_keys=True) + "\n")

    print(
        f"[offline_trap] collected={collection_summary['collected_samples']} "
        f"preserve={collection_summary['preserve_samples']} "
        f"mean_margin={collection_summary['mean_planner_margin']} "
        f"skipped={collection_summary['skipped']}",
        flush=True,
    )
    if observations.shape[0] == 0:
        print("[offline_trap] no samples collected", flush=True)
        return 1

    trainable_summary = _set_trainable_parameters(student_policy, args.policy_head_only)
    metadata["trainable"] = trainable_summary
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"[offline_trap] trainable={trainable_summary['trainable_params']} "
        f"frozen={trainable_summary['frozen_params']}",
        flush=True,
    )

    train_summary = train_with_preservation(
        student_policy=student_policy,
        reference_policy=reference_policy,
        observations=observations,
        planner_actions=planner_actions,
        preserve_observations=preserve_observations,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        preserve_batch_size=args.preserve_batch_size,
        lr=args.lr,
        ce_coef=args.ce_coef,
        kl_coef=args.kl_coef,
        preserve_kl_coef=args.preserve_kl_coef,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )
    (run_dir / "train_summary.json").write_text(
        json.dumps(train_summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[offline_trap] train steps={train_summary['steps']} "
        f"ce={train_summary['last_ce_loss']} kl={train_summary['last_kl_loss']} "
        f"total={train_summary['last_total_loss']}",
        flush=True,
    )

    checkpoint_path = run_dir / "best_eval.pt"
    torch.save(student_policy.state_dict(), checkpoint_path)
    print(f"[offline_trap] saved {checkpoint_path}", flush=True)

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
        f"[offline_trap] eval episodes={args.eval_episodes} "
        f"mean={eval_stats['mean_score']:.2f} std={eval_stats['std_score']:.2f} "
        f"win={eval_stats['win_rate'] * 100:.2f}% max={eval_stats['max_score']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
