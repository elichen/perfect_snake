#!/usr/bin/env python3
"""Harvest and summarize near-perfect deterministic failures for a checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval import SnakePolicy, _load_policy_state
from snake_env import SnakeEnv


ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        f.write("\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")


def load_source_args(source_run_dir: str | None) -> dict[str, Any]:
    if not source_run_dir:
        return {}
    run_json = Path(source_run_dir) / "run.json"
    if not run_json.exists():
        return {}
    try:
        with open(run_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return dict(payload.get("args", {}))


def cfg(cli_value: Any, source_args: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if key in source_args:
        return source_args[key]
    return default


def load_policy(
    checkpoint_path: str,
    *,
    board_size: int,
    device: str,
    network_scale: int,
    flood_fill: bool,
    aux_flood_fill: bool,
    aux_cycle_target: bool,
    aux_tail_target: bool,
    aux_safe_action_target: bool,
    aux_safe_action_soft_target: bool,
    aux_body_age_target: bool,
    head_centered: bool,
    late_head_min_fill: float | None,
) -> SnakePolicy:
    n_channels = (
        5
        + int(flood_fill)
        + int(aux_cycle_target)
        + int(aux_tail_target)
        + int(aux_safe_action_target)
        + 3 * int(aux_safe_action_soft_target)
        + int(aux_body_age_target)
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy = SnakePolicy(
        board_size=board_size,
        scale=network_scale,
        n_channels=n_channels,
        aux_flood_fill=aux_flood_fill,
        aux_cycle_target=aux_cycle_target,
        aux_tail_target=aux_tail_target,
        aux_safe_action_target=aux_safe_action_target,
        aux_safe_action_soft_target=aux_safe_action_soft_target,
        aux_body_age_target=aux_body_age_target,
        head_centered=head_centered,
        late_head_min_fill=late_head_min_fill,
    ).to(device)
    _load_policy_state(
        policy,
        state_dict,
        aux_flood_fill=aux_flood_fill,
        aux_cycle_target=aux_cycle_target,
        aux_tail_target=aux_tail_target,
        aux_body_age_target=aux_body_age_target,
        late_head_min_fill=late_head_min_fill,
    )
    policy.eval()
    return policy


def head_zone(head: tuple[int, int], board_size: int) -> str:
    r, c = head
    on_top = r == 0
    on_bottom = r == board_size - 1
    on_left = c == 0
    on_right = c == board_size - 1
    if (on_top or on_bottom) and (on_left or on_right):
        return "corner"
    if on_top or on_bottom or on_left or on_right:
        return "edge"
    return "interior"


def food_sector(head: tuple[int, int], food: tuple[int, int], direction: int) -> str:
    if food[0] < 0:
        return "none"
    dr = food[0] - head[0]
    dc = food[1] - head[1]
    forward = SnakeEnv.DIRECTIONS[direction]
    right = SnakeEnv.DIRECTIONS[(direction + 1) % 4]
    fwd = dr * forward[0] + dc * forward[1]
    lat = dr * right[0] + dc * right[1]
    if abs(fwd) >= abs(lat):
        if fwd > 0:
            return "front"
        if fwd < 0:
            return "back"
        return "same"
    if lat > 0:
        return "right"
    if lat < 0:
        return "left"
    return "same"


def motif_signature(step_record: dict[str, Any], reason: str, board_size: int) -> str:
    safe_best = step_record.get("safe_best_action")
    taken = step_record.get("action")
    if safe_best is None:
        safe_tag = "safe=na"
    else:
        safe_tag = "safe=match" if int(safe_best) == int(taken) else f"safe=miss->{safe_best}"
    tail = tuple(step_record["tail"])
    head = tuple(step_record["head"])
    tail_adj = int(abs(head[0] - tail[0]) + abs(head[1] - tail[1]) == 1)
    return "|".join(
        [
            f"reason={reason}",
            safe_tag,
            f"zone={head_zone(head, board_size)}",
            f"food={food_sector(head, tuple(step_record['food']), int(step_record['direction']))}",
            f"act={taken}",
            f"tailadj={tail_adj}",
        ]
    )


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest near-perfect deterministic Snake failures")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--source-exp", default=None)
    parser.add_argument("--source-run-dir", default=None)
    parser.add_argument("--output-root", default=str(EXPERIMENTS_DIR))
    parser.add_argument("--board-size", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--harvest-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--network-scale", type=int, default=None)
    parser.add_argument("--history-steps", type=int, default=16)
    parser.add_argument("--min-score", type=int, default=390)
    parser.add_argument("--max-score", type=int, default=396)
    parser.add_argument("--flood-fill", action="store_true")
    parser.add_argument("--aux-flood-fill", action="store_true")
    parser.add_argument("--aux-cycle-target", action="store_true")
    parser.add_argument("--aux-tail-target", action="store_true")
    parser.add_argument("--aux-safe-action-target", action="store_true")
    parser.add_argument("--aux-body-age-target", action="store_true")
    parser.add_argument("--head-centered", action="store_true")
    parser.add_argument("--late-head-min-fill", type=float, default=None)
    args = parser.parse_args()

    source_args = load_source_args(args.source_run_dir)
    board_size = int(cfg(args.board_size, source_args, "board_size", 20))
    network_scale = int(cfg(args.network_scale, source_args, "network_scale", 2))
    flood_fill = bool(args.flood_fill or bool(source_args.get("flood_fill", False)))
    aux_flood_fill = bool(args.aux_flood_fill or bool(source_args.get("aux_flood_fill", False)))
    aux_cycle_target = bool(args.aux_cycle_target or bool(source_args.get("aux_cycle_target", False)))
    aux_tail_target = bool(args.aux_tail_target or bool(source_args.get("aux_tail_target", False)))
    aux_safe_action_target = bool(args.aux_safe_action_target or bool(source_args.get("aux_safe_action_target", False)))
    aux_safe_action_soft_target = bool(getattr(args, "aux_safe_action_soft_target", False) or bool(source_args.get("aux_safe_action_soft_target", False)))
    aux_body_age_target = bool(args.aux_body_age_target or bool(source_args.get("aux_body_age_target", False)))
    head_centered = bool(args.head_centered or bool(source_args.get("head_centered", False)))
    late_head_min_fill = cfg(args.late_head_min_fill, source_args, "late_head_min_fill", None)

    timestamp = int(time.time() * 1_000_000)
    run_dir = Path(args.output_root) / f"{args.exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    failures_path = run_dir / "failures.jsonl"
    run_payload = {
        "exp_name": args.exp_name,
        "time": utc_now(),
        "mode": "failure_harvest",
        "source_exp": args.source_exp,
        "source_run_dir": args.source_run_dir,
        "checkpoint": args.checkpoint,
        "args": {
            "board_size": board_size,
            "episodes": args.episodes,
            "harvest_limit": args.harvest_limit,
            "seed": args.seed,
            "device": args.device,
            "network_scale": network_scale,
            "flood_fill": flood_fill,
            "aux_flood_fill": aux_flood_fill,
            "aux_cycle_target": aux_cycle_target,
            "aux_tail_target": aux_tail_target,
            "aux_safe_action_target": aux_safe_action_target,
            "aux_body_age_target": aux_body_age_target,
            "head_centered": head_centered,
            "late_head_min_fill": late_head_min_fill,
            "history_steps": args.history_steps,
            "min_score": args.min_score,
            "max_score": args.max_score,
        },
    }
    atomic_write_json(run_dir / "run.json", run_payload)

    policy = load_policy(
        args.checkpoint,
        board_size=board_size,
        device=args.device,
        network_scale=network_scale,
        flood_fill=flood_fill,
        aux_flood_fill=aux_flood_fill,
        aux_cycle_target=aux_cycle_target,
        aux_tail_target=aux_tail_target,
        aux_safe_action_target=aux_safe_action_target,
        aux_safe_action_soft_target=aux_safe_action_soft_target,
        aux_body_age_target=aux_body_age_target,
        head_centered=head_centered,
        late_head_min_fill=late_head_min_fill,
    )

    env = SnakeEnv(
        n=board_size,
        gamma=0.99,
        alpha=0.2,
        seed=args.seed,
        flood_fill_obs=flood_fill,
        cycle_target_obs=aux_cycle_target,
        tail_target_obs=aux_tail_target,
        safe_action_target_obs=aux_safe_action_target,
        safe_action_soft_target_obs=aux_safe_action_soft_target,
        body_age_target_obs=aux_body_age_target,
        head_centered=head_centered,
    )

    perfect_score = board_size * board_size - 3
    capture_start_score = max(args.min_score - 2, 0)
    harvested = 0
    wins = 0
    scores: list[int] = []
    signature_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    score_counts: Counter[int] = Counter()
    safe_match = 0
    safe_mismatch = 0
    safe_unknown = 0
    margin_values: list[float] = []

    episodes_run = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        steps = 0
        history: deque[dict[str, Any]] = deque(maxlen=args.history_steps)
        last_info: dict[str, Any] = {}

        while not done:
            safe_scores = None
            safe_best_action = None
            safe_margin = None
            if env.score >= capture_start_score:
                raw_scores = env.score_relative_actions()
                safe_scores = [None if not np.isfinite(v) else round(float(v), 3) for v in raw_scores]
                finite = [(i, float(v)) for i, v in enumerate(raw_scores) if np.isfinite(v)]
                if finite:
                    finite.sort(key=lambda item: item[1], reverse=True)
                    safe_best_action = int(finite[0][0])
                    if len(finite) > 1:
                        safe_margin = round(float(finite[0][1] - finite[1][1]), 3)

            obs_t = torch.as_tensor(obs, device=args.device, dtype=torch.float32).unsqueeze(0)
            logits, _ = policy(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())
            history.append(
                {
                    "score": int(env.score),
                    "length": int(env.snake_length),
                    "head": list(env.snake_head),
                    "tail": list(env.snake[-1]),
                    "food": list(env.food_pos),
                    "direction": int(env.direction),
                    "action": action,
                    "safe_scores": safe_scores,
                    "safe_best_action": safe_best_action,
                    "safe_margin": safe_margin,
                }
            )
            obs, _, terminated, truncated, last_info = env.step(action)
            done = terminated or truncated
            steps += 1

        score = int(last_info.get("score", 0))
        reason = str(last_info.get("reason", "unknown"))
        scores.append(score)
        episodes_run = ep + 1
        if score >= perfect_score:
            wins += 1

        if args.min_score <= score <= args.max_score and reason != "win":
            harvested += 1
            score_counts[score] += 1
            reason_counts[reason] += 1
            last_step = dict(history[-1]) if history else {}
            signature = motif_signature(last_step, reason, board_size) if last_step else f"reason={reason}|empty"
            signature_counts[signature] += 1
            if last_step:
                safe_best_action = last_step.get("safe_best_action")
                if safe_best_action is None:
                    safe_unknown += 1
                elif int(safe_best_action) == int(last_step["action"]):
                    safe_match += 1
                else:
                    safe_mismatch += 1
                if last_step.get("safe_margin") is not None:
                    margin_values.append(float(last_step["safe_margin"]))

            failure_record = {
                "episode": ep,
                "score": score,
                "length": int(last_info.get("length", score + 3)),
                "steps": steps,
                "reason": reason,
                "fill_ratio": round(float((score + 3) / float(board_size * board_size)), 6),
                "signature": signature,
                "terminal_head": list(env.snake_head),
                "terminal_tail": list(env.snake[-1]),
                "terminal_food": list(env.food_pos),
                "terminal_direction": int(env.direction),
                "terminal_snake": [list(cell) for cell in env.snake],
                "history": list(history),
            }
            append_jsonl(failures_path, failure_record)
            if args.harvest_limit is not None and harvested >= args.harvest_limit:
                break

    top_signatures = [
        {"signature": signature, "count": count}
        for signature, count in signature_counts.most_common(10)
    ]
    total_safe_labeled = safe_match + safe_mismatch
    denom = episodes_run if episodes_run else len(scores)
    summary = {
        "checkpoint": args.checkpoint,
        "source_exp": args.source_exp,
        "source_run_dir": args.source_run_dir,
        "episodes": denom,
        "seed": args.seed,
        "mean_score": round(float(np.mean(scores)), 4) if scores else 0.0,
        "min_score": int(min(scores)) if scores else 0,
        "max_score": int(max(scores)) if scores else 0,
        "win_rate": round(float(wins / denom), 6) if denom else 0.0,
        "harvest_min_score": args.min_score,
        "harvest_max_score": args.max_score,
        "harvest_limit": args.harvest_limit,
        "harvested_failures": harvested,
        "harvested_failure_rate": round(float(harvested / denom), 6) if denom else 0.0,
        "failure_reason_counts": dict(reason_counts),
        "failure_score_counts": {str(k): v for k, v in sorted(score_counts.items())},
        "top_signatures": top_signatures,
        "dominant_reason": reason_counts.most_common(1)[0][0] if reason_counts else None,
        "last_action_safe_match_rate": round(float(safe_match / total_safe_labeled), 6) if total_safe_labeled else None,
        "last_action_safe_mismatch_rate": round(float(safe_mismatch / total_safe_labeled), 6) if total_safe_labeled else None,
        "last_action_safe_unknown_count": safe_unknown,
        "mean_safe_margin": round(float(np.mean(margin_values)), 6) if margin_values else None,
        "created_at": utc_now(),
    }
    atomic_write_json(run_dir / "harvest_summary.json", summary)

    print(
        f"failure_harvest: exp={args.exp_name} harvested={harvested} "
        f"win_rate={summary['win_rate']:.2%} mean={summary['mean_score']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
