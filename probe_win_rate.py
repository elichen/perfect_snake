#!/usr/bin/env python3
"""Run a longer deterministic eval probe and write a summary for mission_loop."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from eval import evaluate_checkpoint


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a deeper deterministic eval probe for a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--source-exp", default=None)
    parser.add_argument("--source-run-dir", default=None)
    parser.add_argument("--output-root", default=str(EXPERIMENTS_DIR))
    parser.add_argument("--board-size", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--network-scale", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    args = parser.parse_args()

    source_args = load_source_args(args.source_run_dir)
    board_size = int(cfg(args.board_size, source_args, "board_size", 20))
    network_scale = int(cfg(args.network_scale, source_args, "network_scale", 2))
    flood_fill = bool(source_args.get("flood_fill", False))
    aux_flood_fill = bool(source_args.get("aux_flood_fill", False))
    aux_cycle_target = bool(source_args.get("aux_cycle_target", False))
    aux_tail_target = bool(source_args.get("aux_tail_target", False))
    aux_safe_action_target = bool(source_args.get("aux_safe_action_target", False))
    aux_safe_action_soft_target = bool(source_args.get("aux_safe_action_soft_target", False))
    aux_body_age_target = bool(source_args.get("aux_body_age_target", False))
    aux_cycle_target_min_fill = cfg(None, source_args, "aux_cycle_target_min_fill", None)
    aux_safe_action_target_min_fill = float(cfg(None, source_args, "aux_safe_action_target_min_fill", 0.90))
    aux_safe_action_soft_target_min_fill = float(cfg(None, source_args, "aux_safe_action_soft_target_min_fill", 0.90))
    aux_body_age_target_min_fill = float(cfg(None, source_args, "aux_body_age_target_min_fill", 0.80))
    aux_safe_action_fill_weight = float(cfg(None, source_args, "aux_safe_action_fill_weight", 500.0))
    aux_safe_action_soft_temperature = float(cfg(None, source_args, "aux_safe_action_soft_temperature", 1.0))
    safe_action_bonus = float(cfg(None, source_args, "safe_action_bonus", 0.0))
    safe_action_bonus_min_fill = float(cfg(None, source_args, "safe_action_bonus_min_fill", 0.95))
    safe_action_bonus_fill_weight = float(cfg(None, source_args, "safe_action_bonus_fill_weight", 500.0))
    head_centered = bool(source_args.get("head_centered", False))
    late_head_min_fill = cfg(None, source_args, "late_head_min_fill", None)

    timestamp = int(time.time() * 1_000_000)
    run_dir = Path(args.output_root) / f"{args.exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    run_payload = {
        "exp_name": args.exp_name,
        "time": utc_now(),
        "mode": "win_probe",
        "source_exp": args.source_exp,
        "source_run_dir": args.source_run_dir,
        "checkpoint": args.checkpoint,
        "args": {
            "board_size": board_size,
            "episodes": args.episodes,
            "seed": args.seed,
            "device": args.device,
            "network_scale": network_scale,
            "num_envs": args.num_envs,
            "flood_fill": flood_fill,
            "aux_flood_fill": aux_flood_fill,
            "aux_cycle_target": aux_cycle_target,
            "aux_tail_target": aux_tail_target,
            "aux_safe_action_target": aux_safe_action_target,
            "aux_safe_action_soft_target": aux_safe_action_soft_target,
            "aux_body_age_target": aux_body_age_target,
            "aux_cycle_target_min_fill": aux_cycle_target_min_fill,
            "aux_safe_action_target_min_fill": aux_safe_action_target_min_fill,
            "aux_safe_action_soft_target_min_fill": aux_safe_action_soft_target_min_fill,
            "aux_body_age_target_min_fill": aux_body_age_target_min_fill,
            "aux_safe_action_fill_weight": aux_safe_action_fill_weight,
            "aux_safe_action_soft_temperature": aux_safe_action_soft_temperature,
            "safe_action_bonus": safe_action_bonus,
            "safe_action_bonus_min_fill": safe_action_bonus_min_fill,
            "safe_action_bonus_fill_weight": safe_action_bonus_fill_weight,
            "head_centered": head_centered,
            "late_head_min_fill": late_head_min_fill,
        },
    }
    atomic_write_json(run_dir / "run.json", run_payload)

    stats = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        board_size=board_size,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=True,
        device=args.device,
        network_scale=network_scale,
        flood_fill=flood_fill,
        aux_flood_fill=aux_flood_fill,
        aux_cycle_target=aux_cycle_target,
        aux_tail_target=aux_tail_target,
        aux_safe_action_target=aux_safe_action_target,
        aux_safe_action_soft_target=aux_safe_action_soft_target,
        aux_body_age_target=aux_body_age_target,
        late_head_min_fill=late_head_min_fill,
        aux_cycle_target_min_fill=aux_cycle_target_min_fill,
        aux_safe_action_target_min_fill=aux_safe_action_target_min_fill,
        aux_safe_action_soft_target_min_fill=aux_safe_action_soft_target_min_fill,
        aux_body_age_target_min_fill=aux_body_age_target_min_fill,
        aux_safe_action_fill_weight=aux_safe_action_fill_weight,
        aux_safe_action_soft_temperature=aux_safe_action_soft_temperature,
        safe_action_bonus=safe_action_bonus,
        safe_action_bonus_min_fill=safe_action_bonus_min_fill,
        safe_action_bonus_fill_weight=safe_action_bonus_fill_weight,
        head_centered=head_centered,
        num_envs=args.num_envs,
    )
    summary = {
        "time": utc_now(),
        "mode": "win_probe",
        "source_exp": args.source_exp,
        "checkpoint": args.checkpoint,
        **stats,
    }
    atomic_write_json(run_dir / "win_probe_summary.json", summary)
    print(
        f"win_probe source={args.source_exp} episodes={summary['episodes']} "
        f"wins={summary['wins']} win_rate={summary['win_rate']:.2%} mean={summary['mean_score']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
