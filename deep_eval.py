"""Deep deterministic benchmark for Snake checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval import evaluate_checkpoint


DEFAULT_CHECKPOINT_NAME = "best_eval.pt"


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_target(target: str, checkpoint_name: str) -> tuple[Path, Path]:
    target_path = Path(target).expanduser().resolve()
    if target_path.is_dir():
        run_dir = target_path
        checkpoint_path = run_dir / checkpoint_name
        if not (run_dir / "run.json").exists():
            raise FileNotFoundError(f"Run dir missing run.json: {run_dir}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found in run dir: {checkpoint_path}")
        return run_dir, checkpoint_path

    if target_path.is_file():
        checkpoint_path = target_path
        run_dir = checkpoint_path.parent
        if (run_dir / "run.json").exists():
            return run_dir, checkpoint_path
        raise FileNotFoundError(
            f"Cannot infer run dir for checkpoint without adjacent run.json: {checkpoint_path}"
        )

    raise FileNotFoundError(f"Target not found: {target}")


def _infer_eval_kwargs(run_dir: Path, device: str, num_envs: int) -> dict[str, Any]:
    run = _load_json(run_dir / "run.json")
    args = run.get("args") or {}
    defaults = {
        "board_size": int(args.get("board_size", 20)),
        "episodes": 0,  # caller fills
        "seed": 0,  # caller fills
        "deterministic": True,
        "device": device,
        "network_scale": int(args.get("network_scale", 1)),
        "verbose": False,
        "flood_fill": bool(args.get("flood_fill", False)),
        "body_age_obs": bool(args.get("body_age_obs", False)),
        "obs_history": int(args.get("obs_history", 1)),
        "action_history_obs": int(args.get("action_history_obs", 0)),
        "iterative_cnn": bool(args.get("iterative_cnn", False)),
        "n_iterations": int(args.get("n_iterations", 12)),
        "aux_flood_fill": bool(args.get("aux_flood_fill", False)),
        "aux_cycle_target": bool(args.get("aux_cycle_target", False)),
        "aux_tail_target": bool(args.get("aux_tail_target", False)),
        "aux_safe_action_target": bool(args.get("aux_safe_action_target", False)),
        "aux_safe_action_soft_target": bool(args.get("aux_safe_action_soft_target", False)),
        "aux_body_age_target": bool(args.get("aux_body_age_target", False)),
        "late_head_min_fill": args.get("late_head_min_fill"),
        "aux_cycle_target_min_fill": args.get("aux_cycle_target_min_fill"),
        "aux_safe_action_target_min_fill": float(args.get("aux_safe_action_target_min_fill", 0.90)),
        "aux_safe_action_soft_target_min_fill": float(args.get("aux_safe_action_soft_target_min_fill", 0.90)),
        "body_age_obs_min_fill": float(args.get("body_age_obs_min_fill", 0.90)),
        "aux_body_age_target_min_fill": float(args.get("aux_body_age_target_min_fill", 0.80)),
        "aux_safe_action_fill_weight": float(args.get("aux_safe_action_fill_weight", 500.0)),
        "aux_safe_action_soft_temperature": float(args.get("aux_safe_action_soft_temperature", 1.0)),
        "safe_action_bonus": float(args.get("safe_action_bonus", 0.0)),
        "safe_action_bonus_min_fill": float(args.get("safe_action_bonus_min_fill", 0.95)),
        "safe_action_bonus_fill_weight": float(args.get("safe_action_bonus_fill_weight", 500.0)),
        "head_centered": bool(args.get("head_centered", False)),
        "num_envs": int(num_envs),
        "return_episode_data": True,
    }
    eval_kwargs = run.get("eval_kwargs")
    if eval_kwargs is not None:
        merged = dict(defaults)
        merged.update(eval_kwargs)
        merged["board_size"] = int(merged["board_size"])
        merged["network_scale"] = int(merged["network_scale"])
        merged["obs_history"] = int(merged["obs_history"])
        merged["action_history_obs"] = int(merged["action_history_obs"])
        merged["n_iterations"] = int(merged["n_iterations"])
        merged["episodes"] = 0
        merged["seed"] = 0
        merged["deterministic"] = True
        merged["device"] = device
        merged["num_envs"] = int(num_envs)
        merged["return_episode_data"] = True
        return merged
    return defaults


def _score_cdf(scores: list[int], perfect_score: int) -> dict[str, list[float] | list[int]]:
    counts = [0] * (perfect_score + 1)
    for score in scores:
        counts[int(score)] += 1
    total = max(1, len(scores))

    cdf_leq: list[float] = []
    running = 0
    for count in counts:
        running += count
        cdf_leq.append(running / total)

    cdf_geq: list[float] = [0.0] * (perfect_score + 1)
    running = 0
    for idx in range(perfect_score, -1, -1):
        running += counts[idx]
        cdf_geq[idx] = running / total

    return {
        "score_counts": counts,
        "score_cdf_leq": cdf_leq,
        "score_cdf_geq": cdf_geq,
    }


def _candidate_label(run_dir: Path, checkpoint_path: Path) -> str:
    if checkpoint_path.name == DEFAULT_CHECKPOINT_NAME:
        return run_dir.name
    return f"{run_dir.name}:{checkpoint_path.name}"


def run_deep_eval(
    *,
    targets: list[str],
    episodes: int,
    seed_start: int,
    checkpoint_name: str,
    device: str,
    num_envs: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for target in targets:
        run_dir, checkpoint_path = _resolve_target(target, checkpoint_name)
        kwargs = _infer_eval_kwargs(run_dir, device=device, num_envs=num_envs)
        kwargs["episodes"] = episodes
        kwargs["seed"] = seed_start

        print(
            f"[deep_eval] {run_dir.name} checkpoint={checkpoint_path.name} "
            f"episodes={episodes} seed_start={seed_start} device={device}"
            ,
            flush=True,
        )
        stats = evaluate_checkpoint(
            checkpoint_path=str(checkpoint_path),
            **kwargs,
        )
        scores = [int(score) for score in stats.pop("scores")]
        lengths = [int(length) for length in stats.pop("lengths")]
        death_lengths = [int(length) for length in stats.pop("death_lengths_raw")]
        reasons = [str(reason) for reason in stats.pop("reasons")]
        perfect_score = int(stats["perfect_score"])
        stats.update(_score_cdf(scores, perfect_score))
        stats["label"] = _candidate_label(run_dir, checkpoint_path)
        stats["run_dir"] = str(run_dir)
        stats["run_json"] = str(run_dir / "run.json")
        stats["checkpoint"] = str(checkpoint_path)
        stats["seed_start"] = seed_start
        stats["episodes"] = episodes
        stats["scores"] = scores
        stats["lengths"] = lengths
        stats["death_lengths_raw"] = death_lengths
        stats["reasons"] = reasons
        results.append(stats)
        print(
            f"[deep_eval] done {stats['label']} "
            f"mean={stats['mean_score']:.2f} std={stats['std_score']:.2f} "
            f"win={stats['win_rate']*100:.2f}% max={stats['max_score']}",
            flush=True,
        )

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "deterministic": True,
        "episodes": episodes,
        "seed_start": seed_start,
        "device": device,
        "num_envs": num_envs,
        "targets": targets,
        "results": results,
    }


def _write_score_cdf_csv(path: Path, payload: dict[str, Any]) -> None:
    results = payload["results"]
    if not results:
        return
    max_score = max(int(result["perfect_score"]) for result in results)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "score", "count", "cdf_leq", "cdf_geq"])
        for result in results:
            counts = result["score_counts"]
            cdf_leq = result["score_cdf_leq"]
            cdf_geq = result["score_cdf_geq"]
            for score in range(max_score + 1):
                count = counts[score] if score < len(counts) else 0
                leq = cdf_leq[score] if score < len(cdf_leq) else cdf_leq[-1]
                geq = cdf_geq[score] if score < len(cdf_geq) else 0.0
                writer.writerow([result["label"], score, count, leq, geq])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deep deterministic benchmark across Snake checkpoints")
    parser.add_argument("targets", nargs="+", help="Run dirs or checkpoint files")
    parser.add_argument("--episodes", type=int, default=1000, help="Deterministic episodes per candidate")
    parser.add_argument("--seed-start", type=int, default=1, help="First episode seed; candidates share the same seed schedule")
    parser.add_argument("--checkpoint-name", type=str, default=DEFAULT_CHECKPOINT_NAME, help="Checkpoint filename when a target is a run dir")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--num-envs", type=int, default=64, help="Parallel envs during eval")
    parser.add_argument("--output-prefix", type=str, default=None, help="Output path prefix; writes .json and .csv")
    args = parser.parse_args()

    payload = run_deep_eval(
        targets=args.targets,
        episodes=args.episodes,
        seed_start=args.seed_start,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
        num_envs=args.num_envs,
    )

    if args.output_prefix is None:
        output_prefix = (
            Path("/Users/elichen/code/perfect_snake/experiments")
            / f"deep_eval_{_utc_timestamp()}"
        )
    else:
        output_prefix = Path(args.output_prefix).expanduser().resolve()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_score_cdf_csv(csv_path, payload)

    print(f"[deep_eval] wrote {json_path}", flush=True)
    print(f"[deep_eval] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
