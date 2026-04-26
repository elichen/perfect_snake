#!/usr/bin/env python3
"""Hypothesis-driven Snake experiment loop guided by LOOP_POLICY.md."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
TRAIN = ROOT / "train.py"
HARVEST_FAILURES = ROOT / "harvest_endgame_failures.py"
WIN_PROBE = ROOT / "probe_win_rate.py"
EXPERIMENTS_DIR = ROOT / "experiments"
STATE_PATH = EXPERIMENTS_DIR / "mission_loop_state.json"
JOURNAL_PATH = EXPERIMENTS_DIR / "mission_loop_journal.jsonl"
LOG_PATH = EXPERIMENTS_DIR / "mission_loop.log"
STOP_PATH = EXPERIMENTS_DIR / "mission_loop.stop"
CURRENT_BEST_PATH = EXPERIMENTS_DIR / "current_best.json"
BASE_RESUME = ROOT / "experiments/exp074_multi_path_curriculum_ft_177317600936/best_eval.pt"

DISK_FLOOR_GB = 4.5
CONTROL_INTERESTING_SCORE = 350.0
CONTROL_PROMOTE_SCORE = 370.0
CONTROL_RESUME_SCORE = 372.0
CONTROL_MICRO_SCORE = 372.0
CONTROL_ALT_MICRO_SCORE = 380.0
CONTROL_ALT_NANO_SCORE = 380.0
HARVEST_GATE_SCORE = 379.0
RESUME_CHAIN_SCORE = 365.0
RESUME_SHORT_CHAIN_MAX = 380.0
RESUME_MICRO_CHAIN_SCORE = 380.0
MICRO_TO_SHORT_REVISIT_SCORE = 375.0
CONTROL_COLD_SCORE = 360.0
CONTROL_COLD_STREAK_LIMIT = 5
PARENT_SEED_PROBE_TRIGGER = 10
PARENT_SEED_PROBE_LIMIT = 3
PARENT_MICRO_PROBE_LIMIT = 1
CONTROL_ALT_MICRO_PROBE_LIMIT = 1
CONTROL_ALT_NANO_PROBE_LIMIT = 1
ENDGAME_BASIN_MICRO_PROBE_LIMIT = 1
ENDGAME_BASIN_MICRO_SCORE = 384.0
INCUMBENT_MICRO_PROBE_LIMIT = 2
INCUMBENT_NANO_PROBE_LIMIT = 2
BODY_AGE_MICRO_PROBE_LIMIT = 1
HISTORY_SHORT_PROBE_LIMIT = 1
HISTORICAL_ENDBODY_SHORT_PROBE_LIMIT = 1
ENDGAME_MICRO_PROBE_LIMIT = 1
ENDGAME_SAFE_PROBE_LIMIT = 1
ENDGAME_BODY_AGE_PROBE_LIMIT = 1
FAILURE_HARVEST_LIMIT = 1
FAILURE_HARVEST_EPISODES = 100
FAILURE_HARVEST_TARGET_FAILURES = 40
FAILURE_HARVEST_MIN_SCORE = 390
FAILURE_HARVEST_MAX_SCORE = 396
WIN_PROBE_LIMIT = 1
WIN_PROBE_EPISODES = 200
ALT_SEED_COLD_WINDOW = 3
ALT_SEED_COLD_BAD_LIMIT = 2
ALT_SEED_COLD_SCORE = 370.0
ALT_SEED_COLD_LT20 = 0.05
HISTORICAL_ENDBODY_SHORT_MIN_SCORE = 383.0
RESUME_LR = 1e-6
MICRO_RESUME_TIMESTEPS = 65536
NANO_RESUME_TIMESTEPS = 32768
NANO_RESUME_LR = 5e-7
RESUME_NANO_TRIGGER_SCORE = 392.0
RESUME_NANO_CHAIN_SCORE = 391.0
STALE_RUN_SECONDS = 1800
STALE_RUN_POLL_SECONDS = 10


@dataclass
class EvalStats:
    exp_name: str
    run_dir: str
    seed: int
    source: str
    mean_score: float
    median_score: float
    win_rate: float
    phase_lt20_rate: float
    phase_gte95_rate: float
    death_self_rate: float
    death_wall_rate: float
    agent_steps: int
    epoch: int


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_journal(event: str, **payload: Any) -> None:
    record = {"time": utc_now(), "event": event, **payload}
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        f.write("\n")
    os.replace(tmp, path)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def num_or(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def int_or(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def free_disk_gb() -> float:
    usage = shutil.disk_usage(ROOT)
    return usage.free / (1024 ** 3)


def score_tuple(stats: EvalStats) -> tuple[float, float, float, float]:
    return (
        float(stats.win_rate),
        float(stats.mean_score),
        float(stats.phase_gte95_rate),
        -float(stats.phase_lt20_rate),
    )


def seed_from_exp_name(exp_name: str) -> int | None:
    match = re.search(r"_s(\d+)", exp_name)
    if not match:
        return None
    return int(match.group(1))


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def wait_for_pid(pid: int) -> int:
    while pid_alive(pid):
        time.sleep(2)
    return 0


def activity_paths(run_dir: Path | None, driver_log: Path | None) -> list[Path]:
    paths: list[Path] = []
    if driver_log is not None:
        paths.append(driver_log)
    if run_dir is not None:
        paths.extend(
            [
                run_dir / "metrics.jsonl",
                run_dir / "summary.json",
                run_dir / "harvest_summary.json",
            ]
        )
    return paths


def latest_activity_timestamp(run_dir: Path | None, driver_log: Path | None) -> float:
    latest = 0.0
    for path in activity_paths(run_dir, driver_log):
        try:
            latest = max(latest, path.stat().st_mtime)
        except FileNotFoundError:
            continue
    return latest


def terminate_process_tree(pid: int, grace_seconds: float = 5.0) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return
    children = parent.children(recursive=True)
    for proc in children:
        try:
            proc.terminate()
        except psutil.Error:
            pass
    try:
        parent.terminate()
    except psutil.Error:
        pass
    _, alive = psutil.wait_procs([*children, parent], timeout=grace_seconds)
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass


def wait_for_process_with_stall_guard(
    *,
    pid: int,
    exp_name: str,
    run_dir: Path | None,
    driver_log: Path | None,
    proc: subprocess.Popen[str] | None = None,
    stale_seconds: int = STALE_RUN_SECONDS,
    poll_seconds: int = STALE_RUN_POLL_SECONDS,
) -> int:
    while True:
        if proc is not None:
            exit_code = proc.poll()
            if exit_code is not None:
                return int(exit_code)
        elif not pid_alive(pid):
            return 0

        latest = latest_activity_timestamp(run_dir, driver_log)
        if latest > 0.0:
            idle = time.time() - latest
            if idle > stale_seconds:
                log(
                    f"stale run detected: {exp_name} pid={pid} idle={idle:.0f}s "
                    f"threshold={stale_seconds}s"
                )
                append_journal(
                    "stale_run_detected",
                    exp_name=exp_name,
                    pid=pid,
                    idle_seconds=idle,
                    stale_seconds=stale_seconds,
                    run_dir=str(run_dir) if run_dir else None,
                    driver_log=str(driver_log) if driver_log else None,
                )
                terminate_process_tree(pid)
                if proc is not None:
                    try:
                        return int(proc.wait(timeout=5))
                    except subprocess.TimeoutExpired:
                        terminate_process_tree(pid)
                return -15
        time.sleep(poll_seconds)


def run_matches_control_policy(args: dict[str, Any]) -> bool:
    return (
        args.get("board_size") == 20
        and args.get("num_envs") == 64
        and args.get("horizon") == 256
        and args.get("minibatch_size") == 4096
        and args.get("device") == "cpu"
        and args.get("gamma") == 0.999
        and args.get("gae_lambda") == 0.9
        and args.get("vf_clip_coef") == 1.0
        and args.get("network_scale") == 2
        and bool(args.get("flood_fill")) is True
        and bool(args.get("aux_flood_fill")) is True
        and bool(args.get("head_centered")) is True
        and float(args.get("curriculum_prob", -1)) == 0.1
        and float(args.get("curriculum_min_fill", -1)) == 0.9
        and float(args.get("curriculum_max_fill", -1)) == 0.98
        and float(args.get("curriculum_follow_bonus", -1)) == 0.005
        and float(args.get("curriculum_follow_min_fill", -1)) == 0.95
        and float(args.get("lr", -1)) == 5e-6
        and bool(args.get("no_anneal_lr")) is True
        and bool(args.get("head_centered")) is True
    )


def extract_best_eval(run_dir: Path, exp_name: str, seed: int, source: str) -> EvalStats | None:
    payload = load_json(run_dir / "summary.json")
    if not payload:
        return None
    best = payload.get("best_eval")
    if not best:
        return None
    return EvalStats(
        exp_name=exp_name,
        run_dir=str(run_dir),
        seed=seed,
        source=source,
        mean_score=num_or(best.get("mean_score"), 0.0),
        median_score=num_or(best.get("median_score"), 0.0),
        win_rate=num_or(best.get("win_rate"), 0.0),
        phase_lt20_rate=num_or(best.get("phase_lt20_rate"), 1.0),
        phase_gte95_rate=num_or(best.get("phase_gte95_rate"), 0.0),
        death_self_rate=num_or(best.get("death_self_rate"), 0.0),
        death_wall_rate=num_or(best.get("death_wall_rate"), 0.0),
        agent_steps=int_or(best.get("agent_steps"), 0),
        epoch=int_or(best.get("epoch"), 0),
    )


def incumbent_resume_path(stats: EvalStats | dict[str, Any] | None) -> Path | None:
    if stats is None:
        return None
    run_dir = Path(stats["run_dir"] if isinstance(stats, dict) else stats.run_dir)
    path = run_dir / "best_eval_resume.pt"
    return path if path.exists() else None


def find_run_dir_by_exp_name(exp_name: str) -> Path | None:
    matches = sorted(
        EXPERIMENTS_DIR.glob(f"{exp_name}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def parent_resume_candidate_for_exp(exp_name: str) -> dict[str, str] | None:
    run_dir = find_run_dir_by_exp_name(exp_name)
    if run_dir is None:
        return None
    payload = load_json(run_dir / "run.json")
    if not payload:
        return None
    args = payload.get("args", {})
    resume_state = args.get("resume_state")
    if not resume_state:
        return None
    resume_path = Path(str(resume_state))
    if not resume_path.exists():
        return None
    parent_run_dir = resume_path.parent
    parent_payload = load_json(parent_run_dir / "run.json") or {}
    parent_args = parent_payload.get("args", {})
    source_exp = parent_args.get("exp_name")
    if not source_exp:
        source_exp = parent_run_dir.name.rsplit("_", 1)[0]
    return {
        "resume_state": str(resume_path),
        "source_exp": str(source_exp),
    }


def allocate_parent_seed_probe(
    state: dict[str, Any],
    source_exp: str,
    resume_state: str,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("parent_seed_probe_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= PARENT_SEED_PROBE_LIMIT:
        return None
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    probe_counts[source_exp] = used + 1
    return {
        "family": "resume_short",
        "seed": seed,
        "resume_state": str(resume_state),
        "source_exp": source_exp,
        "reason": reason,
    }


def allocate_parent_micro_probe(
    state: dict[str, Any],
    source_exp: str,
    resume_state: str,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("parent_micro_probe_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= PARENT_MICRO_PROBE_LIMIT:
        return None
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    probe_counts[source_exp] = used + 1
    return {
        "family": "resume_micro",
        "seed": seed,
        "resume_state": str(resume_state),
        "source_exp": source_exp,
        "reason": reason,
    }


def allocate_control_alt_micro_probe(
    state: dict[str, Any],
    source_exp: str,
    resume_state: str,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("control_alt_micro_probe_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= CONTROL_ALT_MICRO_PROBE_LIMIT:
        return None
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    probe_counts[source_exp] = used + 1
    return {
        "family": "resume_micro",
        "seed": seed,
        "resume_state": str(resume_state),
        "source_exp": source_exp,
        "reason": reason,
    }


def allocate_control_alt_nano_probe(
    state: dict[str, Any],
    source_exp: str,
    resume_state: str,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("control_alt_nano_probe_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= CONTROL_ALT_NANO_PROBE_LIMIT:
        return None
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    probe_counts[source_exp] = used + 1
    return {
        "family": "resume_nano",
        "seed": seed,
        "resume_state": str(resume_state),
        "source_exp": source_exp,
        "reason": reason,
    }


def allocate_endgame_basin_micro_probe(
    state: dict[str, Any],
    family: str,
    source_exp: str,
    resume_state: str,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("endgame_basin_micro_probe_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= ENDGAME_BASIN_MICRO_PROBE_LIMIT:
        return None
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    probe_counts[source_exp] = used + 1
    return {
        "family": family,
        "seed": seed,
        "resume_state": str(resume_state),
        "source_exp": source_exp,
        "reason": reason,
    }


def allocate_resume_seed_probe(
    state: dict[str, Any],
    source_exp: str,
    resume_state: str,
    reason: str,
) -> dict[str, Any]:
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    return {
        "family": "resume_short",
        "seed": seed,
        "resume_state": str(resume_state),
        "source_exp": source_exp,
        "reason": reason,
    }


def allocate_incumbent_micro_probe(
    state: dict[str, Any],
    incumbent_exp: str,
    resume_state: str,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("incumbent_micro_probe_counts", {})
    used = int(probe_counts.get(incumbent_exp, 0))
    if used >= INCUMBENT_MICRO_PROBE_LIMIT:
        return None
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    probe_counts[incumbent_exp] = used + 1
    return {
        "family": "resume_micro",
        "seed": seed,
        "resume_state": str(resume_state),
        "source_exp": incumbent_exp,
        "reason": reason,
    }


def allocate_incumbent_nano_probe(
    state: dict[str, Any],
    incumbent_exp: str,
    resume_state: str,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("incumbent_nano_probe_counts", {})
    used = int(probe_counts.get(incumbent_exp, 0))
    if used >= INCUMBENT_NANO_PROBE_LIMIT:
        return None
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    probe_counts[incumbent_exp] = used + 1
    return {
        "family": "resume_nano",
        "seed": seed,
        "resume_state": str(resume_state),
        "source_exp": incumbent_exp,
        "reason": reason,
        "chain_depth": 1,
    }


def allocate_body_age_micro_probe(
    state: dict[str, Any],
    source_exp: str,
    resume_state: str,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("body_age_micro_probe_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= BODY_AGE_MICRO_PROBE_LIMIT:
        return None
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    probe_counts[source_exp] = used + 1
    return {
        "family": "resume_micro_body_age",
        "seed": seed,
        "resume_state": str(resume_state),
        "source_exp": source_exp,
        "reason": reason,
        "chain_depth": 1,
    }


def allocate_history_short_probe(
    state: dict[str, Any],
    source_exp: str,
    resume_state: str,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("history_short_probe_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= HISTORY_SHORT_PROBE_LIMIT:
        return None
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    probe_counts[source_exp] = used + 1
    return {
        "family": "resume_history_short",
        "seed": seed,
        "resume_state": str(resume_state),
        "source_exp": source_exp,
        "reason": reason,
        "chain_depth": 1,
    }


def allocate_failure_harvest_probe(
    state: dict[str, Any],
    source_exp: str,
    source_run_dir: str,
    checkpoint: str,
    seed: int,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("failure_harvest_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= FAILURE_HARVEST_LIMIT:
        return None
    probe_counts[source_exp] = used + 1
    return {
        "family": "failure_harvest",
        "seed": seed,
        "checkpoint": checkpoint,
        "source_exp": source_exp,
        "source_run_dir": source_run_dir,
        "reason": reason,
    }


def allocate_win_probe(
    state: dict[str, Any],
    source_exp: str,
    source_run_dir: str,
    checkpoint: str,
    seed: int,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("win_probe_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= WIN_PROBE_LIMIT:
        return None
    probe_counts[source_exp] = used + 1
    return {
        "family": "win_probe",
        "seed": seed,
        "checkpoint": checkpoint,
        "source_exp": source_exp,
        "source_run_dir": source_run_dir,
        "reason": reason,
    }


def discover_incumbent() -> EvalStats | None:
    current = load_json(CURRENT_BEST_PATH)
    if current:
        try:
            return EvalStats(**current)
        except Exception:
            pass

    best: EvalStats | None = None
    for run_json in EXPERIMENTS_DIR.glob("*/run.json"):
        payload = load_json(run_json)
        if not payload:
            continue
        args = payload.get("args", {})
        exp_name = args.get("exp_name")
        if not exp_name or not run_matches_control_policy(args):
            continue
        seed = args.get("seed")
        if seed is None:
            seed = seed_from_exp_name(exp_name)
        if seed is None:
            continue
        stats = extract_best_eval(run_json.parent, exp_name, int(seed), "control")
        if stats is None:
            continue
        if best is None or score_tuple(stats) > score_tuple(best):
            best = stats
    return best


def discover_best_resume_candidate(
    exclude_exp: str | None = None,
    min_mean_score: float = RESUME_CHAIN_SCORE,
) -> dict[str, Any] | None:
    best: EvalStats | None = None
    best_resume_path: Path | None = None
    for run_json in EXPERIMENTS_DIR.glob("*/run.json"):
        payload = load_json(run_json)
        if not payload:
            continue
        args = payload.get("args", {})
        exp_name = args.get("exp_name")
        if not exp_name or "_resume_" not in exp_name or not exp_name.startswith("ppo_research_"):
            continue
        if exclude_exp and exp_name == exclude_exp:
            continue
        seed = args.get("seed")
        if seed is None:
            seed = seed_from_exp_name(exp_name)
        if seed is None:
            continue
        stats = extract_best_eval(run_json.parent, exp_name, int(seed), "resume_short")
        if stats is None or stats.phase_lt20_rate > 0.0 or stats.mean_score < min_mean_score:
            continue
        if recent_alt_seed_resume_source_is_cold(exp_name):
            continue
        resume_path = run_json.parent / "best_eval_resume.pt"
        if not resume_path.exists():
            continue
        if best is None or score_tuple(stats) > score_tuple(best):
            best = stats
            best_resume_path = resume_path
    if best is None or best_resume_path is None:
        return None
    return {
        "family": "resume_short",
        "seed": int(best.seed),
        "resume_state": str(best_resume_path),
        "source_exp": best.exp_name,
        "reason": "cold_control_revisit_best_resume",
    }


def discover_best_control_alt_micro_candidate(
    state: dict[str, Any],
    exclude_exp: str | None = None,
    min_mean_score: float = CONTROL_ALT_MICRO_SCORE,
) -> dict[str, Any] | None:
    best: EvalStats | None = None
    best_resume_path: Path | None = None
    probe_counts = state.setdefault("control_alt_micro_probe_counts", {})
    for run_json in EXPERIMENTS_DIR.glob("*/run.json"):
        payload = load_json(run_json)
        if not payload:
            continue
        args = payload.get("args", {})
        exp_name = args.get("exp_name")
        if not exp_name or "_ctrl_" not in exp_name or not exp_name.startswith("ppo_research_"):
            continue
        if exclude_exp and exp_name == exclude_exp:
            continue
        if int(probe_counts.get(exp_name, 0)) >= CONTROL_ALT_MICRO_PROBE_LIMIT:
            continue
        seed = args.get("seed")
        if seed is None:
            seed = seed_from_exp_name(exp_name)
        if seed is None:
            continue
        stats = extract_best_eval(run_json.parent, exp_name, int(seed), "control")
        if stats is None or stats.phase_lt20_rate > 0.0 or stats.mean_score < min_mean_score:
            continue
        resume_path = run_json.parent / "best_eval_resume.pt"
        if not resume_path.exists():
            continue
        if best is None or score_tuple(stats) > score_tuple(best):
            best = stats
            best_resume_path = resume_path
    if best is None or best_resume_path is None:
        return None
    return allocate_control_alt_micro_probe(
        state,
        best.exp_name,
        str(best_resume_path),
        "cold_control_revisit_best_control_alt_seed",
    )


def discover_best_control_alt_nano_candidate(
    state: dict[str, Any],
    exclude_exp: str | None = None,
    min_mean_score: float = CONTROL_ALT_NANO_SCORE,
) -> dict[str, Any] | None:
    best: EvalStats | None = None
    best_resume_path: Path | None = None
    probe_counts = state.setdefault("control_alt_nano_probe_counts", {})
    for run_json in EXPERIMENTS_DIR.glob("*/run.json"):
        payload = load_json(run_json)
        if not payload:
            continue
        args = payload.get("args", {})
        exp_name = args.get("exp_name")
        if not exp_name or "_ctrl_" not in exp_name or not exp_name.startswith("ppo_research_"):
            continue
        if exclude_exp and exp_name == exclude_exp:
            continue
        if int(probe_counts.get(exp_name, 0)) >= CONTROL_ALT_NANO_PROBE_LIMIT:
            continue
        seed = args.get("seed")
        if seed is None:
            seed = seed_from_exp_name(exp_name)
        if seed is None:
            continue
        stats = extract_best_eval(run_json.parent, exp_name, int(seed), "control")
        if stats is None or stats.phase_lt20_rate > 0.0 or stats.mean_score < min_mean_score:
            continue
        resume_path = run_json.parent / "best_eval_resume.pt"
        if not resume_path.exists():
            continue
        if best is None or score_tuple(stats) > score_tuple(best):
            best = stats
            best_resume_path = resume_path
    if best is None or best_resume_path is None:
        return None
    return allocate_control_alt_nano_probe(
        state,
        best.exp_name,
        str(best_resume_path),
        "cold_control_revisit_best_control_alt_seed_nano",
    )


def discover_best_endgame_basin_micro_candidate(
    state: dict[str, Any],
    exclude_exp: str | None = None,
    min_mean_score: float = ENDGAME_BASIN_MICRO_SCORE,
) -> dict[str, Any] | None:
    best: EvalStats | None = None
    best_family: str | None = None
    best_resume_path: Path | None = None
    probe_counts = state.setdefault("endgame_basin_micro_probe_counts", {})
    for run_json in EXPERIMENTS_DIR.glob("*/run.json"):
        payload = load_json(run_json)
        if not payload:
            continue
        args = payload.get("args", {})
        exp_name = args.get("exp_name")
        if not exp_name or not exp_name.startswith("ppo_research_"):
            continue
        if exclude_exp and exp_name == exclude_exp:
            continue
        if int(probe_counts.get(exp_name, 0)) >= ENDGAME_BASIN_MICRO_PROBE_LIMIT:
            continue
        source_family: str | None = None
        if "_endmix_" in exp_name:
            source_family = "resume_endgame_combo_micro"
        elif "_endbody_" in exp_name:
            source_family = "resume_endgame_body_age_micro"
        elif "_endsafe_" in exp_name:
            source_family = "resume_endgame_safe_micro"
        if source_family is None:
            continue
        seed = args.get("seed")
        if seed is None:
            seed = seed_from_exp_name(exp_name)
        if seed is None:
            continue
        stats = extract_best_eval(run_json.parent, exp_name, int(seed), source_family)
        if stats is None:
            continue
        if stats.phase_lt20_rate > 0.0 or stats.mean_score < min_mean_score or stats.phase_gte95_rate < 0.8:
            continue
        resume_path = run_json.parent / "best_eval_resume.pt"
        if not resume_path.exists():
            continue
        if best is None or score_tuple(stats) > score_tuple(best):
            best = stats
            best_family = source_family
            best_resume_path = resume_path
    if best is None or best_resume_path is None or best_family is None:
        return None
    return allocate_endgame_basin_micro_probe(
        state,
        best_family,
        best.exp_name,
        str(best_resume_path),
        "cold_control_revisit_best_endgame_basin_micro",
    )


def recent_alt_seed_resume_source_is_cold(source_exp: str) -> bool:
    if not JOURNAL_PATH.exists():
        return False
    recent: list[tuple[float, float]] = []
    try:
        with open(JOURNAL_PATH, "r", encoding="utf-8") as f:
            for raw in reversed(f.readlines()):
                try:
                    record = json.loads(raw)
                except Exception:
                    continue
                spec = record.get("spec") or {}
                if spec.get("family") != "resume_short":
                    continue
                if spec.get("source_exp") != source_exp:
                    continue
                if "alt_seed" not in str(spec.get("reason", "")):
                    continue
                event = record.get("event")
                if event == "eval":
                    recent.append(
                        (
                            num_or(record.get("mean_score"), -1.0),
                            num_or(record.get("phase_lt20_rate"), 0.0),
                        )
                    )
                elif event == "missing_eval":
                    # A PI-killed alt-seed short-resume is already a failed probe.
                    # Treat it as cold evidence so the controller stops revisiting
                    # the same source after repeated early cuts.
                    recent.append((-1.0, 1.0))
                else:
                    continue
                if len(recent) >= ALT_SEED_COLD_WINDOW:
                    break
    except Exception:
        return False
    if len(recent) < ALT_SEED_COLD_BAD_LIMIT:
        return False
    bad = sum(
        1
        for mean_score, lt20 in recent
        if mean_score < ALT_SEED_COLD_SCORE or lt20 > ALT_SEED_COLD_LT20
    )
    return bad >= ALT_SEED_COLD_BAD_LIMIT


def allocate_historical_endbody_short_probe(
    state: dict[str, Any],
    source_exp: str,
    resume_state: str,
    seed: int,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("historical_endbody_short_probe_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= HISTORICAL_ENDBODY_SHORT_PROBE_LIMIT:
        return None
    probe_counts[source_exp] = used + 1
    return {
        "family": "resume_endgame_body_age_short",
        "seed": seed,
        "resume_state": str(resume_state),
        "source_exp": source_exp,
        "reason": reason,
    }


def discover_best_endbody_short_candidate(
    state: dict[str, Any],
    exclude_exp: str | None = None,
    min_mean_score: float = HISTORICAL_ENDBODY_SHORT_MIN_SCORE,
) -> dict[str, Any] | None:
    best: EvalStats | None = None
    best_resume_path: Path | None = None
    for run_json in EXPERIMENTS_DIR.glob("*/run.json"):
        payload = load_json(run_json)
        if not payload:
            continue
        args = payload.get("args", {})
        exp_name = args.get("exp_name")
        if not exp_name or "_endbody_" not in exp_name or not exp_name.startswith("ppo_research_"):
            continue
        if exclude_exp and exp_name == exclude_exp:
            continue
        seed = args.get("seed")
        if seed is None:
            seed = seed_from_exp_name(exp_name)
        if seed is None:
            continue
        stats = extract_best_eval(run_json.parent, exp_name, int(seed), "resume_endgame_body_age_micro")
        if stats is None or stats.phase_lt20_rate > 0.0 or stats.mean_score < min_mean_score:
            continue
        resume_path = run_json.parent / "best_eval_resume.pt"
        if not resume_path.exists():
            continue
        if best is None or score_tuple(stats) > score_tuple(best):
            best = stats
            best_resume_path = resume_path
    if best is None or best_resume_path is None:
        return None
    return allocate_historical_endbody_short_probe(
        state,
        best.exp_name,
        str(best_resume_path),
        int(best.seed),
        "cold_control_revisit_best_endbody_same_seed",
    )


def discover_best_micro_to_short_candidate(
    state: dict[str, Any],
    exclude_exp: str | None = None,
    min_mean_score: float = MICRO_TO_SHORT_REVISIT_SCORE,
) -> dict[str, Any] | None:
    best: EvalStats | None = None
    best_resume_path: Path | None = None
    suppressed_sources = state.get("best_micro_short_suppressed_sources", {})
    for run_json in EXPERIMENTS_DIR.glob("*/run.json"):
        payload = load_json(run_json)
        if not payload:
            continue
        args = payload.get("args", {})
        exp_name = args.get("exp_name")
        if not exp_name or "_micro_" not in exp_name or not exp_name.startswith("ppo_research_"):
            continue
        if exp_name in suppressed_sources:
            continue
        if exclude_exp and exp_name == exclude_exp:
            continue
        seed = args.get("seed")
        if seed is None:
            seed = seed_from_exp_name(exp_name)
        if seed is None:
            continue
        stats = extract_best_eval(run_json.parent, exp_name, int(seed), "resume_micro")
        if stats is None or stats.phase_lt20_rate > 0.0 or stats.mean_score < min_mean_score:
            continue
        resume_path = run_json.parent / "best_eval_resume.pt"
        if not resume_path.exists():
            continue
        if best is None or score_tuple(stats) > score_tuple(best):
            best = stats
            best_resume_path = resume_path
    if best is None or best_resume_path is None:
        return None
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    return {
        "family": "resume_short",
        "seed": seed,
        "resume_state": str(best_resume_path),
        "source_exp": best.exp_name,
        "chain_depth": 1,
        "reason": "cold_control_revisit_best_micro_alt_seed",
    }


def bootstrap_state() -> dict[str, Any]:
    state = load_json(STATE_PATH) or {}
    if state.get("initialized") and "queue" in state and "control_interval" in state:
        if "control_cold_streak" not in state:
            state["control_cold_streak"] = 0
        if "parent_seed_probe_counts" not in state:
            state["parent_seed_probe_counts"] = {}
        if "parent_micro_probe_counts" not in state:
            state["parent_micro_probe_counts"] = {}
        if "incumbent_micro_probe_counts" not in state:
            state["incumbent_micro_probe_counts"] = {}
        if "control_alt_micro_probe_counts" not in state:
            state["control_alt_micro_probe_counts"] = {}
        if "control_alt_nano_probe_counts" not in state:
            state["control_alt_nano_probe_counts"] = {}
        if "incumbent_nano_probe_counts" not in state:
            state["incumbent_nano_probe_counts"] = {}
        if "body_age_micro_probe_counts" not in state:
            state["body_age_micro_probe_counts"] = {}
        if "historical_endbody_short_probe_counts" not in state:
            state["historical_endbody_short_probe_counts"] = {}
        if "endgame_micro_probe_counts" not in state:
            state["endgame_micro_probe_counts"] = {}
        if "endgame_safe_probe_counts" not in state:
            state["endgame_safe_probe_counts"] = {}
        if "endgame_body_age_probe_counts" not in state:
            state["endgame_body_age_probe_counts"] = {}
        if "endgame_combo_probe_counts" not in state:
            state["endgame_combo_probe_counts"] = {}
        if "failure_harvest_counts" not in state:
            state["failure_harvest_counts"] = {}
        if "failure_harvest_results" not in state:
            state["failure_harvest_results"] = {}
        if "win_probe_counts" not in state:
            state["win_probe_counts"] = {}
        if "win_probe_results" not in state:
            state["win_probe_results"] = {}
        if "best_micro_short_suppressed_sources" not in state:
            state["best_micro_short_suppressed_sources"] = {}
        if "cold_parent_suppressed_sources" not in state:
            state["cold_parent_suppressed_sources"] = {}
        atomic_write_json(STATE_PATH, state)
        return state

    incumbent = discover_incumbent()
    max_seed = 42
    for run_json in EXPERIMENTS_DIR.glob("*/run.json"):
        payload = load_json(run_json)
        if not payload:
            continue
        args = payload.get("args", {})
        exp_name = args.get("exp_name")
        seed = args.get("seed")
        if seed is None and exp_name:
            seed = seed_from_exp_name(exp_name)
        if seed is not None:
            max_seed = max(max_seed, int(seed))

    queue: list[dict[str, Any]] = []
    if incumbent is not None:
        queue.append(
            {
                "family": "harvest_repro",
                "seed": int(incumbent.seed),
                "source_exp": incumbent.exp_name,
                "reason": "dense_repro_of_best_control_seed",
            }
        )
        resume_path = incumbent_resume_path(incumbent)
        if resume_path is not None:
            queue.append(
                {
                    "family": "resume_short",
                    "seed": int(incumbent.seed),
                    "resume_state": str(resume_path),
                    "source_exp": incumbent.exp_name,
                    "reason": "conservative_resume_probe_of_incumbent",
                }
            )

    state = {
        "initialized": True,
        "next_seed": max_seed + 1,
        "incumbent": asdict(incumbent) if incumbent else None,
        "last_run": None,
        "active_run": None,
        "loop_index": 0,
        "disk_floor_gb": DISK_FLOOR_GB,
        "queue": queue,
        "control_interval": 3,
        "since_control": 0,
        "control_cold_streak": 0,
        "parent_seed_probe_counts": {},
        "parent_micro_probe_counts": {},
        "incumbent_micro_probe_counts": {},
        "control_alt_micro_probe_counts": {},
        "control_alt_nano_probe_counts": {},
        "incumbent_nano_probe_counts": {},
        "body_age_micro_probe_counts": {},
        "historical_endbody_short_probe_counts": {},
        "endgame_micro_probe_counts": {},
        "endgame_safe_probe_counts": {},
        "endgame_body_age_probe_counts": {},
        "endgame_combo_probe_counts": {},
        "failure_harvest_counts": {},
        "failure_harvest_results": {},
        "win_probe_counts": {},
        "win_probe_results": {},
        "best_micro_short_suppressed_sources": {},
        "cold_parent_suppressed_sources": {},
    }
    atomic_write_json(STATE_PATH, state)
    if incumbent is not None:
        atomic_write_json(CURRENT_BEST_PATH, asdict(incumbent))
    return state


def control_args(seed: int) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", "300000",
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--curriculum-prob", "0.1",
        "--curriculum-min-fill", "0.9",
        "--curriculum-max-fill", "0.98",
        "--curriculum-follow-bonus", "0.005",
        "--curriculum-follow-min-fill", "0.95",
        "--head-centered",
        "--lr", "5e-6",
        "--no-anneal-lr",
        "--eval-every-steps", "250000",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--resume", str(BASE_RESUME),
        "--seed", str(seed),
    ]


def harvest_args(seed: int) -> list[str]:
    args = control_args(seed)
    args.extend([
        "--eval-every-steps", "65536",
        "--checkpoint-interval", "1",
    ])
    return args


def resume_short_args(seed: int, resume_state: str) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", "131072",
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--curriculum-prob", "0.1",
        "--curriculum-min-fill", "0.9",
        "--curriculum-max-fill", "0.98",
        "--curriculum-follow-bonus", "0.005",
        "--curriculum-follow-min-fill", "0.95",
        "--head-centered",
        "--lr", f"{RESUME_LR}",
        "--no-anneal-lr",
        "--eval-every-steps", "32768",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--checkpoint-interval", "1",
        "--resume-state", str(resume_state),
        "--resume-add-steps",
        "--override-resume-lr",
        "--seed", str(seed),
    ]


def resume_micro_args(seed: int, resume_state: str) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", str(MICRO_RESUME_TIMESTEPS),
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--curriculum-prob", "0.1",
        "--curriculum-min-fill", "0.9",
        "--curriculum-max-fill", "0.98",
        "--curriculum-follow-bonus", "0.005",
        "--curriculum-follow-min-fill", "0.95",
        "--head-centered",
        "--lr", f"{RESUME_LR}",
        "--no-anneal-lr",
        "--eval-every-steps", "16384",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--checkpoint-interval", "1",
        "--resume-state", str(resume_state),
        "--resume-add-steps",
        "--override-resume-lr",
        "--seed", str(seed),
    ]


def resume_nano_args(seed: int, resume_state: str) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", str(NANO_RESUME_TIMESTEPS),
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--curriculum-prob", "0.1",
        "--curriculum-min-fill", "0.9",
        "--curriculum-max-fill", "0.98",
        "--curriculum-follow-bonus", "0.005",
        "--curriculum-follow-min-fill", "0.95",
        "--head-centered",
        "--lr", f"{NANO_RESUME_LR}",
        "--no-anneal-lr",
        "--eval-every-steps", "8192",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--checkpoint-interval", "1",
        "--resume-state", str(resume_state),
        "--resume-add-steps",
        "--override-resume-lr",
        "--seed", str(seed),
    ]


def resume_micro_body_age_args(seed: int, resume_state: str) -> list[str]:
    resume_model = str(Path(resume_state).with_name("best_eval.pt"))
    return [
        "--board-size", "20",
        "--timesteps", str(MICRO_RESUME_TIMESTEPS),
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--curriculum-prob", "0.1",
        "--curriculum-min-fill", "0.9",
        "--curriculum-max-fill", "0.98",
        "--curriculum-follow-bonus", "0.005",
        "--curriculum-follow-min-fill", "0.95",
        "--head-centered",
        "--lr", f"{RESUME_LR}",
        "--no-anneal-lr",
        "--eval-every-steps", "16384",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--checkpoint-interval", "1",
        "--resume", resume_model,
        "--seed", str(seed),
        "--aux-body-age-target",
        "--aux-body-age-target-coef", "0.25",
        "--aux-body-age-target-min-fill", "0.80",
    ]


def resume_endgame_micro_args(seed: int, resume_state: str) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", str(MICRO_RESUME_TIMESTEPS),
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--curriculum-prob", "0.2",
        "--curriculum-min-fill", "0.96",
        "--curriculum-max-fill", "0.995",
        "--curriculum-follow-bonus", "0.01",
        "--curriculum-follow-min-fill", "0.97",
        "--head-centered",
        "--lr", f"{RESUME_LR}",
        "--no-anneal-lr",
        "--eval-every-steps", "16384",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--checkpoint-interval", "1",
        "--resume-state", str(resume_state),
        "--resume-add-steps",
        "--override-resume-lr",
        "--seed", str(seed),
    ]


def resume_endgame_safe_micro_args(seed: int, resume_state: str) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", str(MICRO_RESUME_TIMESTEPS),
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--curriculum-prob", "0.2",
        "--curriculum-min-fill", "0.96",
        "--curriculum-max-fill", "0.995",
        "--curriculum-follow-bonus", "0.01",
        "--curriculum-follow-min-fill", "0.97",
        "--aux-safe-action-soft-target",
        "--aux-safe-action-soft-target-coef", "0.25",
        "--aux-safe-action-soft-target-min-fill", "0.98",
        "--aux-safe-action-soft-temperature", "1.0",
        "--safe-action-bonus", "0.005",
        "--safe-action-bonus-min-fill", "0.98",
        "--late-confidence-coef", "0.01",
        "--late-confidence-min-fill", "0.98",
        "--head-centered",
        "--lr", f"{RESUME_LR}",
        "--no-anneal-lr",
        "--eval-every-steps", "16384",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--checkpoint-interval", "1",
        "--resume-state", str(resume_state),
        "--resume-add-steps",
        "--override-resume-lr",
        "--seed", str(seed),
    ]


def resume_endgame_body_age_micro_args(seed: int, resume_state: str) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", str(MICRO_RESUME_TIMESTEPS),
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--curriculum-prob", "0.2",
        "--curriculum-min-fill", "0.96",
        "--curriculum-max-fill", "0.995",
        "--curriculum-follow-bonus", "0.01",
        "--curriculum-follow-min-fill", "0.97",
        "--aux-body-age-target",
        "--aux-body-age-target-coef", "0.25",
        "--aux-body-age-target-min-fill", "0.98",
        "--late-confidence-coef", "0.01",
        "--late-confidence-min-fill", "0.98",
        "--head-centered",
        "--lr", f"{RESUME_LR}",
        "--no-anneal-lr",
        "--eval-every-steps", "16384",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--checkpoint-interval", "1",
        "--resume-state", str(resume_state),
        "--resume-add-steps",
        "--override-resume-lr",
        "--seed", str(seed),
    ]


def resume_endgame_body_age_short_args(seed: int, resume_state: str) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", "131072",
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--curriculum-prob", "0.2",
        "--curriculum-min-fill", "0.96",
        "--curriculum-max-fill", "0.995",
        "--curriculum-follow-bonus", "0.01",
        "--curriculum-follow-min-fill", "0.97",
        "--aux-body-age-target",
        "--aux-body-age-target-coef", "0.25",
        "--aux-body-age-target-min-fill", "0.98",
        "--late-confidence-coef", "0.01",
        "--late-confidence-min-fill", "0.98",
        "--head-centered",
        "--lr", f"{RESUME_LR}",
        "--no-anneal-lr",
        "--eval-every-steps", "32768",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--checkpoint-interval", "1",
        "--resume-state", str(resume_state),
        "--resume-add-steps",
        "--override-resume-lr",
        "--seed", str(seed),
    ]


def resume_endgame_body_age_obs_short_args(seed: int, resume_state: str) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", "131072",
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--body-age-obs",
        "--body-age-obs-min-fill", "0.95",
        "--curriculum-prob", "0.2",
        "--curriculum-min-fill", "0.96",
        "--curriculum-max-fill", "0.995",
        "--curriculum-follow-bonus", "0.01",
        "--curriculum-follow-min-fill", "0.97",
        "--head-centered",
        "--lr", f"{RESUME_LR}",
        "--no-anneal-lr",
        "--eval-every-steps", "16384",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--checkpoint-interval", "1",
        "--resume-state", str(resume_state),
        "--resume-add-steps",
        "--override-resume-lr",
        "--seed", str(seed),
    ]


def resume_history_short_args(seed: int, resume_state: str) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", "131072",
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--obs-history", "2",
        "--curriculum-prob", "0.1",
        "--curriculum-min-fill", "0.9",
        "--curriculum-max-fill", "0.98",
        "--curriculum-follow-bonus", "0.005",
        "--curriculum-follow-min-fill", "0.95",
        "--head-centered",
        "--lr", f"{RESUME_LR}",
        "--no-anneal-lr",
        "--eval-every-steps", "32768",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--checkpoint-interval", "1",
        "--resume-state", str(resume_state),
        "--resume-add-steps",
        "--override-resume-lr",
        "--seed", str(seed),
    ]


def resume_actionhist_short_args(seed: int, resume_state: str) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", "131072",
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--action-history-obs", "2",
        "--curriculum-prob", "0.1",
        "--curriculum-min-fill", "0.9",
        "--curriculum-max-fill", "0.98",
        "--curriculum-follow-bonus", "0.005",
        "--curriculum-follow-min-fill", "0.95",
        "--head-centered",
        "--lr", f"{RESUME_LR}",
        "--no-anneal-lr",
        "--eval-every-steps", "32768",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--checkpoint-interval", "1",
        "--resume-state", str(resume_state),
        "--resume-add-steps",
        "--override-resume-lr",
        "--seed", str(seed),
    ]


def resume_endgame_combo_micro_args(seed: int, resume_state: str) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", str(MICRO_RESUME_TIMESTEPS),
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--curriculum-prob", "0.2",
        "--curriculum-min-fill", "0.96",
        "--curriculum-max-fill", "0.995",
        "--curriculum-follow-bonus", "0.01",
        "--curriculum-follow-min-fill", "0.97",
        "--aux-body-age-target",
        "--aux-body-age-target-coef", "0.25",
        "--aux-body-age-target-min-fill", "0.98",
        "--aux-safe-action-soft-target",
        "--aux-safe-action-soft-target-coef", "0.25",
        "--aux-safe-action-soft-target-min-fill", "0.98",
        "--aux-safe-action-soft-temperature", "1.0",
        "--safe-action-bonus", "0.005",
        "--safe-action-bonus-min-fill", "0.98",
        "--late-confidence-coef", "0.01",
        "--late-confidence-min-fill", "0.98",
        "--head-centered",
        "--lr", f"{RESUME_LR}",
        "--no-anneal-lr",
        "--eval-every-steps", "16384",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--checkpoint-interval", "1",
        "--resume-state", str(resume_state),
        "--resume-add-steps",
        "--override-resume-lr",
        "--seed", str(seed),
    ]


def resume_endgame_elitebc_micro_args(seed: int, resume_state: str) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", str(MICRO_RESUME_TIMESTEPS),
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--curriculum-prob", "0.2",
        "--curriculum-min-fill", "0.96",
        "--curriculum-max-fill", "0.995",
        "--curriculum-follow-bonus", "0.01",
        "--curriculum-follow-min-fill", "0.97",
        "--aux-body-age-target",
        "--aux-body-age-target-coef", "0.25",
        "--aux-body-age-target-min-fill", "0.98",
        "--aux-safe-action-soft-target",
        "--aux-safe-action-soft-target-coef", "0.25",
        "--aux-safe-action-soft-target-min-fill", "0.98",
        "--aux-safe-action-soft-temperature", "1.0",
        "--safe-action-bonus", "0.005",
        "--safe-action-bonus-min-fill", "0.98",
        "--late-confidence-coef", "0.01",
        "--late-confidence-min-fill", "0.98",
        "--elite-bc-coef", "0.10",
        "--elite-score-threshold", "390",
        "--elite-min-fill", "0.98",
        "--elite-buffer-size", "32768",
        "--elite-bc-start-steps", "0",
        "--head-centered",
        "--lr", f"{RESUME_LR}",
        "--no-anneal-lr",
        "--eval-every-steps", "16384",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--checkpoint-interval", "1",
        "--resume-state", str(resume_state),
        "--resume-add-steps",
        "--override-resume-lr",
        "--seed", str(seed),
    ]


def resume_endgame_sharpen_micro_args(seed: int, resume_state: str) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", str(MICRO_RESUME_TIMESTEPS),
        "--num-envs", "64",
        "--horizon", "256",
        "--minibatch-size", "4096",
        "--symmetric",
        "--device", "cpu",
        "--gamma", "0.999",
        "--gae-lambda", "0.9",
        "--vf-clip-coef", "1.0",
        "--network-scale", "2",
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef", "1.0",
        "--curriculum-prob", "0.2",
        "--curriculum-min-fill", "0.96",
        "--curriculum-max-fill", "0.995",
        "--curriculum-follow-bonus", "0.01",
        "--curriculum-follow-min-fill", "0.97",
        "--late-confidence-coef", "0.02",
        "--late-confidence-min-fill", "0.98",
        "--head-centered",
        "--lr", f"{RESUME_LR}",
        "--ent-coef", "0.005",
        "--no-anneal-lr",
        "--eval-every-steps", "16384",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--checkpoint-interval", "1",
        "--resume-state", str(resume_state),
        "--resume-add-steps",
        "--override-resume-lr",
        "--seed", str(seed),
    ]


def failure_harvest_args(spec: dict[str, Any]) -> list[str]:
    return [
        "--checkpoint", str(spec["checkpoint"]),
        "--source-exp", str(spec["source_exp"]),
        "--source-run-dir", str(spec["source_run_dir"]),
        "--episodes", str(FAILURE_HARVEST_EPISODES),
        "--harvest-limit", str(FAILURE_HARVEST_TARGET_FAILURES),
        "--seed", str(int(spec["seed"])),
        "--device", "cpu",
        "--min-score", str(FAILURE_HARVEST_MIN_SCORE),
        "--max-score", str(FAILURE_HARVEST_MAX_SCORE),
    ]


def win_probe_args(spec: dict[str, Any]) -> list[str]:
    return [
        "--checkpoint", str(spec["checkpoint"]),
        "--source-exp", str(spec["source_exp"]),
        "--source-run-dir", str(spec["source_run_dir"]),
        "--episodes", str(WIN_PROBE_EPISODES),
        "--seed", str(int(spec["seed"])),
        "--device", "cpu",
    ]


def build_spec(state: dict[str, Any]) -> dict[str, Any]:
    queue = state.setdefault("queue", [])
    if queue:
        return queue.pop(0)

    if int(state.get("since_control", 0)) >= int(state.get("control_interval", 3)):
        seed = int(state["next_seed"])
        state["next_seed"] = seed + 1
        state["since_control"] = 0
        return {"family": "control", "seed": seed, "reason": "periodic_control_refresh"}

    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    state["since_control"] = int(state.get("since_control", 0)) + 1
    return {"family": "control", "seed": seed, "reason": "default_control"}


def exp_name_for_spec(spec: dict[str, Any], loop_index: int) -> str:
    family = spec["family"]
    seed = int(spec["seed"])
    short = {
        "control": "ctrl",
        "harvest_repro": "harvest",
        "failure_harvest": "failharv",
        "win_probe": "winprobe",
        "resume_short": "resume",
        "resume_micro": "micro",
        "resume_nano": "nano",
        "resume_micro_body_age": "bodyage",
        "resume_endgame_micro": "endmicro",
        "resume_endgame_safe_micro": "endsafe",
        "resume_endgame_body_age_micro": "endbody",
        "resume_endgame_body_age_short": "endbshort",
        "resume_endgame_body_age_obs_short": "endbodyobs",
        "resume_history_short": "hist2",
        "resume_actionhist_short": "acthist",
        "resume_endgame_combo_micro": "endmix",
        "resume_endgame_elitebc_micro": "endelite",
        "resume_endgame_sharpen_micro": "endsharp",
    }.get(family, family)
    return f"ppo_research_{loop_index:03d}_{short}_s{seed}"


def args_for_spec(spec: dict[str, Any]) -> list[str]:
    family = spec["family"]
    seed = int(spec["seed"])
    if family == "control":
        return control_args(seed)
    if family == "harvest_repro":
        return harvest_args(seed)
    if family == "failure_harvest":
        return failure_harvest_args(spec)
    if family == "win_probe":
        return win_probe_args(spec)
    if family == "resume_short":
        return resume_short_args(seed, spec["resume_state"])
    if family == "resume_micro":
        return resume_micro_args(seed, spec["resume_state"])
    if family == "resume_nano":
        return resume_nano_args(seed, spec["resume_state"])
    if family == "resume_micro_body_age":
        return resume_micro_body_age_args(seed, spec["resume_state"])
    if family == "resume_endgame_micro":
        return resume_endgame_micro_args(seed, spec["resume_state"])
    if family == "resume_endgame_safe_micro":
        return resume_endgame_safe_micro_args(seed, spec["resume_state"])
    if family == "resume_endgame_body_age_micro":
        return resume_endgame_body_age_micro_args(seed, spec["resume_state"])
    if family == "resume_endgame_body_age_short":
        return resume_endgame_body_age_short_args(seed, spec["resume_state"])
    if family == "resume_endgame_body_age_obs_short":
        return resume_endgame_body_age_obs_short_args(seed, spec["resume_state"])
    if family == "resume_history_short":
        return resume_history_short_args(seed, spec["resume_state"])
    if family == "resume_actionhist_short":
        return resume_actionhist_short_args(seed, spec["resume_state"])
    if family == "resume_endgame_combo_micro":
        return resume_endgame_combo_micro_args(seed, spec["resume_state"])
    if family == "resume_endgame_elitebc_micro":
        return resume_endgame_elitebc_micro_args(seed, spec["resume_state"])
    if family == "resume_endgame_sharpen_micro":
        return resume_endgame_sharpen_micro_args(seed, spec["resume_state"])
    raise ValueError(f"unknown family: {family}")


def script_for_spec(spec: dict[str, Any]) -> Path:
    if spec["family"] == "failure_harvest":
        return HARVEST_FAILURES
    if spec["family"] == "win_probe":
        return WIN_PROBE
    return TRAIN


def wait_for_run_dir(exp_name: str, timeout_seconds: int = 120) -> Path | None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        matches = sorted(
            EXPERIMENTS_DIR.glob(f"{exp_name}_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]
        time.sleep(1)
    matches = sorted(
        EXPERIMENTS_DIR.glob(f"{exp_name}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def run_experiment(exp_name: str, spec: dict[str, Any], args: list[str], state: dict[str, Any]) -> tuple[Path | None, int]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_path = EXPERIMENTS_DIR / f"{exp_name}.driver.log"
    script = script_for_spec(spec)
    cmd = [str(PYTHON), str(script), *args, "--exp-name", exp_name]
    log(f"launch {exp_name} family={spec['family']}")
    append_journal("launch", exp_name=exp_name, spec=spec, command=cmd)
    with open(log_path, "w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        run_dir = wait_for_run_dir(exp_name)
        if run_dir is not None:
            state["active_run"] = {
                "exp_name": exp_name,
                "run_dir": str(run_dir),
                "pid": proc.pid,
                "driver_log": str(log_path),
                "spec": spec,
            }
            atomic_write_json(STATE_PATH, state)
        exit_code = wait_for_process_with_stall_guard(
            pid=proc.pid,
            exp_name=exp_name,
            run_dir=run_dir,
            driver_log=log_path,
            proc=proc,
        )
    log(f"exit {exp_name} code={exit_code}")
    append_journal("run_complete", exp_name=exp_name, exit_code=exit_code, run_dir=str(run_dir) if run_dir else None)
    return run_dir, exit_code


def cleanup_run(run_dir: Path) -> dict[str, Any]:
    removed = 0
    freed = 0
    pat = re.compile(r"model.*_\d{6}\.pt$|model_\d{6}\.pt$")
    for path in run_dir.rglob("*.pt"):
        if pat.search(path.name) and "best" not in path.name:
            try:
                freed += path.stat().st_size
                path.unlink()
                removed += 1
            except FileNotFoundError:
                pass
    duplicate_dir_removed = False
    duplicate_dir_freed = 0
    run_id = run_dir.name.rsplit("_", 1)[-1]
    duplicate_dir = run_dir.parent / run_id
    if duplicate_dir.is_dir():
        duplicate_files = [p for p in duplicate_dir.rglob("*") if p.is_file()]
        # Some runs leave behind a bare numeric directory that only contains transient
        # checkpoint artifacts. The named run dir preserves the actual experiment results.
        if duplicate_files and all(p.suffix == ".pt" for p in duplicate_files):
            try:
                duplicate_dir_freed = sum(p.stat().st_size for p in duplicate_files)
                shutil.rmtree(duplicate_dir)
                duplicate_dir_removed = True
            except FileNotFoundError:
                pass
    payload = {
        "run_dir": str(run_dir),
        "removed_files": removed,
        "freed_bytes": freed,
        "duplicate_dir_removed": duplicate_dir_removed,
        "duplicate_dir_freed_bytes": duplicate_dir_freed,
    }
    append_journal("cleanup", **payload)
    return payload


def should_update_incumbent(current: EvalStats | None, challenger: EvalStats) -> bool:
    if current is None:
        return True
    return score_tuple(challenger) > score_tuple(current)


def maybe_enqueue_followups(state: dict[str, Any], spec: dict[str, Any], stats: EvalStats, run_dir: Path) -> None:
    queue = state.setdefault("queue", [])
    family = spec["family"]

    if family == "control":
        resume_path = run_dir / "best_eval_resume.pt"
        if stats.mean_score >= HARVEST_GATE_SCORE and stats.phase_lt20_rate == 0.0:
            queue.append(
                {
                    "family": "harvest_repro",
                    "seed": int(spec["seed"]),
                    "source_exp": stats.exp_name,
                    "reason": "rare_exceptional_control_dense_harvest",
                }
            )
        if stats.mean_score >= CONTROL_MICRO_SCORE and stats.phase_lt20_rate == 0.0 and resume_path.exists():
            queue.insert(
                0,
                {
                    "family": "resume_micro",
                    "seed": int(spec["seed"]),
                    "resume_state": str(resume_path),
                    "source_exp": stats.exp_name,
                    "reason": "exceptional_control_micro_resume",
                }
            )
        elif stats.mean_score >= CONTROL_RESUME_SCORE and resume_path.exists():
                queue.append(
                    {
                        "family": "resume_short",
                        "seed": int(spec["seed"]),
                        "resume_state": str(resume_path),
                        "source_exp": stats.exp_name,
                        "chain_depth": 1,
                        "reason": "strong_control_conservative_resume",
                    }
                )

    if family == "harvest_repro" and stats.mean_score >= CONTROL_PROMOTE_SCORE:
        resume_path = run_dir / "best_eval_resume.pt"
        if resume_path.exists():
            queue.append(
                {
                    "family": "resume_short",
                    "seed": int(spec["seed"]),
                    "resume_state": str(resume_path),
                    "source_exp": stats.exp_name,
                    "chain_depth": 1,
                    "reason": "harvest_peak_conservative_resume",
                }
            )

    if family == "resume_short" and stats.mean_score >= RESUME_CHAIN_SCORE and stats.phase_lt20_rate == 0.0:
        if spec.get("reason") == "cold_control_revisit_best_micro" and stats.mean_score < RESUME_MICRO_CHAIN_SCORE:
            suppressed = state.setdefault("best_micro_short_suppressed_sources", {})
            source_exp = spec.get("source_exp")
            if source_exp:
                suppressed[str(source_exp)] = {
                    "failed_exp": stats.exp_name,
                    "mean_score": stats.mean_score,
                }
            append_journal(
                "best_micro_resume_chain_suppressed",
                basis_exp=stats.exp_name,
                mean_score=stats.mean_score,
                source_exp=spec.get("source_exp"),
                reason=spec.get("reason"),
            )
            log(
                f"best-micro revisit: suppress deeper chain after {stats.exp_name} "
                f"mean={stats.mean_score:.2f}"
            )
            return
        if spec.get("reason") in {"cold_control_revisit_best_micro_alt_seed", "pi_alt_seed_micro_to_short_revisit"} and stats.mean_score < RESUME_MICRO_CHAIN_SCORE:
            suppressed = state.setdefault("best_micro_short_suppressed_sources", {})
            source_exp = spec.get("source_exp")
            if source_exp:
                suppressed[str(source_exp)] = {
                    "failed_exp": stats.exp_name,
                    "mean_score": stats.mean_score,
                }
            append_journal(
                "best_micro_alt_seed_resume_source_suppressed",
                basis_exp=stats.exp_name,
                mean_score=stats.mean_score,
                source_exp=spec.get("source_exp"),
                reason=spec.get("reason"),
            )
            log(
                f"best-micro alt-seed revisit: suppress source {spec.get('source_exp')} "
                f"after {stats.exp_name} mean={stats.mean_score:.2f}"
            )
            return
        if "alt_seed" in str(spec.get("reason", "")) and stats.mean_score < RESUME_MICRO_CHAIN_SCORE:
            append_journal(
                "resume_short_alt_seed_chain_suppressed",
                basis_exp=stats.exp_name,
                mean_score=stats.mean_score,
                source_exp=spec.get("source_exp"),
                reason=spec.get("reason"),
            )
            log(
                f"resume-short alt-seed chain: suppress deeper chain after {stats.exp_name} "
                f"mean={stats.mean_score:.2f}"
            )
            return
        if spec.get("reason") == "parent_seed_sweep_followup":
            append_journal(
                "parent_seed_chain_suppressed",
                basis_exp=stats.exp_name,
                mean_score=stats.mean_score,
                source_exp=spec.get("source_exp"),
            )
            log(
                f"parent-seed sweep: suppress deeper same-seed chain after {stats.exp_name} "
                f"mean={stats.mean_score:.2f}"
            )
            return
        if (
            spec.get("reason") in {"cold_control_alt_seed_from_incumbent_parent", "parent_seed_sweep_followup"}
            and stats.mean_score >= CONTROL_MICRO_SCORE
            and stats.mean_score < HARVEST_GATE_SCORE
        ):
            sibling_probe = allocate_parent_seed_probe(
                state,
                str(spec["source_exp"]),
                str(spec["resume_state"]),
                "parent_seed_sweep_followup",
            )
            if sibling_probe is not None:
                queue.insert(0, sibling_probe)
                append_journal(
                    "parent_seed_followup_enqueued",
                    basis_exp=stats.exp_name,
                    mean_score=stats.mean_score,
                    candidate=sibling_probe,
                )
                log(
                    f"parent-seed sweep: enqueue sibling continuation from {spec['source_exp']} "
                    f"seed={sibling_probe['seed']} after {stats.exp_name} mean={stats.mean_score:.2f}"
                )
                return
        resume_path = run_dir / "best_eval_resume.pt"
        if resume_path.exists():
            chain_depth = int(spec.get("chain_depth", 1))
            if stats.mean_score < RESUME_SHORT_CHAIN_MAX and chain_depth < 2:
                queue.insert(
                    0,
                    {
                        "family": "resume_short",
                        "seed": int(spec["seed"]),
                        "resume_state": str(resume_path),
                        "source_exp": stats.exp_name,
                        "chain_depth": chain_depth + 1,
                        "reason": "strong_resume_short_chain",
                    }
                )
            elif stats.mean_score >= RESUME_SHORT_CHAIN_MAX:
                queue.insert(
                    0,
                    {
                        "family": "resume_micro",
                        "seed": int(spec["seed"]),
                        "resume_state": str(resume_path),
                        "source_exp": stats.exp_name,
                        "reason": "strong_resume_micro_probe",
                    }
                )
            else:
                append_journal(
                    "resume_short_chain_suppressed",
                    basis_exp=stats.exp_name,
                    mean_score=stats.mean_score,
                    chain_depth=chain_depth,
                    source_exp=spec.get("source_exp"),
                )
                log(
                    f"resume-short chain: suppress micro follow-up after {stats.exp_name} "
                    f"mean={stats.mean_score:.2f} chain_depth={chain_depth}"
                )

    if family == "resume_short" and spec.get("reason") == "cold_control_revisit_best_resume_alt_seed":
        if stats.mean_score < 340.0 or stats.phase_lt20_rate > 0.0:
            historical_endbody = discover_best_endbody_short_candidate(
                state,
                exclude_exp=(stats.exp_name if stats.exp_name else None),
            )
            if historical_endbody is not None:
                queue.insert(0, historical_endbody)
                append_journal(
                    "weak_alt_seed_resume_endbody_escalation",
                    basis_exp=stats.exp_name,
                    mean_score=stats.mean_score,
                    lt20=stats.phase_lt20_rate,
                    candidate=historical_endbody,
                )
                log(
                    f"resume-short alt-seed weak: escalate from {stats.exp_name} "
                    f"to historical endbody basin {historical_endbody['source_exp']} "
                    f"seed={historical_endbody['seed']}"
                )
                return

    if family == "resume_micro" and stats.mean_score >= RESUME_MICRO_CHAIN_SCORE and stats.phase_lt20_rate == 0.0:
        resume_path = run_dir / "best_eval_resume.pt"
        chain_depth = int(spec.get("chain_depth", 1))
        if resume_path.exists() and stats.mean_score >= RESUME_NANO_TRIGGER_SCORE:
            queue.insert(
                0,
                {
                    "family": "resume_nano",
                    "seed": int(spec["seed"]),
                    "resume_state": str(resume_path),
                    "source_exp": stats.exp_name,
                    "reason": "frontier_micro_to_nano",
                    "chain_depth": 1,
                }
            )
        elif resume_path.exists() and chain_depth < 2:
            queue.insert(
                0,
                {
                    "family": "resume_micro",
                    "seed": int(spec["seed"]),
                    "resume_state": str(resume_path),
                    "source_exp": stats.exp_name,
                    "reason": "strong_resume_micro_chain",
                    "chain_depth": chain_depth + 1,
                }
            )

    if family == "resume_nano" and stats.mean_score >= RESUME_NANO_CHAIN_SCORE and stats.phase_lt20_rate == 0.0:
        resume_path = run_dir / "best_eval_resume.pt"
        chain_depth = int(spec.get("chain_depth", 1))
        if resume_path.exists() and chain_depth < 2:
            queue.insert(
                0,
                {
                    "family": "resume_nano",
                    "seed": int(spec["seed"]),
                    "resume_state": str(resume_path),
                    "source_exp": stats.exp_name,
                    "reason": "strong_resume_nano_chain",
                    "chain_depth": chain_depth + 1,
                }
            )

    if family == "resume_micro_body_age" and stats.mean_score >= CONTROL_MICRO_SCORE and stats.phase_lt20_rate == 0.0:
        resume_path = run_dir / "best_eval_resume.pt"
        chain_depth = int(spec.get("chain_depth", 1))
        if resume_path.exists() and chain_depth < 2:
            queue.insert(
                0,
                {
                    "family": "resume_micro_body_age",
                    "seed": int(spec["seed"]),
                    "resume_state": str(resume_path),
                    "source_exp": stats.exp_name,
                    "reason": "body_age_micro_chain",
                    "chain_depth": chain_depth + 1,
                }
            )

    if family == "resume_endgame_safe_micro" and stats.mean_score >= CONTROL_MICRO_SCORE and stats.phase_lt20_rate == 0.0:
        resume_path = run_dir / "best_eval_resume.pt"
        chain_depth = int(spec.get("chain_depth", 1))
        if resume_path.exists() and chain_depth < 2:
            queue.insert(
                0,
                {
                    "family": "resume_endgame_safe_micro",
                    "seed": int(spec["seed"]),
                    "resume_state": str(resume_path),
                    "source_exp": stats.exp_name,
                    "reason": "endgame_softsafe_chain",
                    "chain_depth": chain_depth + 1,
                }
            )
        if resume_path.exists() and stats.mean_score >= 378.0:
            queue.insert(
                0,
                {
                    "family": "resume_endgame_combo_micro",
                    "seed": int(spec["seed"]),
                    "resume_state": str(resume_path),
                    "source_exp": stats.exp_name,
                    "reason": "endgame_softsafe_combo_followup",
                }
            )

    if family == "resume_endgame_body_age_micro" and stats.mean_score >= 380.0 and stats.phase_lt20_rate == 0.0:
        resume_path = run_dir / "best_eval_resume.pt"
        if resume_path.exists():
            queue.insert(
                0,
                {
                    "family": "resume_endgame_combo_micro",
                    "seed": int(spec["seed"]),
                    "resume_state": str(resume_path),
                    "source_exp": stats.exp_name,
                    "reason": "endgame_body_age_combo_followup",
                }
            )

    if family == "resume_endgame_body_age_short" and stats.mean_score >= 380.0 and stats.phase_lt20_rate == 0.0:
        resume_path = run_dir / "best_eval_resume.pt"
        if resume_path.exists():
            queue.insert(
                0,
                {
                    "family": "resume_endgame_combo_micro",
                    "seed": int(spec["seed"]),
                    "resume_state": str(resume_path),
                    "source_exp": stats.exp_name,
                    "reason": "endgame_body_age_combo_followup",
                }
            )


def discover_failure_harvest_probe(state: dict[str, Any], incumbent: EvalStats | None) -> dict[str, Any] | None:
    if incumbent is None:
        return None
    if incumbent.win_rate > 0.0 or incumbent.phase_lt20_rate > 0.0:
        return None
    if incumbent.phase_gte95_rate < 1.0 or incumbent.mean_score < 390.0:
        return None
    harvested = state.setdefault("failure_harvest_results", {})
    if incumbent.exp_name in harvested:
        return None
    run_dir = Path(incumbent.run_dir)
    checkpoint = run_dir / "best_eval.pt"
    if not checkpoint.exists():
        return None
    return allocate_failure_harvest_probe(
        state,
        incumbent.exp_name,
        str(run_dir),
        str(checkpoint),
        int(incumbent.seed),
        "near_perfect_failure_harvest",
    )


def discover_incumbent_win_probe(state: dict[str, Any], incumbent: EvalStats | None) -> dict[str, Any] | None:
    if incumbent is None:
        return None
    if incumbent.win_rate > 0.0 or incumbent.phase_lt20_rate > 0.0:
        return None
    if incumbent.phase_gte95_rate < 0.95 or incumbent.mean_score < 390.0:
        return None
    probed = state.setdefault("win_probe_results", {})
    if incumbent.exp_name in probed:
        return None
    run_dir = Path(incumbent.run_dir)
    checkpoint = run_dir / "best_eval.pt"
    if not checkpoint.exists():
        return None
    return allocate_win_probe(
        state,
        incumbent.exp_name,
        str(run_dir),
        str(checkpoint),
        int(incumbent.seed),
        "near_perfect_win_probe",
    )


def maybe_enqueue_incumbent_failure_harvest(state: dict[str, Any], incumbent: EvalStats | None) -> None:
    if incumbent is None:
        return
    queue = state.setdefault("queue", [])
    if any(
        spec.get("family") == "failure_harvest" and spec.get("source_exp") == incumbent.exp_name
        for spec in queue
    ):
        return
    probe = discover_failure_harvest_probe(state, incumbent)
    if probe is None:
        return
    queue.insert(0, probe)
    append_journal("failure_harvest_enqueued", incumbent=incumbent.exp_name, candidate=probe)
    log(f"enqueue failure harvest for near-perfect incumbent {incumbent.exp_name}")


def maybe_enqueue_incumbent_win_probe(state: dict[str, Any], incumbent: EvalStats | None) -> None:
    if incumbent is None:
        return
    queue = state.setdefault("queue", [])
    if any(
        spec.get("family") == "win_probe" and spec.get("source_exp") == incumbent.exp_name
        for spec in queue
    ):
        return
    probe = discover_incumbent_win_probe(state, incumbent)
    if probe is None:
        return
    queue.insert(0, probe)
    append_journal("win_probe_enqueued", incumbent=incumbent.exp_name, candidate=probe)
    log(f"enqueue win probe for near-perfect incumbent {incumbent.exp_name}")


def maybe_enqueue_cold_streak_research(state: dict[str, Any], incumbent: EvalStats | None) -> None:
    streak = int(state.get("control_cold_streak", 0))
    queue = state.setdefault("queue", [])
    if streak < CONTROL_COLD_STREAK_LIMIT or queue:
        return
    prefer_local_variance = (
        incumbent is not None
        and float(incumbent.mean_score) >= 390.0
        and float(incumbent.phase_lt20_rate) == 0.0
    )
    if prefer_local_variance:
        incumbent_nano = discover_incumbent_nano_probe(state, incumbent)
        if incumbent_nano is not None:
            queue.append(incumbent_nano)
            state["control_cold_streak"] = 0
            append_journal("cold_streak_incumbent_nano_enqueued", streak=streak, candidate=incumbent_nano)
            log(
                "cold control streak="
                f"{streak}; enqueue frontier nano resume "
                f"seed={incumbent_nano['seed']}"
            )
            return
    if not prefer_local_variance:
        safe_endgame_probe = discover_endgame_safe_probe(state, incumbent)
        if safe_endgame_probe is not None:
            queue.append(safe_endgame_probe)
            state["control_cold_streak"] = 0
            append_journal("cold_streak_endgame_safe_micro_enqueued", streak=streak, candidate=safe_endgame_probe)
            log(
                "cold control streak="
                f"{streak}; enqueue safe endgame micro from {safe_endgame_probe['source_exp']}"
            )
            return
        combo_endgame_probe = discover_endgame_combo_probe(state, incumbent)
        if combo_endgame_probe is not None:
            queue.append(combo_endgame_probe)
            state["control_cold_streak"] = 0
            append_journal("cold_streak_endgame_combo_micro_enqueued", streak=streak, candidate=combo_endgame_probe)
            log(
                "cold control streak="
                f"{streak}; enqueue endgame combo micro from {combo_endgame_probe['source_exp']}"
            )
            return
        body_age_endgame_probe = discover_endgame_body_age_probe(state, incumbent)
        if body_age_endgame_probe is not None:
            queue.append(body_age_endgame_probe)
            state["control_cold_streak"] = 0
            append_journal("cold_streak_endgame_body_age_micro_enqueued", streak=streak, candidate=body_age_endgame_probe)
            log(
                "cold control streak="
                f"{streak}; enqueue endgame body-age micro from {body_age_endgame_probe['source_exp']}"
            )
            return
        endgame_probe = discover_endgame_micro_probe(state, incumbent)
        if endgame_probe is not None:
            queue.append(endgame_probe)
            state["control_cold_streak"] = 0
            append_journal("cold_streak_endgame_micro_enqueued", streak=streak, candidate=endgame_probe)
            log(
                "cold control streak="
                f"{streak}; enqueue near-perfect endgame micro from {endgame_probe['source_exp']}"
            )
            return
    min_mean_score = RESUME_CHAIN_SCORE
    if incumbent is not None:
        min_mean_score = max(min_mean_score, float(incumbent.mean_score) - 5.0)
    candidate = discover_best_resume_candidate(
        exclude_exp=(incumbent.exp_name if incumbent is not None else None),
        min_mean_score=min_mean_score,
    )
    if candidate is None:
        micro_short = discover_best_micro_to_short_candidate(
            state,
            exclude_exp=(incumbent.exp_name if incumbent is not None else None),
            min_mean_score=max(MICRO_TO_SHORT_REVISIT_SCORE, min_mean_score - 4.0),
        )
        if micro_short is not None:
            queue.append(micro_short)
            state["control_cold_streak"] = 0
            append_journal("cold_streak_micro_to_short_enqueued", streak=streak, candidate=micro_short)
            log(
                "cold control streak="
                f"{streak}; enqueue conservative short resume from strong micro basin "
                f"{micro_short['source_exp']}"
            )
            return
        historical_endbody = discover_best_endbody_short_candidate(
            state,
            exclude_exp=(incumbent.exp_name if incumbent is not None else None),
        )
        if historical_endbody is not None:
            queue.append(historical_endbody)
            state["control_cold_streak"] = 0
            append_journal("cold_streak_historical_endbody_short_enqueued", streak=streak, candidate=historical_endbody)
            log(
                "cold control streak="
                f"{streak}; enqueue short continuation from historical endbody basin "
                f"{historical_endbody['source_exp']} seed={historical_endbody['seed']}"
            )
            return
        incumbent_micro = discover_incumbent_micro_probe(state, incumbent)
        if incumbent_micro is not None:
            queue.append(incumbent_micro)
            state["control_cold_streak"] = 0
            append_journal("cold_streak_incumbent_micro_enqueued", streak=streak, candidate=incumbent_micro)
            log(
                "cold control streak="
                f"{streak}; enqueue fresh-seeded incumbent micro resume "
                f"seed={incumbent_micro['seed']}"
            )
            return
        if streak >= PARENT_SEED_PROBE_TRIGGER:
            parent_micro_probe = discover_incumbent_parent_micro_probe(state, incumbent)
            if parent_micro_probe is not None:
                queue.append(parent_micro_probe)
                state["control_cold_streak"] = 0
                append_journal("cold_streak_parent_micro_probe_enqueued", streak=streak, candidate=parent_micro_probe)
                log(
                    "cold control streak="
                    f"{streak}; enqueue alt-seed micro from incumbent parent "
                    f"{parent_micro_probe['source_exp']} seed={parent_micro_probe['seed']}"
                )
                return
            parent_probe = discover_incumbent_parent_probe(state, incumbent)
            if parent_probe is not None:
                queue.append(parent_probe)
                state["control_cold_streak"] = 0
                append_journal("cold_streak_parent_probe_enqueued", streak=streak, candidate=parent_probe)
                log(
                    "cold control streak="
                    f"{streak}; enqueue alt-seed resume from incumbent parent "
                    f"{parent_probe['source_exp']} seed={parent_probe['seed']}"
                )
                return
            body_age_micro = discover_body_age_micro_probe(state, incumbent)
            if body_age_micro is not None:
                queue.append(body_age_micro)
                state["control_cold_streak"] = 0
                append_journal("cold_streak_body_age_micro_enqueued", streak=streak, candidate=body_age_micro)
                log(
                    "cold control streak="
                    f"{streak}; enqueue body-age micro resume "
                    f"seed={body_age_micro['seed']}"
                )
                return
        control_alt_micro = discover_best_control_alt_micro_candidate(
            state,
            exclude_exp=(incumbent.exp_name if incumbent is not None else None),
            min_mean_score=CONTROL_ALT_MICRO_SCORE,
        )
        if control_alt_micro is not None:
            queue.append(control_alt_micro)
            state["control_cold_streak"] = 0
            append_journal("cold_streak_control_alt_micro_enqueued", streak=streak, candidate=control_alt_micro)
            log(
                "cold control streak="
                f"{streak}; enqueue alt-seed micro from strong control "
                f"{control_alt_micro['source_exp']} seed={control_alt_micro['seed']}"
            )
            return
        control_alt_nano = discover_best_control_alt_nano_candidate(
            state,
            exclude_exp=(incumbent.exp_name if incumbent is not None else None),
            min_mean_score=CONTROL_ALT_NANO_SCORE,
        )
        if control_alt_nano is not None:
            queue.append(control_alt_nano)
            state["control_cold_streak"] = 0
            append_journal("cold_streak_control_alt_nano_enqueued", streak=streak, candidate=control_alt_nano)
            log(
                "cold control streak="
                f"{streak}; enqueue alt-seed nano from strong control "
                f"{control_alt_nano['source_exp']} seed={control_alt_nano['seed']}"
            )
            return
        endgame_basin_micro = discover_best_endgame_basin_micro_candidate(
            state,
            exclude_exp=(incumbent.exp_name if incumbent is not None else None),
            min_mean_score=ENDGAME_BASIN_MICRO_SCORE,
        )
        if endgame_basin_micro is not None:
            queue.append(endgame_basin_micro)
            state["control_cold_streak"] = 0
            append_journal("cold_streak_endgame_basin_micro_enqueued", streak=streak, candidate=endgame_basin_micro)
            log(
                "cold control streak="
                f"{streak}; enqueue alt-seed micro from historical endgame basin "
                f"{endgame_basin_micro['source_exp']} seed={endgame_basin_micro['seed']}"
            )
            return
        append_journal(
            "cold_streak_no_candidate",
            streak=streak,
            incumbent=(incumbent.exp_name if incumbent is not None else None),
            min_mean_score=min_mean_score,
        )
        log(
            f"cold control streak={streak}; no qualifying resume candidate above {min_mean_score:.2f}"
        )
        return
    if incumbent is not None and float(incumbent.mean_score) >= 390.0:
        upstream = parent_resume_candidate_for_exp(str(candidate["source_exp"]))
        if upstream is not None and not recent_alt_seed_resume_source_is_cold(str(upstream["source_exp"])):
            candidate = allocate_resume_seed_probe(
                state,
                str(upstream["source_exp"]),
                str(upstream["resume_state"]),
                "cold_control_revisit_best_resume_alt_seed",
            )
        else:
            if upstream is not None:
                append_journal(
                    "cold_source_upstream_suppressed",
                    source_exp=upstream.get("source_exp"),
                    basis_exp=candidate.get("source_exp"),
                )
                log(
                    f"cold control streak={streak}; suppress cold upstream resume source "
                    f"{upstream.get('source_exp')} for basis {candidate.get('source_exp')}"
                )
            candidate = allocate_resume_seed_probe(
                state,
                str(candidate["source_exp"]),
                str(candidate["resume_state"]),
                "cold_control_revisit_best_resume_alt_seed",
            )
    queue.append(candidate)
    state["control_cold_streak"] = 0
    append_journal("cold_streak_resume_enqueued", streak=streak, candidate=candidate)
    log(
        f"cold control streak={streak}; enqueue resume revisit from {candidate['source_exp']}"
    )


def discover_incumbent_parent_probe(state: dict[str, Any], incumbent: EvalStats | None) -> dict[str, Any] | None:
    if incumbent is None or not JOURNAL_PATH.exists():
        return None
    source_exp: str | None = None
    with open(JOURNAL_PATH, "r", encoding="utf-8") as f:
        for raw in reversed(f.readlines()):
            try:
                record = json.loads(raw)
            except Exception:
                continue
            if record.get("event") != "incumbent_update":
                continue
            if record.get("exp_name") != incumbent.exp_name:
                continue
            spec = record.get("spec") or {}
            source_exp = spec.get("source_exp")
            if source_exp:
                break
    if not source_exp:
        return None
    if source_exp in state.get("cold_parent_suppressed_sources", {}):
        return None
    if source_exp in state.get("best_micro_short_suppressed_sources", {}):
        return None

    probe_counts = state.setdefault("parent_seed_probe_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= PARENT_SEED_PROBE_LIMIT:
        return None

    run_dir = find_run_dir_by_exp_name(source_exp)
    if run_dir is None:
        return None
    resume_path = run_dir / "best_eval_resume.pt"
    if not resume_path.exists():
        return None
    return allocate_parent_seed_probe(
        state,
        source_exp,
        str(resume_path),
        "cold_control_alt_seed_from_incumbent_parent",
    )


def discover_incumbent_parent_micro_probe(state: dict[str, Any], incumbent: EvalStats | None) -> dict[str, Any] | None:
    if incumbent is None or not JOURNAL_PATH.exists():
        return None
    source_exp: str | None = None
    with open(JOURNAL_PATH, "r", encoding="utf-8") as f:
        for raw in reversed(f.readlines()):
            try:
                record = json.loads(raw)
            except Exception:
                continue
            if record.get("event") != "incumbent_update":
                continue
            if record.get("exp_name") != incumbent.exp_name:
                continue
            spec = record.get("spec") or {}
            source_exp = spec.get("source_exp")
            if source_exp:
                break
    if not source_exp or "_micro_" not in source_exp:
        return None
    if source_exp in state.get("cold_parent_suppressed_sources", {}):
        return None
    if source_exp in state.get("best_micro_short_suppressed_sources", {}):
        return None

    run_dir = find_run_dir_by_exp_name(source_exp)
    if run_dir is None:
        return None
    seed = seed_from_exp_name(source_exp)
    if seed is None:
        return None
    stats = extract_best_eval(run_dir, source_exp, int(seed), "resume_micro")
    if stats is None or stats.phase_lt20_rate > 0.0 or stats.mean_score < 390.0:
        return None
    resume_path = run_dir / "best_eval_resume.pt"
    if not resume_path.exists():
        return None
    return allocate_parent_micro_probe(
        state,
        source_exp,
        str(resume_path),
        "cold_control_alt_seed_micro_from_incumbent_parent",
    )


def discover_incumbent_micro_probe(state: dict[str, Any], incumbent: EvalStats | None) -> dict[str, Any] | None:
    if incumbent is None:
        return None
    resume_path = incumbent_resume_path(incumbent)
    if resume_path is None or not resume_path.exists():
        return None
    return allocate_incumbent_micro_probe(
        state,
        incumbent.exp_name,
        str(resume_path),
        "cold_control_incumbent_alt_seed_micro",
    )


def discover_incumbent_nano_probe(state: dict[str, Any], incumbent: EvalStats | None) -> dict[str, Any] | None:
    if incumbent is None:
        return None
    if incumbent.mean_score < RESUME_NANO_TRIGGER_SCORE or incumbent.phase_lt20_rate > 0.0:
        return None
    resume_path = incumbent_resume_path(incumbent)
    if resume_path is None or not resume_path.exists():
        return None
    return allocate_incumbent_nano_probe(
        state,
        incumbent.exp_name,
        str(resume_path),
        "cold_control_incumbent_alt_seed_nano",
    )


def discover_body_age_micro_probe(state: dict[str, Any], incumbent: EvalStats | None) -> dict[str, Any] | None:
    if incumbent is None:
        return None
    resume_path = incumbent_resume_path(incumbent)
    if resume_path is None or not resume_path.exists():
        return None
    return allocate_body_age_micro_probe(
        state,
        incumbent.exp_name,
        str(resume_path),
        "cold_control_body_age_micro",
    )


def discover_history_short_probe(state: dict[str, Any], incumbent: EvalStats | None) -> dict[str, Any] | None:
    if incumbent is None:
        return None
    if incumbent.mean_score < 392.0 or incumbent.phase_lt20_rate > 0.0:
        return None
    resume_path = incumbent_resume_path(incumbent)
    if resume_path is None or not resume_path.exists():
        return None
    return allocate_history_short_probe(
        state,
        incumbent.exp_name,
        str(resume_path),
        "cold_control_history_short",
    )


def allocate_endgame_micro_probe(
    state: dict[str, Any],
    source_exp: str,
    resume_state: str,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("endgame_micro_probe_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= ENDGAME_MICRO_PROBE_LIMIT:
        return None
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    probe_counts[source_exp] = used + 1
    return {
        "family": "resume_endgame_micro",
        "seed": seed,
        "resume_state": resume_state,
        "source_exp": source_exp,
        "reason": reason,
    }


def allocate_endgame_safe_probe(
    state: dict[str, Any],
    source_exp: str,
    resume_state: str,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("endgame_safe_probe_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= ENDGAME_SAFE_PROBE_LIMIT:
        return None
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    probe_counts[source_exp] = used + 1
    return {
        "family": "resume_endgame_safe_micro",
        "seed": seed,
        "resume_state": resume_state,
        "source_exp": source_exp,
        "reason": reason,
    }


def allocate_endgame_body_age_probe(
    state: dict[str, Any],
    source_exp: str,
    resume_state: str,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("endgame_body_age_probe_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= ENDGAME_BODY_AGE_PROBE_LIMIT:
        return None
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    probe_counts[source_exp] = used + 1
    return {
        "family": "resume_endgame_body_age_micro",
        "seed": seed,
        "resume_state": resume_state,
        "source_exp": source_exp,
        "reason": reason,
    }


def allocate_endgame_combo_probe(
    state: dict[str, Any],
    source_exp: str,
    resume_state: str,
    reason: str,
) -> dict[str, Any] | None:
    probe_counts = state.setdefault("endgame_combo_probe_counts", {})
    used = int(probe_counts.get(source_exp, 0))
    if used >= ENDGAME_BODY_AGE_PROBE_LIMIT:
        return None
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    probe_counts[source_exp] = used + 1
    return {
        "family": "resume_endgame_combo_micro",
        "seed": seed,
        "resume_state": resume_state,
        "source_exp": source_exp,
        "reason": reason,
    }


def discover_endgame_micro_probe(state: dict[str, Any], incumbent: EvalStats | None) -> dict[str, Any] | None:
    if incumbent is None:
        return None
    if incumbent.exp_name not in state.get("failure_harvest_results", {}):
        return None
    if incumbent.win_rate > 0.0 or incumbent.phase_lt20_rate > 0.0:
        return None
    if incumbent.phase_gte95_rate < 1.0 or incumbent.mean_score < 391.0:
        return None
    resume_path = incumbent_resume_path(incumbent)
    if resume_path is None or not resume_path.exists():
        return None
    return allocate_endgame_micro_probe(
        state,
        incumbent.exp_name,
        str(resume_path),
        "near_perfect_endgame_micro",
    )


def discover_endgame_safe_probe(state: dict[str, Any], incumbent: EvalStats | None) -> dict[str, Any] | None:
    if incumbent is None:
        return None
    analysis = state.get("failure_harvest_results", {}).get(incumbent.exp_name)
    if analysis is None:
        return None
    if incumbent.win_rate > 0.0 or incumbent.phase_lt20_rate > 0.0:
        return None
    if incumbent.phase_gte95_rate < 1.0 or incumbent.mean_score < 391.0:
        return None
    if int_or(analysis.get("harvested_failures"), 0) < 20:
        return None
    if str(analysis.get("dominant_reason")) != "self":
        return None
    mismatch = analysis.get("last_action_safe_mismatch_rate")
    top_signature = analysis.get("top_signature") or {}
    signature_text = top_signature.get("signature") if isinstance(top_signature, dict) else str(top_signature)
    safe_signal = (
        mismatch is not None and float(mismatch) >= 0.5
    ) or ("safe=miss->" in signature_text)
    if not safe_signal:
        return None
    resume_path = incumbent_resume_path(incumbent)
    if resume_path is None or not resume_path.exists():
        return None
    return allocate_endgame_safe_probe(
        state,
        incumbent.exp_name,
        str(resume_path),
        "near_perfect_endgame_safe_micro",
    )


def discover_endgame_body_age_probe(state: dict[str, Any], incumbent: EvalStats | None) -> dict[str, Any] | None:
    if incumbent is None:
        return None
    analysis = state.get("failure_harvest_results", {}).get(incumbent.exp_name)
    if analysis is None:
        return None
    if incumbent.win_rate > 0.0 or incumbent.phase_lt20_rate > 0.0:
        return None
    if incumbent.phase_gte95_rate < 1.0 or incumbent.mean_score < 391.0:
        return None
    if int_or(analysis.get("harvested_failures"), 0) < 20:
        return None
    top_signature = analysis.get("top_signature") or {}
    signature_text = top_signature.get("signature") if isinstance(top_signature, dict) else str(top_signature)
    if "safe=na" not in signature_text:
        return None
    resume_path = incumbent_resume_path(incumbent)
    if resume_path is None or not resume_path.exists():
        return None
    return allocate_endgame_body_age_probe(
        state,
        incumbent.exp_name,
        str(resume_path),
        "near_perfect_endgame_body_age_micro",
    )


def discover_endgame_combo_probe(state: dict[str, Any], incumbent: EvalStats | None) -> dict[str, Any] | None:
    if incumbent is None:
        return None
    analysis = state.get("failure_harvest_results", {}).get(incumbent.exp_name)
    if analysis is None:
        return None
    if incumbent.win_rate > 0.0 or incumbent.phase_lt20_rate > 0.0:
        return None
    if incumbent.phase_gte95_rate < 1.0 or incumbent.mean_score < 391.0:
        return None
    if int_or(analysis.get("harvested_failures"), 0) < 20:
        return None
    if str(analysis.get("dominant_reason")) != "self":
        return None
    resume_path = incumbent_resume_path(incumbent)
    if resume_path is None or not resume_path.exists():
        return None
    return allocate_endgame_combo_probe(
        state,
        incumbent.exp_name,
        str(resume_path),
        "near_perfect_endgame_combo_micro",
    )


def finalize_run(
    state: dict[str, Any],
    incumbent: EvalStats | None,
    spec: dict[str, Any],
    exp_name: str,
    run_dir: Path,
    exit_code: int,
) -> EvalStats | None:
    seed = int(spec["seed"])
    source = spec["family"]
    cleanup = cleanup_run(run_dir)
    state["active_run"] = None

    if spec["family"] == "failure_harvest":
        analysis = load_json(run_dir / "harvest_summary.json")
        state["last_run"] = {
            "exp_name": exp_name,
            "seed": seed,
            "run_dir": str(run_dir),
            "exit_code": exit_code,
            "cleanup": cleanup,
            "spec": spec,
            "analysis": analysis,
        }
        if analysis is None:
            log(f"missing failure harvest summary for {exp_name}")
            append_journal("missing_failure_harvest", exp_name=exp_name, seed=seed, run_dir=str(run_dir), spec=spec)
            atomic_write_json(STATE_PATH, state)
            return incumbent
        source_exp = str(spec.get("source_exp", exp_name))
        state.setdefault("failure_harvest_results", {})[source_exp] = {
            "run_dir": str(run_dir),
            "harvested_failures": int_or(analysis.get("harvested_failures"), 0),
            "dominant_reason": analysis.get("dominant_reason"),
            "last_action_safe_mismatch_rate": analysis.get("last_action_safe_mismatch_rate"),
            "top_signature": (analysis.get("top_signatures") or [None])[0],
        }
        append_journal(
            "failure_harvest_complete",
            exp_name=exp_name,
            seed=seed,
            run_dir=str(run_dir),
            spec=spec,
            harvested_failures=int_or(analysis.get("harvested_failures"), 0),
            dominant_reason=analysis.get("dominant_reason"),
            last_action_safe_mismatch_rate=analysis.get("last_action_safe_mismatch_rate"),
            top_signature=(analysis.get("top_signatures") or [None])[0],
        )
        log(
            "failure_harvest "
            f"source={source_exp} harvested={int_or(analysis.get('harvested_failures'), 0)} "
            f"dominant={analysis.get('dominant_reason')} "
            f"safe_miss={num_or(analysis.get('last_action_safe_mismatch_rate'), 0.0):.2%}"
        )
        safe_probe = discover_endgame_safe_probe(state, incumbent)
        if safe_probe is not None:
            state.setdefault("queue", []).insert(0, safe_probe)
            append_journal(
                "failure_harvest_safe_probe_enqueued",
                source_exp=source_exp,
                candidate=safe_probe,
                harvested_failures=int_or(analysis.get("harvested_failures"), 0),
                last_action_safe_mismatch_rate=analysis.get("last_action_safe_mismatch_rate"),
            )
            log(
                f"failure_harvest source={source_exp} enqueued safe endgame micro "
                f"seed={safe_probe['seed']}"
            )
        atomic_write_json(STATE_PATH, state)
        return incumbent

    if spec["family"] == "win_probe":
        summary = load_json(run_dir / "win_probe_summary.json")
        state["last_run"] = {
            "exp_name": exp_name,
            "seed": seed,
            "run_dir": str(run_dir),
            "exit_code": exit_code,
            "cleanup": cleanup,
            "spec": spec,
            "summary": summary,
        }
        if summary is None:
            log(f"missing win probe summary for {exp_name}")
            append_journal("missing_win_probe", exp_name=exp_name, seed=seed, run_dir=str(run_dir), spec=spec)
            atomic_write_json(STATE_PATH, state)
            return incumbent
        source_exp = str(spec.get("source_exp", exp_name))
        state.setdefault("win_probe_results", {})[source_exp] = {
            "run_dir": str(run_dir),
            "episodes": int_or(summary.get("episodes"), 0),
            "wins": int_or(summary.get("wins"), 0),
            "win_rate": num_or(summary.get("win_rate"), 0.0),
            "mean_score": num_or(summary.get("mean_score"), 0.0),
        }
        append_journal(
            "win_probe_complete",
            exp_name=exp_name,
            seed=seed,
            run_dir=str(run_dir),
            spec=spec,
            episodes=int_or(summary.get("episodes"), 0),
            wins=int_or(summary.get("wins"), 0),
            win_rate=num_or(summary.get("win_rate"), 0.0),
            mean_score=num_or(summary.get("mean_score"), 0.0),
        )
        log(
            "win_probe "
            f"source={source_exp} episodes={int_or(summary.get('episodes'), 0)} "
            f"wins={int_or(summary.get('wins'), 0)} "
            f"win_rate={num_or(summary.get('win_rate'), 0.0):.2%} "
            f"mean={num_or(summary.get('mean_score'), 0.0):.2f}"
        )
        atomic_write_json(STATE_PATH, state)
        return incumbent

    stats = extract_best_eval(run_dir, exp_name, seed, source)
    state["last_run"] = {
        "exp_name": exp_name,
        "seed": seed,
        "run_dir": str(run_dir),
        "exit_code": exit_code,
        "cleanup": cleanup,
        "spec": spec,
        "best_eval": asdict(stats) if stats else None,
    }

    if stats is None:
        log(f"no eval for {exp_name}")
        append_journal("missing_eval", exp_name=exp_name, seed=seed, run_dir=str(run_dir), spec=spec)
        atomic_write_json(STATE_PATH, state)
        return incumbent

    append_journal("eval", spec=spec, **asdict(stats))
    log(
        f"{spec['family']} seed={seed} mean={stats.mean_score:.2f} median={stats.median_score:.2f} "
        f"lt20={stats.phase_lt20_rate:.2%} 95+={stats.phase_gte95_rate:.2%} win={stats.win_rate:.2%}"
    )

    if spec["family"] == "control":
        if stats.mean_score < CONTROL_COLD_SCORE:
            state["control_cold_streak"] = int(state.get("control_cold_streak", 0)) + 1
        else:
            state["control_cold_streak"] = 0
    elif spec["family"] != "resume_short":
        state["control_cold_streak"] = 0

    if should_update_incumbent(incumbent, stats):
        incumbent = stats
        state["incumbent"] = asdict(stats)
        atomic_write_json(CURRENT_BEST_PATH, asdict(stats))
        append_journal("incumbent_update", spec=spec, **asdict(stats))
        log(f"incumbent update: {stats.exp_name} mean={stats.mean_score:.2f}")
        maybe_enqueue_incumbent_win_probe(state, incumbent)
        maybe_enqueue_incumbent_failure_harvest(state, incumbent)

    maybe_enqueue_followups(state, spec, stats, run_dir)
    maybe_enqueue_cold_streak_research(state, incumbent)
    atomic_write_json(STATE_PATH, state)
    return incumbent


def recover_active_run(state: dict[str, Any], incumbent: EvalStats | None) -> EvalStats | None:
    active = state.get("active_run")
    if not active:
        return incumbent
    pid = active.get("pid")
    exp_name = active.get("exp_name")
    run_dir_str = active.get("run_dir")
    spec = active.get("spec") or {"family": "unknown", "seed": seed_from_exp_name(exp_name or "") or 0}
    if not exp_name or not run_dir_str:
        state["active_run"] = None
        atomic_write_json(STATE_PATH, state)
        return incumbent
    run_dir = Path(run_dir_str)
    driver_log_value = active.get("driver_log")
    driver_log = Path(driver_log_value) if driver_log_value else None
    log(f"recovering active run {exp_name}")
    append_journal("recover_active_run", exp_name=exp_name, spec=spec, pid=pid, run_dir=str(run_dir))
    if pid_alive(pid):
        wait_for_process_with_stall_guard(
            pid=int(pid),
            exp_name=exp_name,
            run_dir=run_dir,
            driver_log=driver_log,
            proc=None,
        )
    return finalize_run(state, incumbent, spec, exp_name, run_dir, 0)


def cleanup_sleep_processes() -> None:
    try:
        out = subprocess.check_output(
            ["zsh", "-lc", "ps -ef | rg 'sleep 7200' | rg -v 'rg ' || true"],
            text=True,
        )
    except Exception:
        return
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            os.kill(int(parts[1]), signal.SIGTERM)
        except Exception:
            pass


def main() -> int:
    if not PYTHON.exists():
        raise SystemExit(f"missing interpreter: {PYTHON}")
    if not BASE_RESUME.exists():
        raise SystemExit(f"missing base resume checkpoint: {BASE_RESUME}")

    cleanup_sleep_processes()

    state = bootstrap_state()
    incumbent = EvalStats(**state["incumbent"]) if state.get("incumbent") else None
    if incumbent:
        log(
            f"bootstrap incumbent={incumbent.exp_name} mean={incumbent.mean_score:.2f} "
            f"lt20={incumbent.phase_lt20_rate:.2%} 95+={incumbent.phase_gte95_rate:.2%}"
        )
    else:
        log("bootstrap no incumbent found")

    incumbent = recover_active_run(state, incumbent)

    while True:
        if STOP_PATH.exists():
            log(f"stop file present: {STOP_PATH}")
            append_journal("stop_file", path=str(STOP_PATH))
            return 0

        free_gb = free_disk_gb()
        if free_gb < DISK_FLOOR_GB:
            log(f"disk floor reached: free_gb={free_gb:.2f} threshold={DISK_FLOOR_GB:.2f}")
            append_journal("disk_floor", free_gb=free_gb, threshold_gb=DISK_FLOOR_GB)
            return 0

        maybe_enqueue_incumbent_win_probe(state, incumbent)
        maybe_enqueue_incumbent_failure_harvest(state, incumbent)
        maybe_enqueue_cold_streak_research(state, incumbent)
        atomic_write_json(STATE_PATH, state)

        spec = build_spec(state)
        state["loop_index"] = int(state.get("loop_index", 0)) + 1
        exp_name = exp_name_for_spec(spec, int(state["loop_index"]))
        args = args_for_spec(spec)

        run_dir, exit_code = run_experiment(exp_name, spec, args, state)
        if run_dir is None:
            log(f"missing run dir for {exp_name}")
            append_journal("missing_run_dir", exp_name=exp_name, spec=spec)
            state["active_run"] = None
            atomic_write_json(STATE_PATH, state)
            continue

        incumbent = finalize_run(state, incumbent, spec, exp_name, run_dir, exit_code)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("keyboard interrupt")
        raise
