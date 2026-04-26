#!/usr/bin/env python3
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional


ROOT = "/Users/elichen/code/perfect_snake"
PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
TRAIN = os.path.join(ROOT, "train.py")
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")
BASE_RESUME = os.path.join(
    ROOT,
    "experiments",
    "exp074_multi_path_curriculum_ft_177317600936",
    "best_eval.pt",
)
STATE_PATH = os.path.join(EXPERIMENTS_DIR, "ppo_seed_loop_state.json")
JOURNAL_PATH = os.path.join(EXPERIMENTS_DIR, "ppo_seed_loop_journal.jsonl")
STOP_PATH = os.path.join(EXPERIMENTS_DIR, "ppo_seed_loop.stop")
LOG_PATH = os.path.join(EXPERIMENTS_DIR, "ppo_seed_loop.log")

FRESH_KEEP_THRESHOLD = 335.0
CONTINUATION_ATTEMPT_THRESHOLD = 355.0
CONTINUATION_KEEP_DELTA = -5.0
CONTINUATION_KILL_DELTA = -20.0


@dataclass
class EvalStats:
    mean_score: float
    win_rate: float
    phase_lt20_rate: float
    phase_gte95_rate: float
    run_dir: str
    exp_name: str
    seed: int
    agent_steps: int
    source: str


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def append_journal(event: dict) -> None:
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
    with open(JOURNAL_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def save_state(state: dict) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_PATH)


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def parse_seed(exp_name: str) -> Optional[int]:
    match = re.search(r"_s(\d+)", exp_name)
    if not match:
        return None
    return int(match.group(1))


def list_matching_runs():
    runs = []
    for run_json in glob.glob(os.path.join(EXPERIMENTS_DIR, "*", "run.json")):
        try:
            with open(run_json) as f:
                payload = json.load(f)
        except Exception:
            continue
        args = payload.get("args", {})
        exp_name = args.get("exp_name")
        if not exp_name:
            continue
        seed = args.get("seed")
        if seed is None:
            seed = parse_seed(exp_name)
        run_dir = os.path.dirname(run_json)
        runs.append((exp_name, int(seed) if seed is not None else None, args, run_dir))
    return runs


def extract_best_eval(run_dir: str, exp_name: str, seed: int, source: str) -> Optional[EvalStats]:
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return None
    best = None
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("type") != "eval":
                continue
            if best is None or row.get("mean_score", -1.0) > best.get("mean_score", -1.0):
                best = row
    if best is None:
        return None
    return EvalStats(
        mean_score=float(best.get("mean_score", 0.0)),
        win_rate=float(best.get("win_rate", 0.0)),
        phase_lt20_rate=float(best.get("phase_lt20_rate", 1.0)),
        phase_gte95_rate=float(best.get("phase_gte95_rate", 0.0)),
        run_dir=run_dir,
        exp_name=exp_name,
        seed=seed,
        agent_steps=int(best.get("agent_steps", 0)),
        source=source,
    )


def bootstrap_state() -> dict:
    state = load_state()
    if state.get("initialized"):
        return state

    best_fresh = None
    best_continuation = None
    max_seed = 42

    for exp_name, seed, args, run_dir in list_matching_runs():
        if seed is not None:
            max_seed = max(max_seed, seed)
        if args.get("resume") == os.path.relpath(BASE_RESUME, ROOT) or args.get("resume") == BASE_RESUME:
            stats = extract_best_eval(run_dir, exp_name, seed or -1, "fresh")
            if stats and (best_fresh is None or stats.mean_score > best_fresh.mean_score):
                best_fresh = stats
        if args.get("resume_state"):
            stats = extract_best_eval(run_dir, exp_name, seed or -1, "continuation")
            if stats and (best_continuation is None or stats.mean_score > best_continuation.mean_score):
                best_continuation = stats

    state = {
        "initialized": True,
        "next_seed": max_seed + 1,
        "best_fresh": asdict(best_fresh) if best_fresh else None,
        "best_continuation": asdict(best_continuation) if best_continuation else None,
        "last_run": None,
    }
    save_state(state)
    return state


def run_experiment(exp_name: str, args: list[str]) -> str:
    log(f"launch {exp_name}")
    log_path = os.path.join(EXPERIMENTS_DIR, f"{exp_name}.controller.log")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            [PYTHON, TRAIN, *args, "--exp-name", exp_name],
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log(f"exit {exp_name} code={proc.returncode}")
    runs = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, f"{exp_name}_*")))
    if not runs:
        raise RuntimeError(f"missing run dir for {exp_name}")
    return runs[-1]


def fresh_args(seed: int) -> list[str]:
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
        "--resume", BASE_RESUME,
        "--seed", str(seed),
    ]


def continuation_args(run_dir: str, seed: int) -> list[str]:
    return [
        "--board-size", "20",
        "--timesteps", "1000000",
        "--resume-add-steps",
        "--override-resume-lr",
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
        "--lr", "1e-6",
        "--no-anneal-lr",
        "--eval-every-steps", "250000",
        "--eval-deterministic",
        "--eval-episodes", "20",
        "--resume-state", os.path.join(run_dir, "best_eval_resume.pt"),
        "--seed", str(seed),
    ]


def maybe_update_best(state: dict, key: str, stats: EvalStats) -> bool:
    current = state.get(key)
    if current is None or stats.mean_score > float(current["mean_score"]):
        state[key] = asdict(stats)
        save_state(state)
        return True
    return False


def main() -> int:
    if not os.path.exists(PYTHON):
        raise SystemExit(f"missing interpreter: {PYTHON}")
    state = bootstrap_state()
    log(f"bootstrap next_seed={state['next_seed']} best_fresh={state.get('best_fresh')}")

    while True:
        if os.path.exists(STOP_PATH):
            log(f"stop file present: {STOP_PATH}")
            return 0

        seed = int(state["next_seed"])
        exp_name = f"ppo_loop_screen_s{seed}"
        run_dir = run_experiment(exp_name, fresh_args(seed))
        stats = extract_best_eval(run_dir, exp_name, seed, "fresh")
        if stats is None:
            append_journal({"event": "fresh_missing_eval", "seed": seed, "run_dir": run_dir})
            state["next_seed"] = seed + 1
            save_state(state)
            continue

        append_journal({"event": "fresh_eval", **asdict(stats)})
        state["last_run"] = {"exp_name": exp_name, "run_dir": run_dir, "type": "fresh"}
        fresh_improved = maybe_update_best(state, "best_fresh", stats)
        log(
            f"fresh seed={seed} mean={stats.mean_score:.2f} "
            f"lt20={stats.phase_lt20_rate:.2%} 95+={stats.phase_gte95_rate:.2%}"
        )

        if stats.win_rate > 0:
            append_journal({"event": "fresh_win", **asdict(stats)})
            log("deterministic win found; leaving loop running but pausing on this discovery")
            return 0

        if stats.mean_score >= CONTINUATION_ATTEMPT_THRESHOLD and stats.phase_lt20_rate <= 0.0:
            cont_name = f"ppo_loop_resume_s{seed}_lr1e6"
            cont_run_dir = run_experiment(cont_name, continuation_args(run_dir, seed))
            cont_stats = extract_best_eval(cont_run_dir, cont_name, seed, "continuation")
            if cont_stats is not None:
                append_journal({"event": "continuation_eval", **asdict(cont_stats)})
                maybe_update_best(state, "best_continuation", cont_stats)
                delta = cont_stats.mean_score - stats.mean_score
                log(
                    f"continuation seed={seed} mean={cont_stats.mean_score:.2f} "
                    f"delta={delta:+.2f}"
                )
                if cont_stats.win_rate > 0:
                    append_journal({"event": "continuation_win", **asdict(cont_stats)})
                    return 0
                if delta <= CONTINUATION_KILL_DELTA:
                    append_journal({"event": "continuation_regressed_hard", "seed": seed, "delta": delta})
                elif delta >= CONTINUATION_KEEP_DELTA:
                    append_journal({"event": "continuation_promising", "seed": seed, "delta": delta})

        state["next_seed"] = seed + 1
        save_state(state)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("keyboard interrupt")
        raise
