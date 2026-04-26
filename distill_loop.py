"""Autonomous loop for standalone pure-NN distillation experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from distill.evaluate import evaluate_checkpoint
from distill.evaluate_rnn import evaluate_checkpoint as evaluate_rnn_checkpoint


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
EXPERIMENTS_DIR = ROOT / "experiments"
STATE_PATH = EXPERIMENTS_DIR / "distill_loop_state.json"
JOURNAL_PATH = EXPERIMENTS_DIR / "distill_loop_journal.jsonl"
LOG_PATH = EXPERIMENTS_DIR / "distill_loop.log"

BASE_ARGS = {
    "board_size": 20,
    "device": "cpu",
    "network_scale": 2,
    "flood_fill": True,
    "aux_flood_fill": True,
    "aux_flood_fill_coef": 1.0,
    "head_centered": True,
    "eval_episodes": 5,
}


def _selector(stats: dict) -> tuple[float, float, float, float]:
    return (
        float(stats["win_rate"]),
        float(stats["mean_score"]),
        float(stats["phase_gte95_rate"]),
        -float(stats["phase_lt20_rate"]),
    )


def _handoff_ready(stats: dict) -> bool:
    return (
        float(stats["mean_score"]) >= 380.0
        and float(stats["phase_gte95_rate"]) >= 0.80
        and float(stats["phase_lt20_rate"]) <= 0.05
    )


def _write_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_name(f"{STATE_PATH.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_PATH)


def _append_journal(event: dict) -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def _load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "iteration": 0,
        "best": None,
        "last_long_resume": None,
        "active_run": None,
        "status": "bootstrapping",
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _build_command(spec: dict, save_path: Path) -> list[str]:
    if spec["family"].startswith("cyclecond_"):
        cmd = [
            str(PYTHON),
            "-u",
            "distill_train_bc.py",
            "--board-size",
            str(BASE_ARGS["board_size"]),
            "--steps",
            str(spec["steps"]),
            "--batch-size",
            str(spec["batch_size"]),
            "--lr",
            str(spec["lr"]),
            "--device",
            BASE_ARGS["device"],
            "--network-scale",
            str(BASE_ARGS["network_scale"]),
            "--flood-fill",
            "--head-centered",
            "--cycle-conditioning",
            "--min-fill",
            str(spec["min_fill"]),
            "--max-fill",
            str(spec["max_fill"]),
            "--seed",
            str(spec["seed"]),
            "--save-path",
            str(save_path),
            "--log-every",
            str(spec["log_every"]),
            "--eval-episodes",
            str(spec.get("eval_episodes", BASE_ARGS["eval_episodes"])),
        ]
        return cmd

    if spec["family"].startswith("rnn_"):
        cmd = [
            str(PYTHON),
            "-u",
            "distill_train_bc_rnn.py",
            "--board-size",
            str(BASE_ARGS["board_size"]),
            "--steps",
            str(spec["steps"]),
            "--num-envs",
            str(spec["num_envs"]),
            "--horizon",
            str(spec["horizon"]),
            "--lr",
            str(spec["lr"]),
            "--device",
            BASE_ARGS["device"],
            "--hidden-size",
            str(spec.get("hidden_size", 256)),
            "--batch-size",
            str(spec.get("batch_size", 16)),
            "--flood-fill",
            "--head-centered",
            "--min-fill",
            str(spec["min_fill"]),
            "--max-fill",
            str(spec["max_fill"]),
            "--seed",
            str(spec["seed"]),
            "--save-path",
            str(save_path),
            "--log-every",
            str(spec["log_every"]),
            "--eval-episodes",
            str(spec.get("eval_episodes", BASE_ARGS["eval_episodes"])),
        ]
        if spec.get("cycle_conditioning"):
            cmd.append("--cycle-conditioning")
        if spec.get("prev_action_input"):
            cmd.append("--prev-action-input")
        if spec.get("fill_input"):
            cmd.append("--fill-input")
        if spec.get("save_best_eval", False):
            cmd.append("--save-best-eval")
        if "seq_len" in spec:
            cmd.extend(["--seq-len", str(spec["seq_len"])])
        if "burn_in" in spec:
            cmd.extend(["--burn-in", str(spec["burn_in"])])
        if "offline_episodes" in spec:
            cmd.extend(["--offline-episodes", str(spec["offline_episodes"])])
        if "student_episodes" in spec:
            cmd.extend(["--student-episodes", str(spec["student_episodes"])])
        if "recovery_episodes" in spec:
            cmd.extend(["--recovery-episodes", str(spec["recovery_episodes"])])
        if "round_steps" in spec:
            cmd.extend(["--round-steps", str(spec["round_steps"])])
        if "dagger_rounds" in spec:
            cmd.extend(["--dagger-rounds", str(spec["dagger_rounds"])])
        if "rollin_beta_schedule" in spec:
            cmd.extend(["--rollin-beta-schedule", str(spec["rollin_beta_schedule"])])
        if "student_mix_ratio" in spec:
            cmd.extend(["--student-mix-ratio", str(spec["student_mix_ratio"])])
        if "predeath_mix_ratio" in spec:
            cmd.extend(["--predeath-mix-ratio", str(spec["predeath_mix_ratio"])])
        if "predeath_window" in spec:
            cmd.extend(["--predeath-window", str(spec["predeath_window"])])
        if "perturb_prob" in spec:
            cmd.extend(["--perturb-prob", str(spec["perturb_prob"])])
        if "perturb_min_fill" in spec:
            cmd.extend(["--perturb-min-fill", str(spec["perturb_min_fill"])])
        if "perturb_max_fill" in spec:
            cmd.extend(["--perturb-max-fill", str(spec["perturb_max_fill"])])
        if "perturb_min_steps" in spec:
            cmd.extend(["--perturb-min-steps", str(spec["perturb_min_steps"])])
        if "perturb_max_steps" in spec:
            cmd.extend(["--perturb-max-steps", str(spec["perturb_max_steps"])])
        if "safe_target_coef" in spec:
            cmd.extend(["--safe-target-coef", str(spec["safe_target_coef"])])
        if "safe_target_min_fill" in spec:
            cmd.extend(["--safe-target-min-fill", str(spec["safe_target_min_fill"])])
        if "safe_target_max_fill" in spec:
            cmd.extend(["--safe-target-max-fill", str(spec["safe_target_max_fill"])])
        if "safe_target_temperature" in spec:
            cmd.extend(["--safe-target-temperature", str(spec["safe_target_temperature"])])
        if "future_action_horizon" in spec:
            cmd.extend(["--future-action-horizon", str(spec["future_action_horizon"])])
        if "future_action_coef" in spec:
            cmd.extend(["--future-action-coef", str(spec["future_action_coef"])])
        if "expert_target_min_fill" in spec:
            cmd.extend(["--expert-target-min-fill", str(spec["expert_target_min_fill"])])
        if "expert_target_max_fill" in spec:
            cmd.extend(["--expert-target-max-fill", str(spec["expert_target_max_fill"])])
        if "early_head_max_fill" in spec:
            cmd.extend(["--early-head-max-fill", str(spec["early_head_max_fill"])])
        if "short_eval_episodes" in spec:
            cmd.extend(["--short-eval-episodes", str(spec["short_eval_episodes"])])
        resume = spec.get("resume")
        if resume:
            cmd.extend(["--resume", str(resume)])
        return cmd

    cmd = [
        str(PYTHON),
        "-u",
        "distill_train_bc.py",
        "--board-size",
        str(BASE_ARGS["board_size"]),
        "--steps",
        str(spec["steps"]),
        "--batch-size",
        str(spec["batch_size"]),
        "--lr",
        str(spec["lr"]),
        "--device",
        BASE_ARGS["device"],
        "--network-scale",
        str(BASE_ARGS["network_scale"]),
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef",
        str(BASE_ARGS["aux_flood_fill_coef"]),
        "--head-centered",
        "--min-fill",
        str(spec["min_fill"]),
        "--max-fill",
        str(spec["max_fill"]),
        "--seed",
        str(spec["seed"]),
        "--save-path",
        str(save_path),
        "--log-every",
        str(spec["log_every"]),
        "--eval-episodes",
        str(spec.get("eval_episodes", BASE_ARGS["eval_episodes"])),
    ]
    resume = spec.get("resume")
    if resume:
        cmd.extend(["--resume", str(resume)])
    late_head_min_fill = spec.get("late_head_min_fill")
    if late_head_min_fill is not None:
        cmd.extend(["--late-head-min-fill", str(late_head_min_fill)])
    if spec.get("train_late_head_only"):
        cmd.append("--train-late-head-only")
    return cmd


def _evaluate(checkpoint_path: Path, spec: dict | None = None) -> dict:
    if "cyclecond_" in checkpoint_path.name or "cyclecond_" in str(checkpoint_path.parent):
        return evaluate_checkpoint(
            checkpoint_path=str(checkpoint_path),
            board_size=BASE_ARGS["board_size"],
            episodes=20,
            seed=12345,
            deterministic=True,
            device=BASE_ARGS["device"],
            network_scale=BASE_ARGS["network_scale"],
            flood_fill=BASE_ARGS["flood_fill"],
            aux_flood_fill=False,
            head_centered=BASE_ARGS["head_centered"],
            late_head_min_fill=None,
            cycle_conditioning=True,
        )
    if "rnn_" in checkpoint_path.name or "rnn_" in str(checkpoint_path.parent):
        return evaluate_rnn_checkpoint(
            checkpoint_path=str(checkpoint_path),
            board_size=BASE_ARGS["board_size"],
            episodes=20,
            seed=12345,
            deterministic=True,
            device=BASE_ARGS["device"],
            flood_fill=BASE_ARGS["flood_fill"],
            head_centered=BASE_ARGS["head_centered"],
            hidden_size=256 if spec is None else int(spec.get("hidden_size", 256)),
            cycle_conditioning=False if spec is None else bool(spec.get("cycle_conditioning", False)),
            use_prev_action_input=False if spec is None else bool(spec.get("prev_action_input", False)),
            use_fill_input=False if spec is None else bool(spec.get("fill_input", False)),
            future_action_horizon=0 if spec is None else int(spec.get("future_action_horizon", 0)),
            early_head_max_fill=None if spec is None else spec.get("early_head_max_fill"),
        )
    return evaluate_checkpoint(
        checkpoint_path=str(checkpoint_path),
        board_size=BASE_ARGS["board_size"],
        episodes=20,
        seed=12345,
        deterministic=True,
        device=BASE_ARGS["device"],
        network_scale=BASE_ARGS["network_scale"],
        flood_fill=BASE_ARGS["flood_fill"],
        aux_flood_fill=BASE_ARGS["aux_flood_fill"],
        head_centered=BASE_ARGS["head_centered"],
        late_head_min_fill=None,
    )


def _next_spec(state: dict) -> dict:
    best = state.get("best")
    iteration = int(state.get("iteration", 0))

    # Feedforward expert cloning has collapsed repeatedly to near-zero score.
    # Pivot to recurrent distillation once that failure mode is established.
    if best is not None and float(best["stats"]["mean_score"]) < 5.0 and iteration >= 2:
        if float(best["stats"]["mean_score"]) < 1.0 and iteration >= 8:
            return {
                "name": "distill201_cyclecond_ham_bc_full",
                "family": "cyclecond_full",
                "seed": 44 + iteration,
                "steps": 300,
                "batch_size": 64,
                "log_every": 10,
                "eval_episodes": 5,
                "lr": 1e-4,
                "min_fill": 0.0,
                "max_fill": 1.0,
            }
        if (
            best.get("family") == "rnn_full"
            and float(best["stats"]["mean_score"]) >= 300.0
            and float(best["stats"]["phase_lt20_rate"]) <= 0.05
            and state.get("last_long_resume") != best["checkpoint"]
        ):
            return {
                "name": "distill102_rnn_ham_bc_full_long",
                "family": "rnn_full_long",
                "seed": int(best["seed"]),
                "steps": 1500,
                "num_envs": 8,
                "horizon": 64,
                "log_every": 25,
                "eval_episodes": 5,
                "hidden_size": 256,
                "lr": 1e-4,
                "min_fill": 0.0,
                "max_fill": 1.0,
                "resume": best["checkpoint"],
            }

        return {
            "name": "distill101_rnn_ham_bc_full",
            "family": "rnn_full",
            "seed": 44 + iteration,
            "steps": 500,
            "num_envs": 8,
            "horizon": 64,
            "log_every": 25,
            "eval_episodes": 5,
            "hidden_size": 256,
            "lr": 1e-4,
            "min_fill": 0.0,
            "max_fill": 1.0,
        }

    if (
        best is not None
        and best.get("family") == "full"
        and float(best["stats"]["mean_score"]) >= 300.0
        and float(best["stats"]["phase_lt20_rate"]) <= 0.05
        and state.get("last_long_resume") != best["checkpoint"]
    ):
        return {
            "name": "distill002_scratch_ham_bc_full_long",
            "family": "full_long",
            "seed": int(best["seed"]),
            "steps": 3000,
            "batch_size": 256,
            "log_every": 50,
            "lr": 1e-4,
            "min_fill": 0.0,
            "max_fill": 1.0,
            "resume": best["checkpoint"],
        }

    seed = 44 + iteration
    if iteration % 2 == 0:
        return {
            "name": "distill001_scratch_ham_bc_full",
            "family": "full",
            "seed": seed,
            "steps": 1000,
            "batch_size": 256,
            "log_every": 50,
            "lr": 1e-4,
            "min_fill": 0.0,
            "max_fill": 1.0,
        }
    return {
        "name": "distill003_scratch_ham_bc_lateonly",
        "family": "lateonly",
        "seed": seed,
        "steps": 300,
        "batch_size": 64,
        "log_every": 10,
        "lr": 1e-4,
        "min_fill": 0.90,
        "max_fill": 1.0,
    }


def _run_trial(state: dict, spec: dict) -> dict:
    run_id = str(int(time.time() * 1000))
    run_name = f"{spec['name']}_s{spec['seed']}_{run_id}"
    run_dir = EXPERIMENTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / "model.pt"
    log_path = run_dir / "run.log"
    cmd = _build_command(spec, save_path)

    state["active_run"] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "spec": spec,
        "command": cmd,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    state["status"] = "running"
    _write_state(state)

    with open(LOG_PATH, "a", encoding="utf-8") as master_log, open(log_path, "w", encoding="utf-8") as run_log:
        header = f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {run_name}\n$ {' '.join(shlex.quote(part) for part in cmd)}\n"
        master_log.write(header)
        master_log.flush()
        run_log.write(header)
        run_log.flush()

        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            master_log.write(line)
            run_log.write(line)
        exit_code = process.wait()

    best_eval_path = run_dir / "model.best_eval.pt"
    checkpoint_path = best_eval_path if best_eval_path.exists() else save_path

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "save_path": str(save_path),
        "best_eval_path": str(best_eval_path),
        "seed": int(spec["seed"]),
        "family": spec["family"],
        "command": cmd,
        "exit_code": int(exit_code),
        "spec": spec,
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if exit_code == 0 and checkpoint_path.exists():
        stats = _evaluate(checkpoint_path, spec)
        summary["stats"] = stats
        summary["selector"] = list(_selector(stats))
        summary["handoff_ready"] = _handoff_ready(stats)
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    else:
        summary["stats"] = None
        summary["selector"] = None
        summary["handoff_ready"] = False

    _append_journal(summary)
    return summary


def main() -> None:
    state = _load_state()

    while True:
        spec = _next_spec(state)
        result = _run_trial(state, spec)

        if result["exit_code"] == 0 and result["stats"] is not None:
            best = state.get("best")
            if best is None or _selector(result["stats"]) > _selector(best["stats"]):
                state["best"] = {
                    "run_name": result["run_name"],
                    "run_dir": result["run_dir"],
                    "checkpoint": result["checkpoint"],
                    "seed": result["seed"],
                    "family": result["family"],
                    "stats": result["stats"],
                }
            if result["family"] == "full_long":
                state["last_long_resume"] = spec["resume"]
            state["iteration"] = int(state.get("iteration", 0)) + 1
            state["active_run"] = None
            state["status"] = "searching"
            _write_state(state)
            if result["handoff_ready"]:
                state["status"] = "handoff_ready"
                state["active_run"] = None
                _write_state(state)
                break
        else:
            state["iteration"] = int(state.get("iteration", 0)) + 1
            state["active_run"] = None
            state["status"] = "searching_after_failure"
            _write_state(state)
            time.sleep(5)


if __name__ == "__main__":
    main()
