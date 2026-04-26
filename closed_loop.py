"""Autonomous closed-loop experiment driver for Snake win-rate hill climbing."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from eval import evaluate_checkpoint


ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
STATE_PATH = EXPERIMENTS_DIR / "closed_loop_state.json"
JOURNAL_PATH = EXPERIMENTS_DIR / "closed_loop_journal.jsonl"
PYTHON_BIN = ROOT / ".venv" / "bin" / "python"


BASE_CONFIG: dict[str, Any] = {
    "board_size": 20,
    "timesteps": 4_000_000,
    "num_envs": 64,
    "horizon": 256,
    "minibatch_size": 4096,
    "device": "cpu",
    "gamma": 0.999,
    "gae_lambda": 0.9,
    "vf_clip_coef": 1.0,
    "network_scale": 2,
    "curriculum_prob": 0.10,
    "curriculum_min_fill": 0.90,
    "curriculum_max_fill": 0.98,
    "curriculum_follow_bonus": 0.005,
    "curriculum_follow_min_fill": 0.95,
    "lr": 5e-6,
    "seed": 42,
}


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True))
        handle.write("\n")


def _fmt_num(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _config_fingerprint(config: dict[str, Any]) -> str:
    keys = (
        "curriculum_follow_bonus",
        "curriculum_follow_min_fill",
        "curriculum_prob",
        "curriculum_max_fill",
        "lr",
        "seed",
    )
    return "|".join(f"{key}={config[key]}" for key in keys)


def _score_tuple(stats: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(stats["win_rate"]),
        float(stats["mean_score"]),
        float(stats["phase_gte95_rate"]),
        -float(stats["phase_lt20_rate"]),
    )


def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _trial_key(incumbent_path: str, config: dict[str, Any]) -> str:
    return f"{incumbent_path}|{_config_fingerprint(config)}"


def _state_payload(
    *,
    incumbent: "EvaluatedCheckpoint | None",
    last_run: dict[str, Any] | None,
    active_run: dict[str, Any] | None,
    tried: Iterable[str],
    loop_index: int,
    status: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "time": _timestamp(),
        "status": status,
        "loop_index": int(loop_index),
        "tried": sorted(tried),
        "last_run": last_run,
        "active_run": active_run,
    }
    if incumbent is not None:
        payload["incumbent"] = incumbent.to_dict()
    return payload


@dataclass
class EvaluatedCheckpoint:
    path: str
    source: str
    config: dict[str, Any]
    selection_stats: dict[str, Any]
    confirmed_det_stats: dict[str, Any] | None = None
    confirmed_stoch_stats: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source": self.source,
            "config": self.config,
            "selection_stats": self.selection_stats,
            "confirmed_det_stats": self.confirmed_det_stats,
            "confirmed_stoch_stats": self.confirmed_stoch_stats,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluatedCheckpoint":
        return cls(
            path=str(payload["path"]),
            source=str(payload["source"]),
            config=dict(payload["config"]),
            selection_stats=dict(payload["selection_stats"]),
            confirmed_det_stats=payload.get("confirmed_det_stats"),
            confirmed_stoch_stats=payload.get("confirmed_stoch_stats"),
        )


def log_event(event_type: str, **payload: Any) -> None:
    _append_jsonl(JOURNAL_PATH, {"time": _timestamp(), "event": event_type, **payload})


def load_state() -> dict[str, Any] | None:
    if not STATE_PATH.exists():
        return None
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return None
        return payload
    except Exception:
        return None


def evaluate_candidate(
    checkpoint_path: Path,
    *,
    source: str,
    config: dict[str, Any],
    episodes: int,
    deterministic: bool,
    seed: int,
) -> dict[str, Any]:
    return evaluate_checkpoint(
        checkpoint_path=str(checkpoint_path),
        board_size=int(config["board_size"]),
        episodes=episodes,
        seed=seed,
        deterministic=deterministic,
        device=str(config["device"]),
        network_scale=int(config["network_scale"]),
        flood_fill=True,
        aux_flood_fill=True,
        head_centered=True,
    )


def should_attempt_confirmation(candidate: EvaluatedCheckpoint, det_target: float) -> bool:
    selection_win_rate = float(candidate.selection_stats.get("win_rate", 0.0) or 0.0)
    selection_win_count = int(candidate.selection_stats.get("win_count", 0) or 0)
    return selection_win_rate >= det_target or selection_win_count > 0


def passes_consistency_target(
    candidate: EvaluatedCheckpoint,
    *,
    det_target: float,
    stoch_target: float,
    confirm_episodes: int,
    confirm_seed: int,
) -> bool:
    det_stats = evaluate_candidate(
        Path(candidate.path),
        source=candidate.source,
        config=candidate.config,
        episodes=confirm_episodes,
        deterministic=True,
        seed=confirm_seed,
    )
    candidate.confirmed_det_stats = det_stats
    log_event(
        "confirm_det",
        checkpoint=candidate.path,
        win_rate=det_stats["win_rate"],
        mean_score=det_stats["mean_score"],
        episodes=confirm_episodes,
    )
    print(
        f"confirm_det: checkpoint={candidate.path} "
        f"win_rate={det_stats['win_rate']*100:.1f}% mean_score={det_stats['mean_score']:.2f}"
    )
    if float(det_stats["win_rate"]) < det_target:
        return False

    stoch_stats = evaluate_candidate(
        Path(candidate.path),
        source=candidate.source,
        config=candidate.config,
        episodes=confirm_episodes,
        deterministic=False,
        seed=confirm_seed + 10_000,
    )
    candidate.confirmed_stoch_stats = stoch_stats
    log_event(
        "confirm_stoch",
        checkpoint=candidate.path,
        win_rate=stoch_stats["win_rate"],
        mean_score=stoch_stats["mean_score"],
        episodes=confirm_episodes,
    )
    print(
        f"confirm_stoch: checkpoint={candidate.path} "
        f"win_rate={stoch_stats['win_rate']*100:.1f}% mean_score={stoch_stats['mean_score']:.2f}"
    )
    return float(stoch_stats["win_rate"]) >= stoch_target


def choose_better(
    incumbent: EvaluatedCheckpoint | None,
    challenger: EvaluatedCheckpoint,
) -> EvaluatedCheckpoint:
    if incumbent is None:
        return challenger
    if _score_tuple(challenger.selection_stats) > _score_tuple(incumbent.selection_stats):
        return challenger
    return incumbent


def candidate_paths_for_run(run_dir: Path) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    for filename, source in (
        ("first_train_win.pt", "first_train_win"),
        ("latest_train_win.pt", "latest_train_win"),
        ("best_eval.pt", "best_eval"),
    ):
        path = run_dir / filename
        if path.exists():
            candidates.append((path, source))
    return candidates


def discover_existing_candidates() -> list[tuple[Path, str, dict[str, Any]]]:
    candidates: list[tuple[Path, str, dict[str, Any]]] = []
    base_paths = [
        (
            ROOT / "experiments/exp079_bonus_0p005_capture_177319184061/first_train_win.pt",
            "exp079_first_train_win",
            dict(BASE_CONFIG),
        ),
        (
            ROOT / "experiments/exp079_bonus_0p005_capture_177319184061/latest_train_win.pt",
            "exp079_latest_train_win",
            dict(BASE_CONFIG),
        ),
        (
            ROOT / "experiments/exp079_bonus_0p005_capture_177319184061/best_eval.pt",
            "exp079_best_eval",
            dict(BASE_CONFIG),
        ),
        (
            ROOT / "experiments/exp078_bonus_0p005_177318890304/best_eval.pt",
            "exp078_best_eval",
            dict(BASE_CONFIG),
        ),
        (
            ROOT / "experiments/exp074_multi_path_curriculum_ft_177317600936/best_eval.pt",
            "exp074_best_eval",
            dict(BASE_CONFIG),
        ),
    ]
    for path, source, config in base_paths:
        if path.exists():
            candidates.append((path, source, config))
    return candidates


def build_train_command(exp_name: str, resume_path: Path, config: dict[str, Any]) -> list[str]:
    return [
        str(PYTHON_BIN),
        "-u",
        "train.py",
        "--board-size",
        str(config["board_size"]),
        "--timesteps",
        str(config["timesteps"]),
        "--num-envs",
        str(config["num_envs"]),
        "--horizon",
        str(config["horizon"]),
        "--minibatch-size",
        str(config["minibatch_size"]),
        "--symmetric",
        "--device",
        str(config["device"]),
        "--gamma",
        str(config["gamma"]),
        "--gae-lambda",
        str(config["gae_lambda"]),
        "--vf-clip-coef",
        str(config["vf_clip_coef"]),
        "--network-scale",
        str(config["network_scale"]),
        "--flood-fill",
        "--aux-flood-fill",
        "--aux-flood-fill-coef",
        "1.0",
        "--curriculum-prob",
        str(config["curriculum_prob"]),
        "--curriculum-min-fill",
        str(config["curriculum_min_fill"]),
        "--curriculum-max-fill",
        str(config["curriculum_max_fill"]),
        "--curriculum-follow-bonus",
        str(config["curriculum_follow_bonus"]),
        "--curriculum-follow-min-fill",
        str(config["curriculum_follow_min_fill"]),
        "--head-centered",
        "--lr",
        str(config["lr"]),
        "--no-anneal-lr",
        "--eval-every-steps",
        "1000000",
        "--eval-deterministic",
        "--eval-episodes",
        "5",
        "--seed",
        str(config["seed"]),
        "--resume",
        str(resume_path),
        "--exp-name",
        exp_name,
    ]


def wait_for_run_dir(exp_name: str, previous_paths: set[Path], timeout_seconds: int = 120) -> Path | None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        matches = sorted(EXPERIMENTS_DIR.glob(f"{exp_name}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in matches:
            if path not in previous_paths:
                return path
        time.sleep(1.0)
    matches = sorted(EXPERIMENTS_DIR.glob(f"{exp_name}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def run_training(
    exp_name: str,
    resume_path: Path,
    config: dict[str, Any],
    *,
    persist_active_run: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path | None, int, Path, int]:
    previous = set(EXPERIMENTS_DIR.glob(f"{exp_name}_*"))
    log_path = EXPERIMENTS_DIR / f"{exp_name}.driver.log"
    cmd = build_train_command(exp_name, resume_path, config)
    print(f"launch: exp_name={exp_name} resume={_relpath(resume_path)} config={_config_fingerprint(config)}")
    log_event(
        "run_launch",
        exp_name=exp_name,
        resume=_relpath(resume_path),
        config=config,
        command=cmd,
    )
    with open(log_path, "w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if persist_active_run is not None:
            persist_active_run(
                {
                    "exp_name": exp_name,
                    "resume_path": _relpath(resume_path),
                    "config": dict(config),
                    "log_path": _relpath(log_path),
                    "pid": proc.pid,
                    "run_dir": None,
                }
            )
        run_dir = wait_for_run_dir(exp_name, previous)
        if persist_active_run is not None:
            persist_active_run(
                {
                    "exp_name": exp_name,
                    "resume_path": _relpath(resume_path),
                    "config": dict(config),
                    "log_path": _relpath(log_path),
                    "pid": proc.pid,
                    "run_dir": _relpath(run_dir) if run_dir is not None else None,
                }
            )
        exit_code = proc.wait()
    log_event(
        "run_complete",
        exp_name=exp_name,
        exit_code=exit_code,
        run_dir=_relpath(run_dir) if run_dir is not None else None,
        log_path=_relpath(log_path),
    )
    print(f"run_complete: exp_name={exp_name} exit_code={exit_code} run_dir={_relpath(run_dir) if run_dir else 'missing'}")
    return run_dir, exit_code, log_path, proc.pid


def recover_active_run(active_run: dict[str, Any] | None) -> tuple[Path | None, dict[str, Any] | None]:
    if not active_run:
        return None, None

    exp_name = str(active_run.get("exp_name") or "")
    if not exp_name:
        return None, None

    pid_value = active_run.get("pid")
    pid = int(pid_value) if isinstance(pid_value, int) else None
    run_dir_value = active_run.get("run_dir")
    run_dir = ROOT / str(run_dir_value) if isinstance(run_dir_value, str) and run_dir_value else None
    if run_dir is None:
        matches = sorted(EXPERIMENTS_DIR.glob(f"{exp_name}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        run_dir = matches[0] if matches else None

    if _pid_is_alive(pid):
        print(f"recover_active_run: waiting for pid={pid} exp_name={exp_name}")
        log_event(
            "recover_active_run_wait",
            exp_name=exp_name,
            pid=pid,
            run_dir=_relpath(run_dir) if run_dir is not None else None,
        )
        while _pid_is_alive(pid):
            time.sleep(5.0)

    log_event(
        "recover_active_run",
        exp_name=exp_name,
        pid=pid,
        run_dir=_relpath(run_dir) if run_dir is not None else None,
    )
    recovered_last_run = {
        "exp_name": exp_name,
        "run_dir": _relpath(run_dir) if run_dir is not None else None,
        "exit_code": None,
        "log_path": active_run.get("log_path"),
        "config": dict(active_run.get("config", {})),
        "pid": pid,
        "recovered": True,
    }
    return run_dir, recovered_last_run


def evaluate_checkpoint_set(
    checkpoint_items: Iterable[tuple[Path, str, dict[str, Any]]],
    *,
    selection_episodes: int,
    selection_seed: int,
) -> EvaluatedCheckpoint | None:
    incumbent: EvaluatedCheckpoint | None = None
    for path, source, config in checkpoint_items:
        if not path.exists():
            continue
        print(f"select_eval: checkpoint={_relpath(path)} source={source}")
        stats = evaluate_candidate(
            path,
            source=source,
            config=config,
            episodes=selection_episodes,
            deterministic=True,
            seed=selection_seed,
        )
        candidate = EvaluatedCheckpoint(
            path=str(path),
            source=source,
            config=dict(config),
            selection_stats=stats,
        )
        log_event(
            "selection_eval",
            checkpoint=_relpath(path),
            source=source,
            win_rate=stats["win_rate"],
            mean_score=stats["mean_score"],
            episodes=selection_episodes,
        )
        incumbent = choose_better(incumbent, candidate)
    return incumbent


def build_neighbor_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    neighbors: list[dict[str, Any]] = []

    def add(key: str, value: Any) -> None:
        if config[key] == value:
            return
        updated = dict(config)
        updated[key] = value
        neighbors.append(updated)

    for seed in (43, 44, 45):
        add("seed", seed)
    for bonus in (0.0025, 0.005, 0.01, 0.02):
        add("curriculum_follow_bonus", bonus)
    for follow_min in (0.93, 0.95, 0.97):
        add("curriculum_follow_min_fill", follow_min)
    for curr_prob in (0.05, 0.10, 0.15):
        add("curriculum_prob", curr_prob)
    for curr_max in (0.96, 0.98, 0.99):
        add("curriculum_max_fill", curr_max)
    for lr in (3e-6, 5e-6, 1e-5):
        add("lr", lr)
    return neighbors


def make_exp_name(loop_index: int, config: dict[str, Any]) -> str:
    return (
        f"autoloop_{loop_index:03d}_"
        f"b{_fmt_num(float(config['curriculum_follow_bonus']))}_"
        f"fm{_fmt_num(float(config['curriculum_follow_min_fill']))}_"
        f"cp{_fmt_num(float(config['curriculum_prob']))}_"
        f"cm{_fmt_num(float(config['curriculum_max_fill']))}_"
        f"lr{_fmt_num(float(config['lr']))}_"
        f"s{int(config['seed'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Snake training in a local closed loop")
    parser.add_argument("--selection-episodes", type=int, default=50)
    parser.add_argument("--confirm-episodes", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=12345)
    parser.add_argument("--target-det-win-rate", type=float, default=0.05)
    parser.add_argument("--target-stoch-win-rate", type=float, default=0.02)
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    loaded_state = load_state()
    incumbent: EvaluatedCheckpoint | None = None
    tried: set[str] = set()
    loop_index = 1
    last_run: dict[str, Any] | None = None
    active_run: dict[str, Any] | None = None

    if loaded_state is not None and loaded_state.get("incumbent"):
        incumbent = EvaluatedCheckpoint.from_dict(dict(loaded_state["incumbent"]))
        tried = set(loaded_state.get("tried", []))
        loop_index = int(loaded_state.get("loop_index", 1))
        last_run = loaded_state.get("last_run")
        active_run = loaded_state.get("active_run")
        print(
            f"resume_state: checkpoint={_relpath(Path(incumbent.path))} "
            f"loop_index={loop_index} tried={len(tried)}"
        )
        log_event(
            "resume_state",
            checkpoint=_relpath(Path(incumbent.path)),
            loop_index=loop_index,
            tried=len(tried),
        )
        _atomic_write_json(
            STATE_PATH,
            _state_payload(
                incumbent=incumbent,
                last_run=last_run,
                active_run=active_run,
                tried=tried,
                loop_index=loop_index,
                status="resumed",
            ),
        )
    else:
        _atomic_write_json(
            STATE_PATH,
            _state_payload(
                incumbent=None,
                last_run=None,
                active_run=None,
                tried=tried,
                loop_index=loop_index,
                status="bootstrapping",
            ),
        )

        initial_candidates = discover_existing_candidates()
        incumbent = evaluate_checkpoint_set(
            initial_candidates,
            selection_episodes=args.selection_episodes,
            selection_seed=args.selection_seed,
        )
        if incumbent is None:
            raise RuntimeError("No initial checkpoints found for closed-loop search")

        print(
            f"incumbent: checkpoint={_relpath(Path(incumbent.path))} "
            f"win_rate={incumbent.selection_stats['win_rate']*100:.1f}% "
            f"mean_score={incumbent.selection_stats['mean_score']:.2f}"
        )
        _atomic_write_json(
            STATE_PATH,
            _state_payload(
                incumbent=incumbent,
                last_run=None,
                active_run=None,
                tried=tried,
                loop_index=loop_index,
                status="initialized",
            ),
        )

    if active_run is not None:
        recovered_run_dir, recovered_last_run = recover_active_run(active_run)
        run_candidates: list[tuple[Path, str, dict[str, Any]]] = []
        if recovered_run_dir is not None and recovered_last_run is not None:
            recovered_config = dict(recovered_last_run.get("config", {}))
            for path, source in candidate_paths_for_run(recovered_run_dir):
                run_candidates.append((path, f"{recovered_last_run['exp_name']}:{source}", recovered_config))
        best_recovered_candidate = evaluate_checkpoint_set(
            run_candidates,
            selection_episodes=args.selection_episodes,
            selection_seed=args.selection_seed,
        )
        if best_recovered_candidate is not None:
            incumbent = choose_better(incumbent, best_recovered_candidate)
        last_run = recovered_last_run or last_run
        active_run = None
        _atomic_write_json(
            STATE_PATH,
            _state_payload(
                incumbent=incumbent,
                last_run=last_run,
                active_run=active_run,
                tried=tried,
                loop_index=loop_index,
                status="recovered",
            ),
        )

    if should_attempt_confirmation(incumbent, args.target_det_win_rate) and passes_consistency_target(
        incumbent,
        det_target=args.target_det_win_rate,
        stoch_target=args.target_stoch_win_rate,
        confirm_episodes=args.confirm_episodes,
        confirm_seed=args.selection_seed + 20_000,
    ):
        print("target_reached: incumbent already satisfies the consistency threshold")
        _atomic_write_json(
            STATE_PATH,
            _state_payload(
                incumbent=incumbent,
                last_run=last_run,
                active_run=active_run,
                tried=tried,
                loop_index=loop_index,
                status="completed",
            ),
        )
        return

    while True:
        neighbor_configs = build_neighbor_configs(incumbent.config)
        chosen_config: dict[str, Any] | None = None
        for candidate_config in neighbor_configs:
            key = _trial_key(incumbent.path, candidate_config)
            if key in tried:
                continue
            chosen_config = candidate_config
            tried.add(key)
            break
        if chosen_config is None:
            fallback = dict(incumbent.config)
            fallback["seed"] = int(fallback["seed"]) + 1
            key = _trial_key(incumbent.path, fallback)
            if key in tried:
                tried.clear()
            chosen_config = fallback
            tried.add(key)

        exp_name = make_exp_name(loop_index, chosen_config)
        def persist_active_run(payload: dict[str, Any]) -> None:
            nonlocal active_run
            active_run = dict(payload)
            _atomic_write_json(
                STATE_PATH,
                _state_payload(
                    incumbent=incumbent,
                    last_run=last_run,
                    active_run=active_run,
                    tried=tried,
                    loop_index=loop_index,
                    status="running",
                ),
            )

        run_dir, exit_code, log_path, pid = run_training(
            exp_name,
            Path(incumbent.path),
            chosen_config,
            persist_active_run=persist_active_run,
        )
        active_run = None
        last_run = {
            "exp_name": exp_name,
            "run_dir": _relpath(run_dir) if run_dir is not None else None,
            "exit_code": exit_code,
            "log_path": _relpath(log_path),
            "config": chosen_config,
            "pid": pid,
        }

        run_candidates: list[tuple[Path, str, dict[str, Any]]] = []
        if run_dir is not None:
            for path, source in candidate_paths_for_run(run_dir):
                run_candidates.append((path, f"{exp_name}:{source}", chosen_config))

        best_run_candidate = evaluate_checkpoint_set(
            run_candidates,
            selection_episodes=args.selection_episodes,
            selection_seed=args.selection_seed,
        )
        if best_run_candidate is not None:
            better = choose_better(incumbent, best_run_candidate)
            if better is not incumbent:
                incumbent = better
                print(
                    f"promote: checkpoint={_relpath(Path(incumbent.path))} "
                    f"win_rate={incumbent.selection_stats['win_rate']*100:.1f}% "
                    f"mean_score={incumbent.selection_stats['mean_score']:.2f}"
                )
                log_event(
                    "promote",
                    checkpoint=_relpath(Path(incumbent.path)),
                    source=incumbent.source,
                    win_rate=incumbent.selection_stats["win_rate"],
                    mean_score=incumbent.selection_stats["mean_score"],
                )

        status = "running"
        if should_attempt_confirmation(incumbent, args.target_det_win_rate) and passes_consistency_target(
            incumbent,
            det_target=args.target_det_win_rate,
            stoch_target=args.target_stoch_win_rate,
            confirm_episodes=args.confirm_episodes,
            confirm_seed=args.selection_seed + loop_index * 1000,
        ):
            status = "completed"
            _atomic_write_json(
                STATE_PATH,
                _state_payload(
                    incumbent=incumbent,
                    last_run=last_run,
                    active_run=active_run,
                    tried=tried,
                    loop_index=loop_index,
                    status=status,
                ),
            )
            print(
                f"target_reached: checkpoint={_relpath(Path(incumbent.path))} "
                f"det_win_rate={incumbent.confirmed_det_stats['win_rate']*100:.1f}% "
                f"stoch_win_rate={incumbent.confirmed_stoch_stats['win_rate']*100:.1f}%"
            )
            return

        _atomic_write_json(
            STATE_PATH,
            _state_payload(
                incumbent=incumbent,
                last_run=last_run,
                active_run=active_run,
                tried=tried,
                loop_index=loop_index,
                status=status,
            ),
        )
        loop_index += 1


if __name__ == "__main__":
    main()
