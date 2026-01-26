"""Experiment tracking helpers for Snake training."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional


def _utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True, default=_json_default)
        handle.write("\n")
    os.replace(tmp_path, path)


def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=_json_default))
        handle.write("\n")


def _run_git(args: list[str], cwd: str) -> Optional[str]:
    if shutil.which("git") is None:
        return None
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _git_info(cwd: str) -> Dict[str, Any]:
    commit = _run_git(["rev-parse", "HEAD"], cwd)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    status = _run_git(["status", "--porcelain"], cwd)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
    }


def _is_better_eval(candidate: Dict[str, Any], current: Optional[Dict[str, Any]]) -> bool:
    if current is None:
        return True
    cand_win = float(candidate.get("win_rate", 0.0) or 0.0)
    curr_win = float(current.get("win_rate", 0.0) or 0.0)
    if cand_win != curr_win:
        return cand_win > curr_win
    cand_score = float(candidate.get("mean_score", 0.0) or 0.0)
    curr_score = float(current.get("mean_score", 0.0) or 0.0)
    return cand_score > curr_score


class ExperimentTracker:
    def __init__(
        self,
        *,
        exp_name: str,
        run_id: str,
        data_dir: str,
        args: Dict[str, Any],
        config: Dict[str, Any],
        command: str,
        cwd: str,
    ) -> None:
        self.exp_name = str(exp_name)
        self.run_id = str(run_id)
        self.data_dir = data_dir
        self.run_dir = os.path.join(self.data_dir, f"{self.exp_name}_{self.run_id}")
        self.start_time = _utc_now()
        self.command = command
        self.args = args
        self.config = config
        self.git = _git_info(cwd)

        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        self.summary_path = os.path.join(self.run_dir, "summary.json")
        self.run_path = os.path.join(self.run_dir, "run.json")
        self.index_path = os.path.join(self.data_dir, "index.jsonl")

        self.best_eval: Optional[Dict[str, Any]] = None
        self.last_eval: Optional[Dict[str, Any]] = None
        self.last_train: Optional[Dict[str, Any]] = None
        self.checkpoints: list[Dict[str, Any]] = []
        self._checkpoint_paths: set[str] = set()
        self._disabled = False

        self._safe(self._init_paths)

    def _init_paths(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)

        # Archive source files for reproducibility
        code_dir = os.path.join(self.run_dir, "code")
        os.makedirs(code_dir, exist_ok=True)
        for src_file in ["snake_env.py", "train.py"]:
            src_path = os.path.join(os.path.dirname(__file__), src_file)
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(code_dir, src_file))

        run_record = {
            "schema_version": 1,
            "exp_name": self.exp_name,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "start_time": self.start_time,
            "command": self.command,
            "args": self.args,
            "config": self.config,
            "git": self.git,
            "python": {
                "executable": sys.executable,
                "version": sys.version.split()[0],
            },
        }
        _write_json(self.run_path, run_record)

    def _disable(self, error: Exception) -> None:
        if not self._disabled:
            print(f"experiment_tracker_disabled: {error}", file=sys.stderr)
        self._disabled = True

    def _safe(self, fn) -> None:
        if self._disabled:
            return
        try:
            fn()
        except Exception as exc:
            self._disable(exc)

    def log_train(self, logs: Dict[str, Any]) -> None:
        def _log() -> None:
            record = {"type": "train", "time": _utc_now(), **logs}
            _append_jsonl(self.metrics_path, record)
            self.last_train = record

        self._safe(_log)

    def log_eval(
        self,
        stats: Dict[str, Any],
        *,
        agent_steps: int,
        epoch: int,
        deterministic: bool,
    ) -> None:
        def _log() -> None:
            record = {
                "type": "eval",
                "time": _utc_now(),
                "agent_steps": int(agent_steps),
                "epoch": int(epoch),
                "deterministic": bool(deterministic),
                **stats,
            }
            _append_jsonl(self.metrics_path, record)
            self.last_eval = record
            if _is_better_eval(record, self.best_eval):
                self.best_eval = record

        self._safe(_log)

    def log_checkpoint(self, path: str, *, epoch: int, agent_steps: int) -> None:
        def _log() -> None:
            if path in self._checkpoint_paths:
                return
            if not os.path.exists(path):
                return
            record = {
                "type": "checkpoint",
                "time": _utc_now(),
                "path": path,
                "epoch": int(epoch),
                "agent_steps": int(agent_steps),
            }
            _append_jsonl(self.metrics_path, record)
            self.checkpoints.append(record)
            self._checkpoint_paths.add(path)

        self._safe(_log)

    def finalize(
        self,
        *,
        status: str,
        final_checkpoint: Optional[str],
        elapsed_seconds: Optional[float] = None,
    ) -> None:
        def _finalize() -> None:
            end_time = _utc_now()
            summary = {
                "schema_version": 1,
                "exp_name": self.exp_name,
                "run_id": self.run_id,
                "run_dir": self.run_dir,
                "start_time": self.start_time,
                "end_time": end_time,
                "elapsed_seconds": float(elapsed_seconds) if elapsed_seconds is not None else None,
                "status": status,
                "last_train": self.last_train,
                "last_eval": self.last_eval,
                "best_eval": self.best_eval,
                "checkpoints": self.checkpoints,
                "final_checkpoint": final_checkpoint,
            }
            _write_json(self.summary_path, summary)

            index_record = {
                "time": end_time,
                "exp_name": self.exp_name,
                "run_id": self.run_id,
                "run_dir": self.run_dir,
                "status": status,
                "best_eval": self.best_eval,
                "last_eval": self.last_eval,
                "final_checkpoint": final_checkpoint,
            }
            _append_jsonl(self.index_path, index_record)

        self._safe(_finalize)
