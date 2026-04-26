"""Watchdog for the standalone distillation loop."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
STATE_PATH = EXPERIMENTS_DIR / "distill_loop_state.json"
LOG_PATH = EXPERIMENTS_DIR / "distill_loop_watchdog.log"
RESTART_LOG_PATH = EXPERIMENTS_DIR / "distill_loop_watchdog.restarts.jsonl"
SESSION_NAME = "distill_loop_driver"
LAUNCH_CMD = (
    "screen -dmS distill_loop_driver "
    "zsh -lc 'cd /Users/elichen/code/perfect_snake && "
    ".venv/bin/python -u distill_loop.py >> experiments/distill_loop.screen.log 2>&1'"
)


def session_alive() -> bool:
    result = subprocess.run(
        ["screen", "-ls"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return SESSION_NAME in result.stdout


def read_state() -> dict | None:
    if not STATE_PATH.exists():
        return None
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def append_restart(reason: str, state: dict | None) -> None:
    RESTART_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": reason,
        "state": state,
    }
    with open(RESTART_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def launch_loop(reason: str) -> None:
    state = read_state()
    subprocess.run(["zsh", "-lc", LAUNCH_CMD], cwd=ROOT, check=False)
    append_restart(reason, state)
    append_log(f"relaunched {SESSION_NAME}: {reason}")


def main() -> None:
    append_log("watchdog started")
    while True:
        alive = session_alive()
        if not alive:
            launch_loop("screen session missing")
        time.sleep(30)


if __name__ == "__main__":
    main()
