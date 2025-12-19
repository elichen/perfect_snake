"""Summarize tracked Snake experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_runs(data_dir: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not os.path.isdir(data_dir):
        return runs
    for name in sorted(os.listdir(data_dir)):
        run_dir = os.path.join(data_dir, name)
        if not os.path.isdir(run_dir):
            continue
        run_path = os.path.join(run_dir, "run.json")
        if not os.path.exists(run_path):
            continue
        run = _load_json(run_path) or {}
        summary = _load_json(os.path.join(run_dir, "summary.json")) or {}
        runs.append({"run_dir": run_dir, "run": run, "summary": summary})
    runs.sort(key=_sort_key)
    return runs


def _sort_key(entry: Dict[str, Any]) -> tuple:
    run = entry.get("run", {})
    start_time = run.get("start_time", "")
    run_id = run.get("run_id", "")
    try:
        run_id_num = int(run_id)
    except Exception:
        run_id_num = 0
    return (start_time, run_id_num, entry.get("run_dir", ""))


def _fmt_eval(record: Optional[Dict[str, Any]]) -> str:
    if not record:
        return "na"
    score = record.get("mean_score")
    win = record.get("win_rate")
    score_str = "na" if score is None else f"{float(score):.2f}"
    win_str = "na" if win is None else f"{float(win) * 100:.1f}%"
    return f"{score_str}/{win_str}"


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(widths[i], len(row[i])) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt.format(*row))


def _resolve_run(data_dir: str, token: str) -> Optional[str]:
    if os.path.isdir(token) and os.path.exists(os.path.join(token, "run.json")):
        return token
    candidates = []
    for entry in _find_runs(data_dir):
        run_dir = entry["run_dir"]
        name = os.path.basename(run_dir)
        if token == name or token in name:
            candidates.append(run_dir)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        print(f"run not found: {token}", file=sys.stderr)
    else:
        print("multiple runs matched:", file=sys.stderr)
        for candidate in candidates:
            print(f"  {candidate}", file=sys.stderr)
    return None


def _command_list(data_dir: str, as_json: bool) -> None:
    runs = _find_runs(data_dir)
    if as_json:
        print(json.dumps(runs, indent=2, sort_keys=True))
        return

    rows: List[List[str]] = []
    for entry in runs:
        run = entry.get("run", {})
        summary = entry.get("summary", {})
        args = run.get("args", {})
        run_dir = os.path.basename(entry.get("run_dir", ""))
        exp_name = run.get("exp_name", "na")
        status = summary.get("status", "running")
        board = str(args.get("board_size", "na"))
        scale = str(args.get("network_scale", "na"))
        seed = str(args.get("seed", "na"))
        steps = str(args.get("timesteps", "na"))
        best_eval = _fmt_eval(summary.get("best_eval"))
        last_eval = _fmt_eval(summary.get("last_eval"))
        rows.append(
            [
                run_dir,
                exp_name,
                status,
                board,
                scale,
                seed,
                steps,
                best_eval,
                last_eval,
            ]
        )

    headers = [
        "run_dir",
        "exp_name",
        "status",
        "board",
        "scale",
        "seed",
        "timesteps",
        "best_eval",
        "last_eval",
    ]
    _print_table(headers, rows)


def _command_show(data_dir: str, token: str, as_json: bool) -> None:
    run_dir = _resolve_run(data_dir, token)
    if run_dir is None:
        sys.exit(1)
    run = _load_json(os.path.join(run_dir, "run.json")) or {}
    summary = _load_json(os.path.join(run_dir, "summary.json")) or {}
    payload = {"run_dir": run_dir, "run": run, "summary": summary}
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Snake experiments")
    parser.add_argument("--data-dir", type=str, default="experiments")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--json", action="store_true")

    show_parser = subparsers.add_parser("show", help="Show a specific experiment")
    show_parser.add_argument("run", type=str)
    show_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    command = args.command or "list"
    if command == "list":
        _command_list(args.data_dir, args.json)
    elif command == "show":
        _command_show(args.data_dir, args.run, args.json)
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
