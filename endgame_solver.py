"""Bounded exact endgame solver for near-full Snake states.

The solver estimates the optimal probability of eventually filling the board
from a concrete state, averaging over future uniform food spawns. It is meant
for high-fill diagnostics and teacher labels where shallow one-ply safety is
too weak.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from offline_trap_dagger import _as_cell, _reconstruct_history_snakes


Cell = tuple[int, int]
Snake = tuple[Cell, ...]

DIRECTIONS: dict[int, Cell] = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}
ACTION_DELTA = {0: -1, 1: 0, 2: 1}


class EndgameSolver:
    def __init__(
        self,
        *,
        board_size: int,
        max_depth: int = 800,
        max_nodes: int = 2_000_000,
        spawn_mode: str = "mean",
    ) -> None:
        if spawn_mode not in {"mean", "min", "max"}:
            raise ValueError(f"unsupported spawn_mode: {spawn_mode}")
        self.n = int(board_size)
        self.board_area = self.n * self.n
        self.max_depth = int(max_depth)
        self.max_nodes = int(max_nodes)
        self.spawn_mode = spawn_mode
        self.all_cells: tuple[Cell, ...] = tuple(
            (r, c) for r in range(self.n) for c in range(self.n)
        )
        self.nodes = 0
        self.cutoffs = 0
        self.cycles = 0
        self._visiting: set[tuple[Snake, int, Cell, int]] = set()

    def reset_counters(self) -> None:
        self.nodes = 0
        self.cutoffs = 0
        self.cycles = 0
        self._value.cache_clear()
        self._visiting.clear()

    def action_values(
        self,
        *,
        snake: Iterable[Cell],
        direction: int,
        food: Cell,
    ) -> list[float]:
        snake_tuple = tuple(snake)
        self.reset_counters()
        return [
            self._action_value(snake_tuple, int(direction), food, int(action), self.max_depth)
            for action in range(3)
        ]

    def _action_value(
        self,
        snake: Snake,
        direction: int,
        food: Cell,
        action: int,
        depth_left: int,
    ) -> float:
        if depth_left <= 0 or self.nodes >= self.max_nodes:
            self.cutoffs += 1
            return 0.0

        new_dir = (direction + ACTION_DELTA[action]) % 4
        dr, dc = DIRECTIONS[new_dir]
        hr, hc = snake[0]
        new_head = (hr + dr, hc + dc)

        if not (0 <= new_head[0] < self.n and 0 <= new_head[1] < self.n):
            return 0.0
        if new_head in snake[:-1]:
            return 0.0
        if new_head in snake and not (new_head == snake[-1] and new_head != food):
            return 0.0

        if new_head == food:
            next_snake = (new_head, *snake)
            if len(next_snake) >= self.board_area:
                return 1.0
            occupied = set(next_snake)
            empty = [cell for cell in self.all_cells if cell not in occupied]
            if not empty:
                return 1.0
            values = [
                self._value(next_snake, new_dir, next_food, depth_left - 1)
                for next_food in empty
            ]
            if self.spawn_mode == "min":
                return float(min(values))
            if self.spawn_mode == "max":
                return float(max(values))
            return float(np.mean(values))

        next_snake = (new_head, *snake[:-1])
        return self._value(next_snake, new_dir, food, depth_left - 1)

    @lru_cache(maxsize=None)
    def _value(self, snake: Snake, direction: int, food: Cell, depth_left: int) -> float:
        if len(snake) >= self.board_area:
            return 1.0
        if depth_left <= 0 or self.nodes >= self.max_nodes:
            self.cutoffs += 1
            return 0.0

        key = (snake, direction, food, depth_left)
        if key in self._visiting:
            self.cycles += 1
            return 0.0

        self.nodes += 1
        self._visiting.add(key)
        try:
            best = max(
                self._action_value(snake, direction, food, action, depth_left)
                for action in range(3)
            )
        finally:
            self._visiting.remove(key)
        return float(best)


def _summarize_values(values: list[float], policy_action: int) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    best_action = int(np.argmax(arr))
    sorted_values = np.sort(arr)
    margin = float(sorted_values[-1] - sorted_values[-2]) if len(sorted_values) > 1 else 0.0
    return {
        "values": [float(v) for v in values],
        "best_action": best_action,
        "best_value": float(arr[best_action]),
        "policy_action": int(policy_action),
        "policy_value": float(arr[int(policy_action)]),
        "margin": margin,
        "disagrees": bool(best_action != int(policy_action)),
        "improves_policy": bool(float(arr[best_action]) > float(arr[int(policy_action)]) + 1e-12),
    }


def analyze_failures(
    *,
    failures_path: Path,
    board_size: int,
    step_back: int,
    max_rows: int | None,
    max_depth: int,
    max_nodes: int,
    spawn_mode: str,
    terminal_score_min: int,
    terminal_score_max: int,
    candidate_score_min: int,
    output_path: Path | None,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in failures_path.read_text().splitlines() if line.strip()]
    solver = EndgameSolver(
        board_size=board_size,
        max_depth=max_depth,
        max_nodes=max_nodes,
        spawn_mode=spawn_mode,
    )
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    score_counts: Counter[int] = Counter()

    for row_idx, row in enumerate(rows):
        if max_rows is not None and len(records) >= max_rows:
            break
        terminal_score = int(row.get("score", 0))
        if terminal_score < terminal_score_min or terminal_score > terminal_score_max:
            skipped["terminal_score"] += 1
            continue
        history = row.get("history") or []
        target_idx = len(history) - 1 - step_back
        if target_idx < 0:
            skipped["history_too_short"] += 1
            continue
        snakes = _reconstruct_history_snakes(row)
        if snakes is None or len(snakes) != len(history):
            skipped["reconstruct"] += 1
            continue
        hist = history[target_idx]
        if int(hist["score"]) < candidate_score_min:
            skipped["candidate_score"] += 1
            continue

        values = solver.action_values(
            snake=snakes[target_idx],
            direction=int(hist["direction"]),
            food=_as_cell(hist["food"]),
        )
        summary = _summarize_values(values, int(hist["action"]))
        record = {
            "row": row_idx,
            "episode": int(row.get("episode", row_idx)),
            "terminal_score": terminal_score,
            "score": int(hist["score"]),
            "length": int(hist["length"]),
            "step_back": step_back,
            "food": list(_as_cell(hist["food"])),
            "head": list(_as_cell(hist["head"])),
            "direction": int(hist["direction"]),
            "solver_nodes": int(solver.nodes),
            "solver_cutoffs": int(solver.cutoffs),
            "solver_cycles": int(solver.cycles),
            **summary,
        }
        records.append(record)
        score_counts[int(hist["score"])] += 1

    actionable = [
        record
        for record in records
        if record["improves_policy"] and record["best_value"] > 0.0
    ]
    result = {
        "failures_path": str(failures_path),
        "board_size": board_size,
        "step_back": step_back,
        "max_depth": max_depth,
        "max_nodes": max_nodes,
        "spawn_mode": spawn_mode,
        "rows_seen": len(rows),
        "records": len(records),
        "actionable": len(actionable),
        "positive_best": sum(int(record["best_value"] > 0.0) for record in records),
        "positive_policy": sum(int(record["policy_value"] > 0.0) for record in records),
        "disagreements": sum(int(record["disagrees"]) for record in records),
        "improvements": sum(int(record["improves_policy"]) for record in records),
        "mean_best_value": float(np.mean([record["best_value"] for record in records]))
        if records
        else 0.0,
        "mean_policy_value": float(np.mean([record["policy_value"] for record in records]))
        if records
        else 0.0,
        "score_histogram": {str(k): v for k, v in sorted(score_counts.items())},
        "skipped": dict(sorted(skipped.items())),
        "top_actionable": sorted(
            actionable,
            key=lambda record: (record["best_value"] - record["policy_value"], record["margin"]),
            reverse=True,
        )[:20],
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"summary": result, "records": records}, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze harvested failures with an exact endgame solver")
    parser.add_argument("--failures", required=True)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--step-back", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=800)
    parser.add_argument("--max-nodes", type=int, default=2_000_000)
    parser.add_argument("--spawn-mode", choices=["mean", "min", "max"], default="mean")
    parser.add_argument("--terminal-score-min", type=int, default=393)
    parser.add_argument("--terminal-score-max", type=int, default=396)
    parser.add_argument("--candidate-score-min", type=int, default=390)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sys.setrecursionlimit(max(10_000, args.max_depth * 4))
    result = analyze_failures(
        failures_path=Path(args.failures).expanduser().resolve(),
        board_size=args.board_size,
        step_back=args.step_back,
        max_rows=args.max_rows,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        spawn_mode=args.spawn_mode,
        terminal_score_min=args.terminal_score_min,
        terminal_score_max=args.terminal_score_max,
        candidate_score_min=args.candidate_score_min,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
