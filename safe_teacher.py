"""Hamiltonian-cycle + safe-shortcut teacher, and an efficiency evaluator.

This is the GOAL.md "teacher feasibility probe": prove a teacher can play both
SAFELY (100% wins) and EFFICIENTLY (steps_per_food far below the ~100 the pure cycle
uses) before we invest in distilling it into a pure NN.

The pure cycle follower (distill/expert.py) wins 100% but at cycle-grade efficiency.
This adds the classic Tapsell-style safe shortcut: move toward the food along the
cycle order, taking the largest jump that (a) does not overshoot the food and (b)
stays safely behind the tail, so the snake can always fall back to the full cycle and
complete the board. Inference here is pure planning over the fixed cycle; it is a
TRAIN-TIME teacher, not the final policy.
"""

from __future__ import annotations

import argparse
import json
from collections import deque

import numpy as np

from snake_env import SnakeEnv
from distill.expert import find_aligned_cycle, relative_action_toward


def _move_cell(env: SnakeEnv, action: int):
    delta = {0: -1, 1: 0, 2: 1}
    new_dir = (env.direction + delta[action]) % 4
    dr, dc = env.DIRECTIONS[new_dir]
    hr, hc = env.snake_head
    return (hr + dr, hc + dc)


def _bfs_dist(start, goal, blocked, n):
    """Shortest 4-connected path length start->goal over cells not in `blocked`; None if none."""
    if start == goal:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        (r, c), d = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            cell = (nr, nc)
            if 0 <= nr < n and 0 <= nc < n and cell not in blocked and cell not in seen:
                if cell == goal:
                    return d + 1
                seen.add(cell)
                q.append((cell, d + 1))
    return None


def _reach_size(start, blocked, n):
    """Number of free cells reachable from start (flood fill)."""
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            cell = (nr, nc)
            if 0 <= nr < n and 0 <= nc < n and cell not in blocked and cell not in seen:
                seen.add(cell)
                q.append(cell)
    return len(seen)


def bfs_safe_action(env: SnakeEnv) -> int:
    """Greedy shortest-path-to-food, gated by tail reachability; stall safely otherwise.

    Acts from ANY state (no cycle alignment needed), so it can label off-cycle student
    states for DAgger. Safety rule: a move is safe if, after taking it, the new head can
    still reach the new tail through free cells (then the snake can survive by tail-
    following). Prefer the safe move that is closest to the food; if none reaches food,
    take the safe move that keeps the most space; if nothing is safe, best-effort.
    """
    n = env.n
    snake = env.snake
    head, tail = snake[0], snake[-1]
    food = env.food_pos
    body = set(snake)

    options = []
    for action in (0, 1, 2):
        cell = _move_cell(env, action)
        if not (0 <= cell[0] < n and 0 <= cell[1] < n):
            continue
        eats = cell == food
        # legal target: free, or the tail cell (which vacates) when not eating
        if cell in body and not (cell == tail and not eats):
            continue
        if eats:
            body_after = body | {cell}
            new_tail = tail
        else:
            body_after = (body - {tail}) | {cell}
            new_tail = snake[-2] if len(snake) >= 2 else cell
        blocked_after = body_after - {cell}
        reach_tail = cell == new_tail or _bfs_dist(cell, new_tail, blocked_after - {new_tail}, n) is not None
        food_d = -1 if eats else _bfs_dist(cell, food, body_after - {cell}, n)
        space = _reach_size(cell, blocked_after, n)
        options.append({"action": action, "eats": eats, "safe": reach_tail,
                        "food_d": food_d, "space": space})

    if not options:
        return 1

    safe = [o for o in options if o["safe"]]
    pool = safe if safe else options

    # 1) prefer reaching food: among the pool, the move with a finite path to food, min distance
    food_moves = [o for o in pool if o["eats"] or (o["food_d"] is not None and o["food_d"] >= 0)]
    if food_moves:
        best = min(food_moves, key=lambda o: (-1 if o["eats"] else o["food_d"], -o["space"]))
        return best["action"]
    # 2) no path to food: stall by keeping the most reachable space
    best = max(pool, key=lambda o: o["space"])
    return best["action"]


def _legal_neighbor_cells(env: SnakeEnv):
    """Yield (action, cell) for the 3 non-reverse moves whose cell is on the board."""
    hr, hc = env.snake_head
    # actions: 0=turn left, 1=straight, 2=turn right  (delta on direction)
    delta = {0: -1, 1: 0, 2: 1}
    for action in (0, 1, 2):
        new_dir = (env.direction + delta[action]) % 4
        dr, dc = env.DIRECTIONS[new_dir]
        cell = (hr + dr, hc + dc)
        if 0 <= cell[0] < env.n and 0 <= cell[1] < env.n:
            yield action, cell


def safe_shortcut_action(
    env: SnakeEnv,
    cyc_index: dict,
    N: int,
    *,
    disable_fill: float,
    tail_margin: int,
) -> int:
    """Return a relative action: take the largest safe shortcut toward food, else follow cycle.

    cyc_index maps cell -> position in the cycle list. The pure follow step goes to
    position (h-1) mod N, so forward cycle distance from cell a to b is
    (index[a] - index[b]) mod N.
    """
    snake = env.snake
    head = snake[0]
    tail = snake[-1]
    food = env.food_pos
    h = cyc_index[head]
    f = cyc_index[food]
    t = cyc_index[tail]

    dist_food = (h - f) % N
    dist_tail = (h - t) % N
    fill = len(snake) / float(N)

    body_block = set(snake[:-1])  # tail vacates this step (we never eat the tail), so it's free

    best_action = None
    best_jump = -1
    follow_action = None
    for action, cell in _legal_neighbor_cells(env):
        if cell in body_block:
            continue
        jump = (h - cyc_index[cell]) % N
        if jump == 1:
            follow_action = action  # pure-cycle fallback
        # shortcut conditions
        if fill >= disable_fill:
            continue
        if jump < 1:
            continue
        if jump > dist_food:
            continue  # would overshoot the food on the cycle
        if jump >= dist_tail - tail_margin:
            continue  # must stay safely behind the tail
        if jump > best_jump:
            best_jump = jump
            best_action = action

    if best_action is not None and best_jump > 1:
        return best_action
    if follow_action is not None:
        return follow_action
    # No safe cycle move found (should not happen if the invariant holds); take any legal
    # move so the failure is visible in the win rate.
    for action, _ in _legal_neighbor_cells(env):
        return action
    return 1


def evaluate(
    *,
    board_size: int,
    seeds: list[int],
    mode: str,
    disable_fill: float,
    tail_margin: int,
    max_steps: int,
) -> dict:
    N = board_size * board_size
    perfect = N - 3
    fill_edges = [0.0, 0.5, 0.8, 0.95, 1.01]
    fill_labels = ["0_50", "50_80", "80_95", "95_100"]

    results = []
    steps_in_bucket = np.zeros(len(fill_labels))
    food_in_bucket = np.zeros(len(fill_labels))
    total_steps = 0
    total_food = 0
    wins = 0

    for seed in seeds:
        env = SnakeEnv(n=board_size, seed=seed)
        env.reset(seed=seed)
        cycle, _ = find_aligned_cycle(env)
        cyc_index = {cell: i for i, cell in enumerate(cycle)}
        done = False
        info = {}
        steps = 0
        while not done and steps < max_steps:
            fill = len(env.snake) / float(N)
            bucket = int(np.searchsorted(fill_edges, fill, side="right") - 1)
            bucket = max(0, min(len(fill_labels) - 1, bucket))
            prev_score = int(info.get("score", len(env.snake) - 3))
            if mode == "cycle":
                h = cyc_index[env.snake_head]
                target = cycle[(h - 1) % len(cycle)]
                action = relative_action_toward(env, target)
            elif mode == "bfs":
                action = bfs_safe_action(env)
            elif mode == "repo_cycle":
                from rnn_cycle_shortcut_patch import _teacher_action as _repo_teacher
                action = _repo_teacher(env, cycle, cyc_index, "cycle", 2000, 64, -1)
            else:
                action = safe_shortcut_action(
                    env, cyc_index, N, disable_fill=disable_fill, tail_margin=tail_margin
                )
            _, _, terminated, truncated, info = env.step(action)
            steps += 1
            steps_in_bucket[bucket] += 1
            if int(info.get("score", 0)) > prev_score:
                food_in_bucket[bucket] += 1
            done = terminated or truncated

        score = int(info.get("score", 0))
        reason = str(info.get("reason", "unknown"))
        win = score >= perfect
        wins += int(win)
        total_steps += steps
        total_food += score
        results.append(
            {"seed": seed, "score": score, "steps": steps, "reason": reason, "win": win}
        )

    spf_bucket = {
        fill_labels[i]: (float(steps_in_bucket[i] / food_in_bucket[i]) if food_in_bucket[i] else None)
        for i in range(len(fill_labels))
    }
    win_steps = [r["steps"] for r in results if r["win"]]
    win_scores_ok = wins == len(seeds)
    summary = {
        "mode": mode,
        "board_size": board_size,
        "episodes": len(seeds),
        "wins": wins,
        "win_rate": wins / max(1, len(seeds)),
        "steps_per_food_all": total_steps / max(1, total_food),
        "steps_per_food_by_fill": spf_bucket,
        "mean_win_steps": float(np.mean(win_steps)) if win_steps else None,
        "p95_win_steps": float(np.percentile(win_steps, 95)) if win_steps else None,
        "disable_fill": disable_fill,
        "tail_margin": tail_margin,
        "all_win": win_scores_ok,
        "failures": [r for r in results if not r["win"]][:10],
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Safe-shortcut Snake teacher feasibility probe")
    p.add_argument("--board-size", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=70001)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--mode", choices=["cycle", "safe", "bfs", "repo_cycle"], default="safe")
    p.add_argument("--disable-fill", type=float, default=0.5,
                   help="disable shortcuts once fill >= this (1.0 = never disable)")
    p.add_argument("--tail-margin", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200000)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    summary = evaluate(
        board_size=args.board_size,
        seeds=seeds,
        mode=args.mode,
        disable_fill=args.disable_fill,
        tail_margin=args.tail_margin,
        max_steps=args.max_steps,
    )
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
