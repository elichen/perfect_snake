"""Perfect Hamiltonian-cycle expert for Snake."""

from __future__ import annotations

import argparse

import numpy as np

from eval_metrics import summarize_phase_metrics
from snake_env import SnakeEnv


def find_aligned_cycle(env: SnakeEnv) -> tuple[list[tuple[int, int]], int]:
    snake = env.snake
    for cycle in env._curriculum_cycles:
        cycle_len = len(cycle)
        for start_idx in range(cycle_len):
            if all(cycle[(start_idx + i) % cycle_len] == snake[i] for i in range(len(snake))):
                return cycle, start_idx
    raise RuntimeError(f"no Hamiltonian cycle matches snake state: {snake[:8]}")


def relative_action_toward(env: SnakeEnv, target: tuple[int, int]) -> int:
    hr, hc = env.snake_head
    tr, tc = target
    delta = (tr - hr, tc - hc)
    try:
        new_dir = next(
            direction
            for direction, direction_delta in env.DIRECTIONS.items()
            if direction_delta == delta
        )
    except StopIteration as exc:
        raise RuntimeError(
            f"target {target} is not adjacent to head {env.snake_head}"
        ) from exc

    turn = (new_dir - env.direction) % 4
    if turn == 3:
        return 0
    if turn == 0:
        return 1
    if turn == 1:
        return 2
    raise RuntimeError(
        f"Hamiltonian expert would need reverse move: dir={env.direction} target_dir={new_dir}"
    )


def expert_action(env: SnakeEnv, cycle: list[tuple[int, int]], head_idx: int) -> tuple[int, int]:
    target_idx = (head_idx - 1) % len(cycle)
    action = relative_action_toward(env, cycle[target_idx])
    return action, target_idx


def evaluate_expert(*, board_size: int, episodes: int, seed: int, verbose: bool) -> dict:
    perfect_score = board_size * board_size - 3
    scores: list[int] = []
    reasons: list[str] = []
    lengths: list[int] = []

    for ep in range(episodes):
        env = SnakeEnv(n=board_size, seed=seed + ep)
        _, _ = env.reset(seed=seed + ep)
        cycle, head_idx = find_aligned_cycle(env)
        done = False
        info = {}

        while not done:
            action, head_idx = expert_action(env, cycle, head_idx)
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        score = int(info.get("score", 0))
        reason = str(info.get("reason", "unknown"))
        length = int(info.get("length", score + 3))
        scores.append(score)
        reasons.append(reason)
        lengths.append(length)
        if verbose:
            print({"episode": ep + 1, "score": score, "reason": reason, "length": length})

    wins = sum(int(score >= perfect_score) for score in scores)
    stats = {
        "episodes": episodes,
        "wins": wins,
        "win_rate": float(wins / max(1, episodes)),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "min_score": int(min(scores)) if scores else 0,
        "max_score": int(max(scores)) if scores else 0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
    }
    stats.update(
        summarize_phase_metrics(
            scores=scores,
            terminal_lengths=lengths,
            reasons=reasons,
            perfect_score=perfect_score,
            episodes=episodes,
        )
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Hamiltonian-cycle Snake expert")
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    stats = evaluate_expert(
        board_size=args.board_size,
        episodes=args.episodes,
        seed=args.seed,
        verbose=args.verbose,
    )

    print()
    print("=" * 60)
    print("Hamiltonian Cycle Agent")
    print("=" * 60)
    print(f"  Board size:    {args.board_size}x{args.board_size}")
    print(f"  Perfect score: {args.board_size * args.board_size - 3}")
    print(f"  Episodes:      {stats['episodes']}")
    print()
    print(f"  Win rate:      {stats['win_rate']*100:.1f}% ({stats['wins']}/{stats['episodes']})")
    print(f"  Mean score:    {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
    print(f"  Median score:  {stats['median_score']:.0f}")
    print(f"  Score range:   [{stats['min_score']}, {stats['max_score']}]")
    print(f"  Mean length:   {stats['mean_length']:.1f} steps")
    print()
    print("  --- Phase Buckets ---")
    print(f"  <20%        : {stats['phase_lt20_count']:3d} ({stats['phase_lt20_rate']*100:4.1f}%)")
    print(f"  20-80%      : {stats['phase_20_80_count']:3d} ({stats['phase_20_80_rate']*100:4.1f}%)")
    print(f"  80-95%      : {stats['phase_80_95_count']:3d} ({stats['phase_80_95_rate']*100:4.1f}%)")
    print(f"  95-100%     : {stats['phase_gte95_count']:3d} ({stats['phase_gte95_rate']*100:4.1f}%)")
    print(f"  win         : {stats['win_count']:3d} ({stats['win_rate']*100:4.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()

