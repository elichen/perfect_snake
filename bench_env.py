"""Profile snake_env step+obs cost under the sweep config.

Usage: .venv/bin/python bench_env.py [--steps 30000] [--profile]
"""
import argparse
import cProfile
import io
import pstats
import time

import numpy as np

from snake_env import SnakeEnv

DIRECTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


def make_env(seed: int, curriculum_prob: float = 0.3) -> SnakeEnv:
    env = SnakeEnv(
        n=20,
        flood_fill_obs=True,
        head_centered=True,
        curriculum_prob=curriculum_prob,
    )
    env.reset(seed=seed)
    return env


def cycle_follow_action(env: SnakeEnv) -> int:
    """Relative action that follows the curriculum cycle (keeps long snakes alive)."""
    if env._curriculum_cycle is None or env._curriculum_head_idx is None:
        return -1
    cycle = env._curriculum_cycle
    target = cycle[(env._curriculum_head_idx - 1) % len(cycle)]
    hr, hc = env.snake_head
    want = (target[0] - hr, target[1] - hc)
    for rel in (0, 1, 2):
        d = (env.direction + {0: -1, 1: 0, 2: 1}[rel]) % 4
        if DIRECTIONS[d] == want:
            return rel
    return -1


def safe_random_action(env: SnakeEnv, rng: np.random.Generator) -> int:
    """Pick a random relative action that doesn't immediately hit wall/body."""
    hr, hc = env.snake_head
    body = set(env.snake[:-1])
    candidates = []
    for rel in (0, 1, 2):
        d = (env.direction + {0: -1, 1: 0, 2: 1}[rel]) % 4
        dr, dc = DIRECTIONS[d]
        nh = (hr + dr, hc + dc)
        if 0 <= nh[0] < env.n and 0 <= nh[1] < env.n and nh not in body:
            candidates.append(rel)
    if not candidates:
        return 1
    return int(rng.choice(candidates))


def run(steps: int, seed: int = 0, long_snake: bool = False) -> dict:
    env = make_env(seed, curriculum_prob=1.0 if long_snake else 0.3)
    rng = np.random.default_rng(seed)
    lengths = []
    t0 = time.perf_counter()
    for _ in range(steps):
        a = cycle_follow_action(env) if long_snake else -1
        if a < 0:
            a = safe_random_action(env, rng)
        obs, r, term, trunc, info = env.step(a)
        lengths.append(env.snake_length)
        if term or trunc:
            env.reset()
    dt = time.perf_counter() - t0
    return {
        "steps": steps,
        "seconds": dt,
        "sps": steps / dt,
        "mean_len": float(np.mean(lengths)),
        "p90_len": float(np.percentile(lengths, 90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--long-snake", action="store_true",
                    help="curriculum_prob=1.0 + cycle-following: realistic long-snake states")
    args = ap.parse_args()

    # Warmup (imports, scipy lazy load)
    run(2000, seed=1, long_snake=args.long_snake)

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        stats = run(args.steps, seed=0, long_snake=args.long_snake)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(25)
        print(s.getvalue())
    else:
        stats = run(args.steps, seed=0, long_snake=args.long_snake)
    print({k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()})


if __name__ == "__main__":
    main()
