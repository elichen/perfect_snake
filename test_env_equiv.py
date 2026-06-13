"""Bitwise equivalence test: optimized snake_env vs the pre-optimization reference.

Reference is /tmp/snake_env_ref.py (a copy of snake_env.py made before the
occupancy-grid/flood-fill optimizations). Both envs are driven with identical
seeds and identical action sequences; every step must produce identical obs
(exact array equality), reward (exact float equality), terminated/truncated,
and info.
"""
import importlib.util
import sys

import numpy as np

from snake_env import SnakeEnv as NewEnv

spec = importlib.util.spec_from_file_location("snake_env_ref", "/tmp/snake_env_ref.py")
ref_mod = importlib.util.module_from_spec(spec)
sys.modules["snake_env_ref"] = ref_mod
spec.loader.exec_module(ref_mod)
RefEnv = ref_mod.SnakeEnv

DIRECTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
REL = {0: -1, 1: 0, 2: 1}


def safe_random_action(env, rng):
    hr, hc = env.snake_head
    body = set(env.snake[:-1])
    candidates = []
    for rel in (0, 1, 2):
        d = (env.direction + REL[rel]) % 4
        dr, dc = DIRECTIONS[d]
        nh = (hr + dr, hc + dc)
        if 0 <= nh[0] < env.n and 0 <= nh[1] < env.n and nh not in body:
            candidates.append(rel)
    if not candidates:
        return 1
    return int(rng.choice(candidates))


def cycle_follow_action(env, rng, deviate_p=0.05):
    if env._curriculum_cycle is not None and env._curriculum_head_idx is not None \
            and rng.random() >= deviate_p:
        cycle = env._curriculum_cycle
        target = cycle[(env._curriculum_head_idx - 1) % len(cycle)]
        hr, hc = env.snake_head
        want = (target[0] - hr, target[1] - hc)
        for rel in (0, 1, 2):
            d = (env.direction + REL[rel]) % 4
            if DIRECTIONS[d] == want:
                return rel
    return safe_random_action(env, rng)


def run_config(name, kwargs, steps, policy, seed0=7000):
    ref = RefEnv(**kwargs)
    new = NewEnv(**kwargs)
    rng = np.random.default_rng(123)
    seed = seed0
    o_ref, i_ref = ref.reset(seed=seed)
    o_new, i_new = new.reset(seed=seed)
    assert np.array_equal(o_ref, o_new), f"{name}: reset obs mismatch"
    assert i_ref == i_new, f"{name}: reset info mismatch"
    n_term = n_eat = n_win = n_stall = n_pen = 0
    for t in range(steps):
        a = policy(ref, rng)
        ro, rr, rt, rtr, ri = ref.step(a)
        no, nr, nt, ntr, ni = new.step(a)
        assert np.array_equal(ro, no), f"{name}: obs mismatch at step {t}"
        assert rr == nr, f"{name}: reward mismatch at step {t}: {rr} vs {nr}"
        assert rt == nt and rtr == ntr, f"{name}: term/trunc mismatch at step {t}"
        assert ri == ni, f"{name}: info mismatch at step {t}: {ri} vs {ni}"
        if rr >= 1.0:
            n_eat += 1
        if ri.get("reason") == "win":
            n_win += 1
        if ri.get("reason") == "stall":
            n_stall += 1
        if rr < -0.001 and not rt and not rtr:
            n_pen += 1
        if rt or rtr:
            n_term += 1
            seed += 1
            o_ref, _ = ref.reset(seed=seed)
            o_new, _ = new.reset(seed=seed)
            assert np.array_equal(o_ref, o_new), f"{name}: reset obs mismatch (seed {seed})"
    print(f"OK {name}: {steps} steps, {n_term} episodes, {n_eat} eats, "
          f"{n_win} wins, {n_stall} stalls, {n_pen} shaped-neg steps")


def main():
    run_config(
        "sweep20 (flood+headcentered+curr)",
        dict(n=20, flood_fill_obs=True, head_centered=True, curriculum_prob=0.3,
             gamma=0.999, alpha=0.2),
        8000, safe_random_action)
    run_config(
        "penalties (topo+tailsafety low fill)",
        dict(n=20, flood_fill_obs=True, head_centered=True, curriculum_prob=0.5,
             topology_penalty=-0.15, topology_penalty_min_fill=0.3,
             tail_safety_penalty=-0.15, tail_safety_min_fill=0.3),
        4000, lambda e, r: cycle_follow_action(e, r, deviate_p=0.10))
    run_config(
        "plain (5ch egocentric)",
        dict(n=20),
        6000, safe_random_action)
    run_config(
        "history (obs_history=2, action_history=1, flood)",
        dict(n=20, flood_fill_obs=True, obs_history=2, action_history_obs=1,
             curriculum_prob=0.3),
        5000, safe_random_action)
    run_config(
        "safe_action_bonus (snapshot/restore path)",
        dict(n=10, flood_fill_obs=True, safe_action_bonus=0.01,
             safe_action_bonus_min_fill=0.0, curriculum_prob=0.3),
        1500, safe_random_action)
    run_config(
        "longsnake 10x10 cycle-follow (eats/wins/stalls)",
        dict(n=10, flood_fill_obs=True, head_centered=True, curriculum_prob=1.0,
             gamma=0.999),
        15000, lambda e, r: cycle_follow_action(e, r, deviate_p=0.02))
    run_config(
        "longsnake 20x20 cycle-follow",
        dict(n=20, flood_fill_obs=True, head_centered=True, curriculum_prob=1.0,
             gamma=0.999),
        12000, lambda e, r: cycle_follow_action(e, r, deviate_p=0.05))
    print("ALL CONFIGS BITWISE-EQUIVALENT")


if __name__ == "__main__":
    main()
