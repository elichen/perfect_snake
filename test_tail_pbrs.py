"""Tests for the tail-safety PBRS mechanism and the ff-based tail-reachability.

1. Fuzz: _tail_reachable (flood-fill-derived) == _tail_reachable_bfs (reference BFS)
   across thousands of random reachable/unreachable states.
2. PBRS: drive a pbrs-on env and a pbrs-off shadow env with identical seeds/actions;
   the reward difference at every surviving step must equal gamma*phi(s') - phi(s)
   computed from the shadow env's own tail-reachability, and obs/termination must
   be identical (shaping must not leak into observations).
"""
import numpy as np

from snake_env import SnakeEnv

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


def fuzz_tail_reachable(n_states=4000):
    env = SnakeEnv(n=10, flood_fill_obs=True, head_centered=True, curriculum_prob=0.7)
    rng = np.random.default_rng(0)
    seed = 31000
    env.reset(seed=seed)
    checked = unreachable = 0
    for t in range(n_states):
        a = safe_random_action(env, rng)
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            seed += 1
            env.reset(seed=seed)
            continue
        fast = env._tail_reachable()
        ref = env._tail_reachable_bfs()
        assert fast == ref, f"tail_reachable mismatch at t={t}: fast={fast} ref={ref}\nsnake={env.snake}"
        checked += 1
        unreachable += int(not ref)
    assert unreachable > 0, "fuzz never produced an unreachable-tail state; weak test"
    print(f"OK tail_reachable fuzz: {checked} states, {unreachable} unreachable, all match BFS")


def check_pbrs(coef=-0.3, min_fill=0.0, steps=6000):
    kwargs = dict(n=10, flood_fill_obs=True, head_centered=True, curriculum_prob=0.7,
                  gamma=0.999)
    on = SnakeEnv(tail_safety_pbrs=coef, tail_safety_pbrs_min_fill=min_fill, **kwargs)
    off = SnakeEnv(**kwargs)
    rng = np.random.default_rng(1)
    seed = 32000
    o_on, _ = on.reset(seed=seed)
    o_off, _ = off.reset(seed=seed)
    assert np.array_equal(o_on, o_off)

    def phi(env):
        fill = env.snake_length / float(env.n * env.n)
        if fill >= min_fill and not env._tail_reachable_bfs():
            return coef
        return 0.0

    prev_phi = phi(off)
    n_nonzero = 0
    for t in range(steps):
        a = safe_random_action(off, rng)
        oo, ro, to, tro, io_ = on.step(a)
        of, rf, tf, trf, if_ = off.step(a)
        assert np.array_equal(oo, of), f"pbrs leaked into obs at t={t}"
        assert (to, tro, io_) == (tf, trf, if_), f"pbrs changed termination at t={t}"
        if to or tro:
            if io_.get("reason") == "win":
                expected = 0.0  # shaped terms are zeroed on terminal transitions
                assert abs((ro - rf) - expected) < 1e-12, f"win-step mismatch at t={t}"
            else:
                assert ro == rf, f"death reward must be unshaped at t={t}"
            seed += 1
            o_on, _ = on.reset(seed=seed)
            o_off, _ = off.reset(seed=seed)
            assert np.array_equal(o_on, o_off)
            prev_phi = phi(off)
            continue
        cur_phi = phi(off)
        expected = kwargs["gamma"] * cur_phi - prev_phi
        diff = ro - rf
        assert abs(diff - expected) < 1e-9, \
            f"pbrs term mismatch at t={t}: got {diff}, expected {expected}"
        if expected != 0.0:
            n_nonzero += 1
        prev_phi = cur_phi
    assert n_nonzero > 0, "pbrs never fired; weak test"
    print(f"OK pbrs: {steps} steps, {n_nonzero} nonzero shaped terms, all match "
          f"gamma*phi(s')-phi(s) from shadow env")


if __name__ == "__main__":
    fuzz_tail_reachable()
    check_pbrs()
    print("ALL PBRS TESTS PASS")
