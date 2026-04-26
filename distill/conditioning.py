"""Observation conditioning helpers for expert-distillation experiments."""

from __future__ import annotations

from typing import Optional

import numpy as np

from snake_env import SnakeEnv


def find_cycle_condition(env: SnakeEnv) -> tuple[Optional[int], Optional[int]]:
    snake = env.snake
    for cycle_idx, cycle in enumerate(env._curriculum_cycles):
        cycle_len = len(cycle)
        for head_idx in range(cycle_len):
            if all(cycle[(head_idx + i) % cycle_len] == snake[i] for i in range(len(snake))):
                return cycle_idx, head_idx
    return None, None


def conditioning_channels(env: SnakeEnv) -> int:
    return 2 + 4 + len(env._curriculum_cycles)


def augment_observation(obs: np.ndarray, env: SnakeEnv, cycle_idx: Optional[int]) -> np.ndarray:
    h, w = obs.shape[1:]
    extra = np.zeros((conditioning_channels(env), h, w), dtype=np.float32)

    # Absolute head position, broadcast.
    hr, hc = env.snake_head
    extra[0, :, :] = hr / float(max(1, env.n - 1))
    extra[1, :, :] = hc / float(max(1, env.n - 1))

    # Absolute direction one-hot, broadcast.
    extra[2 + env.direction, :, :] = 1.0

    # Deterministic cycle identity one-hot, broadcast.
    if cycle_idx is not None:
        extra[6 + cycle_idx, :, :] = 1.0

    return np.concatenate([obs, extra], axis=0)
