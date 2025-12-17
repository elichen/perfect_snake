"""Snake environment for RL training.

Observation: 9-channel tensor (n+2, n+2) with wall border
  - Ch 0: head (one-hot)
  - Ch 1: body (one-hot, includes head)
  - Ch 2: food (one-hot)
  - Ch 3-6: direction one-hot broadcast (up/right/down/left)
  - Ch 7: normalized length broadcast
  - Ch 8: walls (1 on border, 0 in playable area)

Actions: 0=turn left, 1=straight, 2=turn right (relative)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SnakeEnv(gym.Env):
    """
    Snake game environment for RL training with FULL-BOARD observation.

    Observation (float32): 9-channel grid ((n+2) x (n+2)) with wall border.

    Action space: Discrete(3)
        - 0: Turn left (relative)
        - 1: Go straight
        - 2: Turn right
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Direction constants: 0=up, 1=right, 2=down, 3=left
    DIRECTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),  # right
        2: (1, 0),  # down
        3: (0, -1),  # left
    }

    def __init__(
        self,
        n: int = 20,
        max_no_food: Optional[int] = None,
        render_mode: Optional[str] = None,
        gamma: float = 0.995,
        alpha: float = 0.2,
        survival_bonus: float = 0.0,
        random_offset: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.n = n
        self.max_no_food_base = max_no_food
        self.render_mode = render_mode
        self.gamma = gamma
        self.alpha = alpha
        self.survival_bonus = survival_bonus
        self.random_offset = random_offset  # Placeholder for API compatibility

        # Action space: turn left, straight, turn right
        self.action_space = spaces.Discrete(3)

        # Observation space: 9 channels, (n+2) x (n+2) with a wall border
        self.n_channels = 9
        self.obs_n = self.n + 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_channels, self.obs_n, self.obs_n),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)
        self._walls = np.zeros((self.obs_n, self.obs_n), dtype=np.float32)
        self._walls[0, :] = 1.0
        self._walls[-1, :] = 1.0
        self._walls[:, 0] = 1.0
        self._walls[:, -1] = 1.0

        # Game state (initialized in reset)
        self.snake: list[Tuple[int, int]] = []
        self.direction: int = 0
        self.food_pos: Tuple[int, int] = (0, 0)
        self.steps_since_food: int = 0
        self.score: int = 0
        self.prev_phi: float = 0.0
        self.total_steps: int = 0

    @property
    def snake_head(self) -> Tuple[int, int]:
        return self.snake[0]

    @property
    def snake_length(self) -> int:
        return len(self.snake)

    @property
    def max_no_food(self) -> int:
        if self.max_no_food_base is not None:
            return self.max_no_food_base
        return max(80 + 4 * self.snake_length, 2 * self.n * self.n)

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _compute_phi(self) -> float:
        d = self._manhattan_distance(self.snake_head, self.food_pos)
        max_d = 2 * (self.n - 1)
        d_norm = d / max_d if max_d > 0 else 0.0
        return -self.alpha * d_norm

    def _place_food(self) -> None:
        snake_set = set(self.snake)
        empty_cells = [
            (r, c)
            for r in range(self.n)
            for c in range(self.n)
            if (r, c) not in snake_set
        ]
        if empty_cells:
            idx = self.rng.integers(len(empty_cells))
            self.food_pos = empty_cells[idx]
        else:
            # Grid is full (game won)
            self.food_pos = (-1, -1)

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros((self.n_channels, self.obs_n, self.obs_n), dtype=np.float32)

        # Channel 0: Head
        hr, hc = self.snake_head
        obs[0, hr + 1, hc + 1] = 1.0

        # Channel 1: Body (includes head, matching web agent)
        for r, c in self.snake:
            obs[1, r + 1, c + 1] = 1.0

        # Channel 2: Food
        fr, fc = self.food_pos
        if fr >= 0:
            obs[2, fr + 1, fc + 1] = 1.0

        # Channels 3-6: Direction one-hot (broadcast)
        dir_channel = 3 + int(self.direction)
        obs[dir_channel, :, :] = 1.0

        # Channel 7: Normalized length (broadcast)
        obs[7, :, :] = self.snake_length / float(self.n * self.n)

        # Channel 8: Walls (constant padded border)
        obs[8, :, :] = self._walls

        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        center = self.n // 2

        # Random initial direction
        self.direction = int(self.rng.integers(4))

        # Build snake aligned with direction
        dr, dc = self.DIRECTIONS[self.direction]
        self.snake = []
        for i in range(3):
            r = center - i * dr
            c = center - i * dc
            r = max(0, min(self.n - 1, r))
            c = max(0, min(self.n - 1, c))
            self.snake.append((r, c))

        self._place_food()

        self.steps_since_food = 0
        self.score = 0
        self.total_steps = 0
        self.prev_phi = self._compute_phi()

        obs = self._get_observation()
        info = {
            "length": self.snake_length,
            "score": self.score,
            "food_pos": self.food_pos,
        }

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.total_steps += 1

        # Map relative action to absolute direction
        delta = {0: -1, 1: 0, 2: 1}
        new_dir = (self.direction + delta[int(action)]) % 4
        self.direction = new_dir

        dr, dc = self.DIRECTIONS[new_dir]
        hr, hc = self.snake_head
        new_head = (hr + dr, hc + dc)

        terminated = False
        truncated = False
        base_reward = 0.0
        reason = None

        # Check wall collision
        if not (0 <= new_head[0] < self.n and 0 <= new_head[1] < self.n):
            terminated = True
            base_reward = -1.0
            reason = "wall"
        # Check self collision (excluding tail if it will move)
        elif new_head in self.snake[:-1]:
            terminated = True
            base_reward = -1.0
            reason = "self"
        # Check if tail stays (only if eating food, tail won't move)
        elif new_head == self.snake[-1] and new_head != self.food_pos:
            pass
        elif new_head in self.snake:
            terminated = True
            base_reward = -1.0
            reason = "self"

        if not terminated:
            if new_head == self.food_pos:
                self.snake.insert(0, new_head)
                self.score += 1
                self.steps_since_food = 0
                base_reward = 1.0

                if self.snake_length >= self.n * self.n:
                    terminated = True
                    reason = "win"
                else:
                    self._place_food()
            else:
                self.snake.insert(0, new_head)
                self.snake.pop()
                self.steps_since_food += 1

        if not terminated and self.steps_since_food > self.max_no_food:
            truncated = True
            base_reward += -0.5
            reason = "stall"

        if not terminated:
            phi = self._compute_phi()
            r_shape = self.gamma * phi - self.prev_phi
            self.prev_phi = phi
        else:
            r_shape = 0.0

        if terminated and reason != "win":
            reward = base_reward
        else:
            reward = base_reward + r_shape + self.survival_bonus

        obs = self._get_observation()
        info = {
            "length": self.snake_length,
            "score": self.score,
            "reason": reason,
            "steps": self.total_steps,
            "food_pos": self.food_pos,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            self._render_ascii()
            return None
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    def _render_ascii(self) -> None:
        snake_set = set(self.snake)
        head = self.snake_head

        print(f"\nScore: {self.score}  Length: {self.snake_length}  Steps: {self.total_steps}")
        print("+" + "-" * self.n + "+")

        for r in range(self.n):
            row = "|"
            for c in range(self.n):
                pos = (r, c)
                if pos == head:
                    row += "O"
                elif pos in snake_set:
                    row += "#"
                elif pos == self.food_pos:
                    row += "*"
                else:
                    row += " "
            row += "|"
            print(row)

        print("+" + "-" * self.n + "+")

    def _render_rgb(self) -> np.ndarray:
        cell_size = 20
        img_size = self.n * cell_size
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        img[:, :] = [40, 40, 40]

        for i in range(self.n + 1):
            pos = i * cell_size
            img[pos : pos + 1, :] = [60, 60, 60]
            img[:, pos : pos + 1] = [60, 60, 60]

        fr, fc = self.food_pos
        if fr >= 0:
            r1, r2 = fr * cell_size + 2, (fr + 1) * cell_size - 2
            c1, c2 = fc * cell_size + 2, (fc + 1) * cell_size - 2
            img[r1:r2, c1:c2] = [255, 50, 50]

        for i, (r, c) in enumerate(self.snake):
            r1, r2 = r * cell_size + 1, (r + 1) * cell_size - 1
            c1, c2 = c * cell_size + 1, (c + 1) * cell_size - 1
            if i == 0:
                img[r1:r2, c1:c2] = [50, 255, 50]
            else:
                img[r1:r2, c1:c2] = [30, 180, 30]

        return img

    def close(self) -> None:
        pass
