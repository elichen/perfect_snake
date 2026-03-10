"""Snake environment for RL training.

Observation: 5-channel tensor (n+2, n+2), snake-centric (egocentric)
  Grid is rotated so snake always faces "up" (forward).
  - Ch 0: head (one-hot)
  - Ch 1: body (one-hot, includes head)
  - Ch 2: food (one-hot)
  - Ch 3: normalized length broadcast
  - Ch 4: walls (1 on border, 0 in playable area)

Actions: 0=turn left, 1=straight, 2=turn right (relative)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SnakeEnv(gym.Env):
    """
    Snake game environment for RL training.

    Observation (float32): 5-channel (n+2) x (n+2), snake-centric (egocentric).
    Grid is rotated so snake always faces "up".

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
        seed: Optional[int] = None,
        stall_penalty: float = -1.0,
        stall_terminates: bool = True,
        max_no_food_base: Optional[int] = None,
        flood_fill_obs: bool = False,
        curriculum_prob: float = 0.0,
        curriculum_min_fill: float = 0.5,
        curriculum_max_fill: float = 0.85,
        head_centered: bool = False,
    ):
        super().__init__()

        self.n = n
        # Support both old 'max_no_food' and new 'max_no_food_base' parameter names
        self._max_no_food_override = max_no_food_base if max_no_food_base is not None else max_no_food
        self.render_mode = render_mode
        self.gamma = gamma
        self.alpha = alpha
        self.survival_bonus = survival_bonus
        self.stall_penalty = stall_penalty
        self.stall_terminates = stall_terminates
        self.flood_fill_obs = flood_fill_obs
        self.curriculum_prob = curriculum_prob
        self.curriculum_min_fill = curriculum_min_fill
        self.curriculum_max_fill = curriculum_max_fill
        self.head_centered = head_centered

        # Action space: turn left, straight, turn right
        self.action_space = spaces.Discrete(3)

        # Observation space: 5 or 6 channels (egocentric)
        self.n_channels = 6 if flood_fill_obs else 5
        if head_centered:
            self.obs_n = 2 * (n - 1) + 1  # 39 for n=20
        else:
            self.obs_n = self.n + 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_channels, self.obs_n, self.obs_n),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)
        self._walls = np.zeros((self.obs_n, self.obs_n), dtype=np.float32)
        if not head_centered:
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
        if self._max_no_food_override is not None:
            return self._max_no_food_override
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

    def _flood_fill(self) -> np.ndarray:
        """Flood-fill reachability from head using scipy connected components."""
        from scipy.ndimage import label

        n = self.n
        # Build passable grid (1 = empty, 0 = body)
        passable = np.ones((n, n), dtype=np.int32)
        for r, c in self.snake:
            passable[r, c] = 0

        # Label connected components
        labels, _ = label(passable)

        # Find which components are reachable from head (head is on body, check neighbors)
        hr, hc = self.snake_head
        head_labels = set()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = hr + dr, hc + dc
            if 0 <= nr < n and 0 <= nc < n and labels[nr, nc] > 0:
                head_labels.add(labels[nr, nc])

        if not head_labels:
            return np.zeros((n, n), dtype=np.float32)

        reachable = np.zeros((n, n), dtype=np.float32)
        for lbl in head_labels:
            reachable[labels == lbl] = 1.0
        reachable[hr, hc] = 1.0  # Mark head position too
        return reachable

    def _get_observation(self) -> np.ndarray:
        """Snake-centric observation (5 or 6 channels, rotated so snake faces 'up')."""
        if self.head_centered:
            return self._get_observation_head_centered()

        obs = np.zeros((self.n_channels, self.obs_n, self.obs_n), dtype=np.float32)

        # Channel 0: Head
        hr, hc = self.snake_head
        obs[0, hr + 1, hc + 1] = 1.0

        # Channel 1: Body (includes head)
        for r, c in self.snake:
            obs[1, r + 1, c + 1] = 1.0

        # Channel 2: Food
        fr, fc = self.food_pos
        if fr >= 0:
            obs[2, fr + 1, fc + 1] = 1.0

        # Channel 3: Normalized length (broadcast)
        obs[3, :, :] = self.snake_length / float(self.n * self.n)

        # Channel 4: Walls
        obs[4, :, :] = self._walls

        # Channel 5: Flood-fill reachability from head
        if self.flood_fill_obs:
            obs[5, 1:-1, 1:-1] = self._flood_fill()

        # Rotate so snake always faces "up" (direction 0)
        if self.direction != 0:
            obs = np.rot90(obs, k=self.direction, axes=(1, 2)).copy()

        return obs

    def _get_observation_head_centered(self) -> np.ndarray:
        """Head-centered observation: head always at grid center, 39x39 for n=20."""
        obs = np.zeros((self.n_channels, self.obs_n, self.obs_n), dtype=np.float32)
        hr, hc = self.snake_head
        c = self.obs_n // 2  # center index (19 for 39x39)

        # Channel 0: Head (always at center)
        obs[0, c, c] = 1.0

        # Channel 1: Body (includes head)
        for r, col in self.snake:
            obs[1, r - hr + c, col - hc + c] = 1.0

        # Channel 2: Food
        fr, fc = self.food_pos
        if fr >= 0:
            obs[2, fr - hr + c, fc - hc + c] = 1.0

        # Channel 3: Normalized length (broadcast)
        obs[3, :, :] = self.snake_length / float(self.n * self.n)

        # Channel 4: Walls (everything outside the board)
        row_board = np.arange(self.obs_n) + hr - c
        col_board = np.arange(self.obs_n) + hc - c
        obs[4] = ((row_board < 0) | (row_board >= self.n))[:, None] | \
                  ((col_board < 0) | (col_board >= self.n))[None, :]

        # Channel 5: Flood-fill reachability
        if self.flood_fill_obs:
            ff = self._flood_fill()
            # Compute overlap between grid and board
            gr_start = max(0, c - hr)
            gc_start = max(0, c - hc)
            br_start = max(0, hr - c)
            bc_start = max(0, hc - c)
            h = min(self.n - br_start, self.obs_n - gr_start)
            w = min(self.n - bc_start, self.obs_n - gc_start)
            obs[5, gr_start:gr_start+h, gc_start:gc_start+w] = \
                ff[br_start:br_start+h, bc_start:bc_start+w]

        # Rotate so snake always faces "up"
        if self.direction != 0:
            obs = np.rot90(obs, k=self.direction, axes=(1, 2)).copy()

        return obs

    def _build_hamiltonian_path(self) -> list:
        """Build a zigzag Hamiltonian path covering the entire board."""
        path = []
        for r in range(self.n):
            if r % 2 == 0:
                for c in range(self.n):
                    path.append((r, c))
            else:
                for c in range(self.n - 1, -1, -1):
                    path.append((r, c))
        return path

    def _reset_with_fill(self) -> None:
        """Curriculum reset: place snake along a zigzag path at random fill level."""
        board_area = self.n * self.n
        min_len = max(3, int(self.curriculum_min_fill * board_area))
        max_len = min(board_area - 1, int(self.curriculum_max_fill * board_area))
        target_len = int(self.rng.integers(min_len, max_len + 1))

        # Build a zigzag path and pick a random starting offset
        path = self._build_hamiltonian_path()
        max_start = len(path) - target_len
        start_idx = int(self.rng.integers(0, max(1, max_start + 1)))

        # Snake: head is at start_idx, body follows along the path
        self.snake = path[start_idx : start_idx + target_len]

        # Set direction based on head → first body segment
        hr, hc = self.snake[0]
        br, bc = self.snake[1]
        dr, dc = hr - br, hc - bc
        for d, (ddr, ddc) in self.DIRECTIONS.items():
            if (ddr, ddc) == (dr, dc):
                self.direction = d
                break
        else:
            self.direction = int(self.rng.integers(4))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.curriculum_prob > 0 and self.rng.random() < self.curriculum_prob:
            self._reset_with_fill()
        else:
            center = self.n // 2
            self.direction = int(self.rng.integers(4))
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
        self.score = max(0, self.snake_length - 3)
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
            if self.stall_terminates:
                terminated = True  # Proper termination - PPO won't bootstrap
            else:
                truncated = True  # Old behavior - PPO bootstraps (underpenalizes)
            base_reward += self.stall_penalty
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
