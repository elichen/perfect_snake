"""Snake environment for RL training.

Observation: 5+ channel tensor (n+2, n+2), snake-centric (egocentric)
  Grid is rotated so snake always faces "up" (forward).
  - Ch 0: head (one-hot)
  - Ch 1: body (one-hot, includes head)
  - Ch 2: food (one-hot)
  - Ch 3: normalized length broadcast
  - Ch 4: walls (1 on border, 0 in playable area)

Actions: 0=turn left, 1=straight, 2=turn right (relative)
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SnakeEnv(gym.Env):
    """
    Snake game environment for RL training.

    Observation (float32): 5+ channel (n+2) x (n+2), snake-centric (egocentric).
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
        topology_penalty: float = 0.0,
        topology_penalty_min_fill: float = 0.80,
        tail_safety_penalty: float = 0.0,
        tail_safety_min_fill: float = 0.80,
        tail_safety_pbrs: float = 0.0,
        tail_safety_pbrs_min_fill: float = 0.80,
        seed: Optional[int] = None,
        stall_penalty: float = -1.0,
        stall_terminates: bool = True,
        max_no_food_base: Optional[int] = None,
        flood_fill_obs: bool = False,
        body_age_obs: bool = False,
        obs_history: int = 1,
        action_history_obs: int = 0,
        curriculum_prob: float = 0.0,
        curriculum_min_fill: float = 0.5,
        curriculum_max_fill: float = 0.85,
        curriculum_follow_bonus: float = 0.0,
        curriculum_follow_min_fill: float = 0.85,
        cycle_target_obs: bool = False,
        tail_target_obs: bool = False,
        safe_action_target_obs: bool = False,
        safe_action_soft_target_obs: bool = False,
        body_age_target_obs: bool = False,
        body_age_obs_min_fill: float = 0.90,
        cycle_target_min_fill: Optional[float] = None,
        safe_action_target_min_fill: float = 0.90,
        safe_action_soft_target_min_fill: float = 0.90,
        body_age_target_min_fill: float = 0.80,
        safe_action_fill_weight: float = 500.0,
        safe_action_soft_temperature: float = 1.0,
        safe_action_bonus: float = 0.0,
        safe_action_bonus_min_fill: float = 0.95,
        safe_action_bonus_fill_weight: float = 500.0,
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
        self.topology_penalty = topology_penalty
        self.topology_penalty_min_fill = topology_penalty_min_fill
        self.tail_safety_penalty = tail_safety_penalty
        self.tail_safety_min_fill = tail_safety_min_fill
        self.tail_safety_pbrs = tail_safety_pbrs
        self.tail_safety_pbrs_min_fill = tail_safety_pbrs_min_fill
        self.stall_penalty = stall_penalty
        self.stall_terminates = stall_terminates
        self.flood_fill_obs = flood_fill_obs
        self.body_age_obs = body_age_obs
        self.obs_history = max(1, int(obs_history))
        self.action_history_obs = max(0, int(action_history_obs))
        self.curriculum_prob = curriculum_prob
        self.curriculum_min_fill = curriculum_min_fill
        self.curriculum_max_fill = curriculum_max_fill
        self.curriculum_follow_bonus = curriculum_follow_bonus
        self.curriculum_follow_min_fill = curriculum_follow_min_fill
        self.cycle_target_obs = cycle_target_obs
        self.tail_target_obs = tail_target_obs
        self.safe_action_target_obs = safe_action_target_obs
        self.safe_action_soft_target_obs = safe_action_soft_target_obs
        self.body_age_target_obs = body_age_target_obs
        self.body_age_obs_min_fill = body_age_obs_min_fill
        self.cycle_target_min_fill = (
            curriculum_follow_min_fill
            if cycle_target_min_fill is None
            else cycle_target_min_fill
        )
        self.safe_action_target_min_fill = safe_action_target_min_fill
        self.safe_action_soft_target_min_fill = safe_action_soft_target_min_fill
        self.body_age_target_min_fill = body_age_target_min_fill
        self.safe_action_fill_weight = safe_action_fill_weight
        self.safe_action_soft_temperature = safe_action_soft_temperature
        self.safe_action_bonus = safe_action_bonus
        self.safe_action_bonus_min_fill = safe_action_bonus_min_fill
        self.safe_action_bonus_fill_weight = safe_action_bonus_fill_weight
        self.head_centered = head_centered

        # Action space: turn left, straight, turn right
        self.action_space = spaces.Discrete(3)

        # Observation space:
        # encoder-visible channels are current-frame-first, then prior frames,
        # followed by single-frame auxiliary training targets.
        self.action_history_channels = 3 * self.action_history_obs
        self.base_obs_channels = 5 + int(flood_fill_obs) + int(body_age_obs) + self.action_history_channels
        self.encoder_visible_channels = self.base_obs_channels * self.obs_history
        self.flood_fill_channel = 5 if flood_fill_obs else None
        self.body_age_obs_channel = 5 + int(flood_fill_obs) if body_age_obs else None
        next_channel = self.base_obs_channels
        self.cycle_target_channel = next_channel if cycle_target_obs else None
        next_channel += int(cycle_target_obs)
        self.tail_target_channel = next_channel if tail_target_obs else None
        next_channel += int(tail_target_obs)
        self.safe_action_target_channel = next_channel if safe_action_target_obs else None
        next_channel += int(safe_action_target_obs)
        self.safe_action_soft_target_channel = next_channel if safe_action_soft_target_obs else None
        next_channel += 3 * int(safe_action_soft_target_obs)
        self.body_age_target_channel = next_channel if body_age_target_obs else None
        next_channel += int(body_age_target_obs)
        self.aux_target_channels = next_channel - self.base_obs_channels
        self.single_frame_n_channels = next_channel
        self.n_channels = self.encoder_visible_channels + self.aux_target_channels
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
        self._obs_history_frames: Deque[np.ndarray] = deque(maxlen=max(0, self.obs_history - 1))
        self._history_zero_frame: np.ndarray | None = None
        self._action_history: Deque[int] = deque(maxlen=self.action_history_obs)
        self._walls = np.zeros((self.obs_n, self.obs_n), dtype=np.float32)
        if not head_centered:
            self._walls[0, :] = 1.0
            self._walls[-1, :] = 1.0
            self._walls[:, 0] = 1.0
            self._walls[:, -1] = 1.0

        # Body occupancy grid (includes head), maintained incrementally by step()
        # and rebuilt on reset/restore. Single source for collision checks, the body
        # obs channel, food placement, and the flood-fill passable mask.
        self._occ = np.zeros((n, n), dtype=bool)
        # Precomputed 4-connectivity structure + label output buffer for flood fill
        # (avoids scipy generate_binary_structure + an allocation per call).
        self._ff_structure = np.array(
            [[False, True, False], [True, True, True], [False, True, False]]
        )
        self._ff_labels = np.zeros((n, n), dtype=np.int32)
        # Per-state flood-fill cache: obs, penalties, and tail-safety all need the
        # same reachability map. Invalidated at step() entry and on reset/restore.
        self._ff_cache: Optional[np.ndarray] = None

        # Game state (initialized in reset)
        self._snake: list[Tuple[int, int]] = []
        self.direction: int = 0
        self.food_pos: Tuple[int, int] = (0, 0)
        self.steps_since_food: int = 0
        self.score: int = 0
        self.prev_phi: float = 0.0
        self.prev_tail_phi: float = 0.0
        self.total_steps: int = 0
        self._curriculum_cycles = self._build_curriculum_cycles()
        self._curriculum_cycle: Optional[list[Tuple[int, int]]] = None
        self._curriculum_head_idx: Optional[int] = None

    @property
    def snake(self) -> list[Tuple[int, int]]:
        return self._snake

    @snake.setter
    def snake(self, value: list[Tuple[int, int]]) -> None:
        # Rebuild the occupancy grid on assignment so external scripts that set
        # env.snake directly stay consistent. In-place mutation (insert/pop/append)
        # bypasses this; step() maintains occ itself and reset() rebuilds explicitly.
        self._snake = list(value)
        self._rebuild_occ()
        self._ff_cache = None

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

    def _rebuild_occ(self) -> None:
        self._occ[:] = False
        for r, c in self.snake:
            self._occ[r, c] = True

    def _place_food(self) -> None:
        # Row-major order over empty cells matches the original list comprehension,
        # so the same RNG draw selects the same cell.
        empty_flat = np.flatnonzero(~self._occ.ravel())
        if empty_flat.size:
            idx = self.rng.integers(empty_flat.size)
            flat = int(empty_flat[idx])
            self.food_pos = (flat // self.n, flat % self.n)
        else:
            # Grid is full (game won)
            self.food_pos = (-1, -1)

    def _tail_reachable(self) -> bool:
        """Can the head reach its own tail through free cells (tail treated as a
        passable goal, since it vacates as the snake moves)?

        If yes, the snake can always survive by following its tail — the core
        space-filling viability invariant. This is a binary, survival-relevant signal,
        unlike a stranded-cell count, so penalizing its loss does not push the policy
        toward blunt boundary-avoidance.
        """
        if len(self.snake) < 2:
            return True
        hr, hc = self.snake[0]
        tr, tc = self.snake[-1]
        if abs(tr - hr) + abs(tc - hc) == 1:
            return True
        # Tail is reachable iff a free neighbor of the tail sits in the head's
        # flood-fill component (the cached map the obs computes anyway).
        ff = self._flood_fill()
        n = self.n
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = tr + dr, tc + dc
            if 0 <= nr < n and 0 <= nc < n and ff[nr, nc] == 1.0:
                return True
        return False

    def _tail_reachable_bfs(self) -> bool:
        """Reference BFS implementation (kept for equivalence testing)."""
        if len(self.snake) < 2:
            return True
        from collections import deque

        n = self.n
        head = self.snake[0]
        tail = self.snake[-1]
        body = set(self.snake)
        seen = {head}
        q = deque([head])
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                cell = (nr, nc)
                if cell == tail:
                    return True
                if cell in body or cell in seen:
                    continue
                seen.add(cell)
                q.append(cell)
        return False

    def _flood_fill(self) -> np.ndarray:
        """Flood-fill reachability from head (cached per state)."""
        if self._ff_cache is None:
            self._ff_cache = self._flood_fill_compute()
        return self._ff_cache

    def _flood_fill_compute(self) -> np.ndarray:
        from scipy.ndimage import label

        n = self.n
        # Label connected components of the passable (non-body) cells
        label(~self._occ, structure=self._ff_structure, output=self._ff_labels)
        labels = self._ff_labels

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

    def _current_cycle_target(self) -> Optional[Tuple[int, int]]:
        if not self.cycle_target_obs:
            return None
        if self._curriculum_cycle is None or self._curriculum_head_idx is None:
            return None
        if self.snake_length / float(self.n * self.n) < self.cycle_target_min_fill:
            return None
        next_idx = (self._curriculum_head_idx - 1) % len(self._curriculum_cycle)
        return self._curriculum_cycle[next_idx]

    def _current_tail_target(self) -> Optional[Tuple[int, int]]:
        if not self.tail_target_obs or not self.snake:
            return None
        return self.snake[-1]

    def _current_safe_action_target(self) -> Optional[Tuple[int, int]]:
        if not self.safe_action_target_obs or not self.snake:
            return None
        if self.snake_length / float(self.n * self.n) < self.safe_action_target_min_fill:
            return None

        scores = self.score_relative_actions(fill_weight=self.safe_action_fill_weight)
        if not np.isfinite(np.max(scores)):
            return None

        action = int(np.argmax(scores))
        delta = {0: -1, 1: 0, 2: 1}
        new_dir = (self.direction + delta[action]) % 4
        dr, dc = self.DIRECTIONS[new_dir]
        hr, hc = self.snake_head
        return (hr + dr, hc + dc)

    def _current_safe_action_soft_target(self) -> Optional[np.ndarray]:
        if not self.safe_action_soft_target_obs or not self.snake:
            return None
        if self.snake_length / float(self.n * self.n) < self.safe_action_soft_target_min_fill:
            return None

        raw_scores = np.asarray(
            self.score_relative_actions(fill_weight=self.safe_action_fill_weight),
            dtype=np.float32,
        )
        finite_mask = np.isfinite(raw_scores)
        if not np.any(finite_mask):
            return None

        logits = np.full(3, -np.inf, dtype=np.float32)
        finite_scores = raw_scores[finite_mask]
        max_score = float(np.max(finite_scores))
        temp = max(float(self.safe_action_soft_temperature), 1e-6)
        logits[finite_mask] = (raw_scores[finite_mask] - max_score) / temp
        weights = np.exp(logits - np.max(logits[finite_mask]))
        weights[~finite_mask] = 0.0
        total = float(np.sum(weights))
        if total <= 0.0:
            return None
        return (weights / total).astype(np.float32)

    def _body_age_map(self, min_fill: float) -> Optional[np.ndarray]:
        if not self.snake:
            return None
        if self.snake_length / float(self.n * self.n) < min_fill:
            return None

        target = np.zeros((self.n, self.n), dtype=np.float32)
        denom = float(self.n * self.n)
        length = self.snake_length
        for idx, (r, c) in enumerate(self.snake):
            steps_until_free = length - idx
            target[r, c] = steps_until_free / denom
        return target

    def _current_body_age_target(self) -> Optional[np.ndarray]:
        if not self.body_age_target_obs:
            return None
        return self._body_age_map(self.body_age_target_min_fill)

    def _get_observation_single_frame(self) -> np.ndarray:
        """Single-frame observation before temporal stacking."""
        if self.head_centered:
            return self._get_observation_head_centered()

        obs = np.zeros((self.single_frame_n_channels, self.obs_n, self.obs_n), dtype=np.float32)

        # Channel 0: Head
        hr, hc = self.snake_head
        obs[0, hr + 1, hc + 1] = 1.0

        # Channel 1: Body (includes head)
        obs[1, 1:-1, 1:-1] = self._occ

        # Channel 2: Food
        fr, fc = self.food_pos
        if fr >= 0:
            obs[2, fr + 1, fc + 1] = 1.0

        # Channel 3: Normalized length (broadcast)
        obs[3, :, :] = self.snake_length / float(self.n * self.n)

        # Channel 4: Walls
        obs[4, :, :] = self._walls

        # Channel 5: Flood-fill reachability from head
        if self.flood_fill_obs and self.flood_fill_channel is not None:
            obs[self.flood_fill_channel, 1:-1, 1:-1] = self._flood_fill()

        if self.body_age_obs and self.body_age_obs_channel is not None:
            body_age = self._body_age_map(self.body_age_obs_min_fill)
            if body_age is not None:
                obs[self.body_age_obs_channel, 1:-1, 1:-1] = body_age

        if self.action_history_obs > 0:
            action_offset = 5 + int(self.flood_fill_obs) + int(self.body_age_obs)
            for hist_idx, rel_action in enumerate(self._action_history):
                obs[action_offset + hist_idx * 3 + int(rel_action), :, :] = 1.0

        if self.cycle_target_obs and self.cycle_target_channel is not None:
            target = self._current_cycle_target()
            if target is not None:
                tr, tc = target
                obs[self.cycle_target_channel, tr + 1, tc + 1] = 1.0

        if self.tail_target_obs and self.tail_target_channel is not None:
            target = self._current_tail_target()
            if target is not None:
                tr, tc = target
                obs[self.tail_target_channel, tr + 1, tc + 1] = 1.0

        if self.safe_action_target_obs and self.safe_action_target_channel is not None:
            target = self._current_safe_action_target()
            if target is not None:
                tr, tc = target
                if 0 <= tr < self.n and 0 <= tc < self.n:
                    obs[self.safe_action_target_channel, tr + 1, tc + 1] = 1.0

        if self.safe_action_soft_target_obs and self.safe_action_soft_target_channel is not None:
            target = self._current_safe_action_soft_target()
            if target is not None:
                for action_idx in range(3):
                    obs[self.safe_action_soft_target_channel + action_idx, :, :] = target[action_idx]

        if self.body_age_target_obs and self.body_age_target_channel is not None:
            target = self._current_body_age_target()
            if target is not None:
                obs[self.body_age_target_channel, 1:-1, 1:-1] = target

        # Rotate so snake always faces "up" (direction 0)
        if self.direction != 0:
            obs = np.rot90(obs, k=self.direction, axes=(1, 2)).copy()

        return obs

    def _get_observation_head_centered(self) -> np.ndarray:
        """Single-frame head-centered observation."""
        obs = np.zeros((self.single_frame_n_channels, self.obs_n, self.obs_n), dtype=np.float32)
        hr, hc = self.snake_head
        c = self.obs_n // 2  # center index (19 for 39x39)

        # Overlap between the head-centered window and the board (board always fits)
        gr_start = max(0, c - hr)
        gc_start = max(0, c - hc)
        br_start = max(0, hr - c)
        bc_start = max(0, hc - c)
        bh = min(self.n - br_start, self.obs_n - gr_start)
        bw = min(self.n - bc_start, self.obs_n - gc_start)

        # Channel 0: Head (always at center)
        obs[0, c, c] = 1.0

        # Channel 1: Body (includes head)
        obs[1, gr_start:gr_start + bh, gc_start:gc_start + bw] = \
            self._occ[br_start:br_start + bh, bc_start:bc_start + bw]

        # Channel 2: Food
        fr, fc = self.food_pos
        if fr >= 0:
            obs[2, fr - hr + c, fc - hc + c] = 1.0

        # Channel 3: Normalized length (broadcast)
        obs[3, :, :] = self.snake_length / float(self.n * self.n)

        # Channel 4: Walls (everything outside the board; the board rectangle always
        # fits fully inside the head-centered window)
        obs[4] = 1.0
        obs[4, gr_start:gr_start + bh, gc_start:gc_start + bw] = 0.0

        # Channel 5: Flood-fill reachability
        if self.flood_fill_obs and self.flood_fill_channel is not None:
            ff = self._flood_fill()
            obs[self.flood_fill_channel, gr_start:gr_start + bh, gc_start:gc_start + bw] = \
                ff[br_start:br_start + bh, bc_start:bc_start + bw]

        if self.body_age_obs and self.body_age_obs_channel is not None:
            body_age = self._body_age_map(self.body_age_obs_min_fill)
            if body_age is not None:
                for r, col in self.snake:
                    rr = r - hr + c
                    cc = col - hc + c
                    if 0 <= rr < self.obs_n and 0 <= cc < self.obs_n:
                        obs[self.body_age_obs_channel, rr, cc] = body_age[r, col]

        if self.action_history_obs > 0:
            action_offset = 5 + int(self.flood_fill_obs) + int(self.body_age_obs)
            for hist_idx, rel_action in enumerate(self._action_history):
                obs[action_offset + hist_idx * 3 + int(rel_action), :, :] = 1.0

        if self.cycle_target_obs and self.cycle_target_channel is not None:
            target = self._current_cycle_target()
            if target is not None:
                tr, tc = target
                obs[self.cycle_target_channel, tr - hr + c, tc - hc + c] = 1.0

        if self.tail_target_obs and self.tail_target_channel is not None:
            target = self._current_tail_target()
            if target is not None:
                tr, tc = target
                obs[self.tail_target_channel, tr - hr + c, tc - hc + c] = 1.0

        if self.safe_action_target_obs and self.safe_action_target_channel is not None:
            target = self._current_safe_action_target()
            if target is not None:
                tr, tc = target
                rr = tr - hr + c
                cc = tc - hc + c
                if 0 <= rr < self.obs_n and 0 <= cc < self.obs_n:
                    obs[self.safe_action_target_channel, rr, cc] = 1.0

        if self.safe_action_soft_target_obs and self.safe_action_soft_target_channel is not None:
            target = self._current_safe_action_soft_target()
            if target is not None:
                for action_idx in range(3):
                    obs[self.safe_action_soft_target_channel + action_idx, :, :] = target[action_idx]

        if self.body_age_target_obs and self.body_age_target_channel is not None:
            target = self._current_body_age_target()
            if target is not None:
                for r, col in self.snake:
                    rr = r - hr + c
                    cc = col - hc + c
                    if 0 <= rr < self.obs_n and 0 <= cc < self.obs_n:
                        obs[self.body_age_target_channel, rr, cc] = target[r, col]

        # Rotate so snake always faces "up"
        if self.direction != 0:
            obs = np.rot90(obs, k=self.direction, axes=(1, 2)).copy()

        return obs

    def _history_zero(self) -> np.ndarray:
        if self._history_zero_frame is None:
            self._history_zero_frame = np.zeros(
                (self.base_obs_channels, self.obs_n, self.obs_n),
                dtype=np.float32,
            )
        return self._history_zero_frame

    def _get_observation(self) -> np.ndarray:
        """Observation with current-frame-first temporal stacking."""
        single = self._get_observation_single_frame()
        base = single[:self.base_obs_channels]
        if self.aux_target_channels > 0:
            aux = single[self.base_obs_channels:self.single_frame_n_channels]
        else:
            aux = None

        if self.obs_history > 1:
            history_frames = list(self._obs_history_frames)
            while len(history_frames) < self.obs_history - 1:
                history_frames.append(self._history_zero())
            stacked_base = np.concatenate([base, *history_frames[: self.obs_history - 1]], axis=0)
        else:
            stacked_base = base

        obs = stacked_base if aux is None else np.concatenate([stacked_base, aux], axis=0)
        if self.obs_history > 1:
            self._obs_history_frames.appendleft(base.copy())
        return obs

    def _build_hamiltonian_cycle(self) -> list[Tuple[int, int]]:
        """Build a Hamiltonian cycle for even board sizes."""
        if self.n % 2 != 0:
            raise ValueError("Curriculum Hamiltonian cycle requires an even board size")

        cycle = []

        # Keep the first column as the return corridor so the walk closes into a cycle.
        for c in range(self.n):
            cycle.append((0, c))
        for r in range(1, self.n):
            if r % 2 == 1:
                cols = range(self.n - 1, 0, -1)
            else:
                cols = range(1, self.n)
            for c in cols:
                cycle.append((r, c))
        for r in range(self.n - 1, 0, -1):
            cycle.append((r, 0))
        return cycle

    def _transform_cycle(
        self,
        cycle: list[Tuple[int, int]],
        *,
        transpose: bool = False,
        flip_rows: bool = False,
        flip_cols: bool = False,
    ) -> list[Tuple[int, int]]:
        transformed = []
        for r, c in cycle:
            if transpose:
                r, c = c, r
            if flip_rows:
                r = self.n - 1 - r
            if flip_cols:
                c = self.n - 1 - c
            transformed.append((r, c))
        return transformed

    def _build_curriculum_cycles(self) -> list[list[Tuple[int, int]]]:
        """Build a small family of Hamiltonian cycles for curriculum resets."""
        base_cycle = self._build_hamiltonian_cycle()
        variants = []
        seen = set()
        transforms = (
            (False, False, False),
            (False, False, True),
            (False, True, False),
            (False, True, True),
            (True, False, False),
            (True, False, True),
            (True, True, False),
            (True, True, True),
        )
        for transpose, flip_rows, flip_cols in transforms:
            cycle = self._transform_cycle(
                base_cycle,
                transpose=transpose,
                flip_rows=flip_rows,
                flip_cols=flip_cols,
            )
            for candidate in (cycle, list(reversed(cycle))):
                key = tuple(candidate)
                if key not in seen:
                    seen.add(key)
                    variants.append(candidate)
        return variants

    def _reset_with_fill(self) -> None:
        """Curriculum reset: place snake on a sampled Hamiltonian cycle segment."""
        board_area = self.n * self.n
        min_len = max(3, int(self.curriculum_min_fill * board_area))
        max_len = min(board_area - 1, int(self.curriculum_max_fill * board_area))
        target_len = int(self.rng.integers(min_len, max_len + 1))

        cycle = self._curriculum_cycles[int(self.rng.integers(len(self._curriculum_cycles)))]
        start_idx = int(self.rng.integers(0, len(cycle)))

        # Snake: head is at start_idx, body follows forward along the cycle.
        self.snake = [cycle[(start_idx + i) % len(cycle)] for i in range(target_len)]
        self._curriculum_cycle = cycle
        self._curriculum_head_idx = start_idx

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
            self._curriculum_cycle = None
            self._curriculum_head_idx = None
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

        self._rebuild_occ()
        self._place_food()

        self.steps_since_food = 0
        self.score = max(0, self.snake_length - 3)
        self.total_steps = 0
        self.prev_phi = self._compute_phi()
        self.prev_tail_phi = 0.0
        if self.tail_safety_pbrs != 0.0:
            if (
                self.snake_length / float(self.n * self.n) >= self.tail_safety_pbrs_min_fill
                and not self._tail_reachable()
            ):
                self.prev_tail_phi = self.tail_safety_pbrs
        self._obs_history_frames.clear()
        self._action_history.clear()

        obs = self._get_observation()
        info = {
            "length": self.snake_length,
            "score": self.score,
            "food_pos": self.food_pos,
        }

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.total_steps += 1
        self._ff_cache = None

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
        follow_bonus = 0.0
        safe_action_bonus = 0.0
        reason = None
        followed_cycle = False

        if self._curriculum_cycle is not None and self._curriculum_head_idx is not None:
            next_idx = (self._curriculum_head_idx - 1) % len(self._curriculum_cycle)
            expected_head = self._curriculum_cycle[next_idx]
            if new_head == expected_head:
                followed_cycle = True
                if (
                    self.curriculum_follow_bonus != 0.0
                    and self.snake_length / float(self.n * self.n) >= self.curriculum_follow_min_fill
                ):
                    follow_bonus = self.curriculum_follow_bonus
            else:
                if (
                    self.curriculum_follow_bonus != 0.0
                    and self.snake_length / float(self.n * self.n) >= self.curriculum_follow_min_fill
                ):
                    follow_bonus = -self.curriculum_follow_bonus
                self._curriculum_cycle = None
                self._curriculum_head_idx = None

        if (
            self.safe_action_bonus != 0.0
            and self.snake_length / float(self.n * self.n) >= self.safe_action_bonus_min_fill
        ):
            safe_scores = self.score_relative_actions(fill_weight=self.safe_action_bonus_fill_weight)
            best_score = max(safe_scores)
            if np.isfinite(best_score) and np.isfinite(safe_scores[int(action)]):
                if safe_scores[int(action)] >= best_score - 1e-6:
                    safe_action_bonus = self.safe_action_bonus

        # Check wall collision
        if not (0 <= new_head[0] < self.n and 0 <= new_head[1] < self.n):
            terminated = True
            base_reward = -1.0
            reason = "wall"
        elif self._occ[new_head]:
            # Moving into the tail cell is safe only when the tail vacates this
            # step (i.e. not eating); every other occupied cell is a self collision.
            if new_head != self.snake[-1] or new_head == self.food_pos:
                terminated = True
                base_reward = -1.0
                reason = "self"

        if not terminated:
            if new_head == self.food_pos:
                self.snake.insert(0, new_head)
                self._occ[new_head] = True
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
                old_tail = self.snake.pop()
                self._occ[old_tail] = False
                self._occ[new_head] = True
                self.steps_since_food += 1

            if followed_cycle and self._curriculum_head_idx is not None:
                self._curriculum_head_idx = (self._curriculum_head_idx - 1) % len(self._curriculum_cycle)

        if not terminated and self.steps_since_food > self.max_no_food:
            if self.stall_terminates:
                terminated = True  # Proper termination - PPO won't bootstrap
            else:
                truncated = True  # Old behavior - PPO bootstraps (underpenalizes)
            base_reward += self.stall_penalty
            reason = "stall"

        # Topology penalty: at high fill, punish a surviving move that strands free
        # cells (splits the free space so the head can no longer reach part of it).
        # That stranding is the self-trap, set a few moves before death — penalizing it
        # at its cause gives the dense endgame-discipline signal RL otherwise lacks.
        topo_penalty = 0.0
        if (
            not terminated
            and self.topology_penalty != 0.0
            and self.snake_length / float(self.n * self.n) >= self.topology_penalty_min_fill
        ):
            reachable = float(np.sum(self._flood_fill()))
            total_free = float(self.n * self.n - self.snake_length)
            if total_free > 0:
                stranded_frac = max(0.0, (total_free - reachable) / total_free)
                topo_penalty = self.topology_penalty * stranded_frac

        # Tail-safety penalty: at high fill, a flat penalty for a surviving move that
        # leaves the tail unreachable (a move that survives now but dooms the snake).
        # Binary and survival-relevant; the cleaner replacement for topology_penalty.
        tail_pen = 0.0
        if (
            not terminated
            and self.tail_safety_penalty != 0.0
            and self.snake_length / float(self.n * self.n) >= self.tail_safety_min_fill
        ):
            if not self._tail_reachable():
                tail_pen = self.tail_safety_penalty

        # Tail-safety PBRS: potential phi = coef (negative) while the tail is
        # unreachable at high fill, else 0. The shaped term gamma*phi(s') - phi(s)
        # charges entry into an unreachable state and refunds recovery, telescoping
        # to ~0 along surviving paths — unlike the flat penalty, it cannot reshape
        # the mid-game value landscape (Ng et al. policy invariance).
        tail_pbrs = 0.0
        if self.tail_safety_pbrs != 0.0 and not terminated:
            phi_tail = 0.0
            if (
                self.snake_length / float(self.n * self.n) >= self.tail_safety_pbrs_min_fill
                and not self._tail_reachable()
            ):
                phi_tail = self.tail_safety_pbrs
            tail_pbrs = self.gamma * phi_tail - self.prev_tail_phi
            self.prev_tail_phi = phi_tail

        if not terminated:
            phi = self._compute_phi()
            r_shape = self.gamma * phi - self.prev_phi
            self.prev_phi = phi
        else:
            r_shape = 0.0

        if terminated and reason != "win":
            reward = base_reward
        else:
            reward = base_reward + r_shape + self.survival_bonus + follow_bonus + safe_action_bonus + topo_penalty + tail_pen + tail_pbrs

        if self.action_history_obs > 0:
            self._action_history.appendleft(int(action))
        obs = self._get_observation()
        info = {
            "length": self.snake_length,
            "score": self.score,
            "reason": reason,
            "steps": self.total_steps,
            "food_pos": self.food_pos,
        }
        return obs, reward, terminated, truncated, info

    def _snapshot_state(self) -> tuple:
        return (
            list(self.snake),
            self.direction,
            self.food_pos,
            self.steps_since_food,
            self.score,
            self.prev_phi,
            self.prev_tail_phi,
            self.total_steps,
            self._curriculum_cycle,
            self._curriculum_head_idx,
            [frame.copy() for frame in self._obs_history_frames],
            list(self._action_history),
            copy.deepcopy(self.rng.bit_generator.state),
        )

    def _restore_state(self, snapshot: tuple) -> None:
        (
            snake,
            direction,
            food_pos,
            steps_since_food,
            score,
            prev_phi,
            prev_tail_phi,
            total_steps,
            curriculum_cycle,
            curriculum_head_idx,
            obs_history_frames,
            action_history,
            rng_state,
        ) = snapshot
        self.snake = list(snake)
        self.direction = direction
        self.food_pos = food_pos
        self.steps_since_food = steps_since_food
        self.score = score
        self.prev_phi = prev_phi
        self.prev_tail_phi = prev_tail_phi
        self.total_steps = total_steps
        self._curriculum_cycle = curriculum_cycle
        self._curriculum_head_idx = curriculum_head_idx
        self._obs_history_frames.clear()
        for frame in obs_history_frames:
            self._obs_history_frames.append(frame.copy())
        self._action_history.clear()
        for action in action_history:
            self._action_history.append(int(action))
        self.rng.bit_generator.state = copy.deepcopy(rng_state)

    def score_relative_actions(self, fill_weight: float = 500.0) -> list[float]:
        """Score relative actions with a one-step lookahead heuristic.

        Scores are ordered for relative actions [left, straight, right].
        Immediate wins score +inf, immediate deaths score -inf. Surviving
        actions are ranked by reachable-space after the move plus a small
        preference for higher fill, which nudges the heuristic toward food
        collection without ignoring maneuverability.
        """
        scores = []
        snapshot = self._snapshot_state()
        board_area = float(self.n * self.n)
        safe_action_target_obs = self.safe_action_target_obs
        safe_action_soft_target_obs = self.safe_action_soft_target_obs
        safe_action_bonus = self.safe_action_bonus
        self.safe_action_target_obs = False
        self.safe_action_soft_target_obs = False
        self.safe_action_bonus = 0.0
        for action in range(3):
            _, _, terminated, truncated, info = self.step(action)
            if terminated or truncated:
                if info.get("reason") == "win":
                    heuristic = float("inf")
                else:
                    heuristic = float("-inf")
            else:
                reachable = float(np.sum(self._flood_fill()))
                heuristic = reachable + fill_weight * (self.snake_length / board_area)
            scores.append(heuristic)
            self._restore_state(snapshot)
        self.safe_action_target_obs = safe_action_target_obs
        self.safe_action_soft_target_obs = safe_action_soft_target_obs
        self.safe_action_bonus = safe_action_bonus
        return scores

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
