"""Train Snake with PufferLib PPO."""

from __future__ import annotations

import multiprocessing
import sys

# Use 'spawn' to avoid PyTorch/MPS crashes on worker exit (macOS)
# Must be set before any multiprocessing happens
try:
    if sys.platform == 'darwin':
        multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # Already set

import argparse
import copy
from collections import deque
import os
import shlex
import time

import gymnasium as gym
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

import pufferlib
import pufferlib.emulation
import pufferlib.vector
from pufferlib import pufferl

from eval_metrics import summarize_phase_metrics
from snake_env import SnakeEnv
from experiment_tracker import ExperimentTracker


class SnakeSymmetricAugmentation(gym.Wrapper):
    """Horizontal flip augmentation for Snake.

    With probability `flip_prob`, flips the observation horizontally each episode.
    When flipped:
      - Observation columns are reversed
      - Actions swapped: left (0) ↔ right (2)
    """

    def __init__(self, env: gym.Env, flip_prob: float = 0.5, seed: int = 0):
        super().__init__(env)
        self.flip_prob = flip_prob
        self.rng = np.random.default_rng(seed + 1)  # Different seed from env
        self.flipped = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed + 1)
        self.flipped = self.rng.random() < self.flip_prob
        obs, info = self.env.reset(seed=seed, options=options)
        if self.flipped:
            obs = np.flip(obs, axis=2).copy()
        return obs, info

    def step(self, action):
        if self.flipped:
            # Swap left (0) and right (2), keep straight (1)
            if action == 0:
                action = 2
            elif action == 2:
                action = 0
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.flipped:
            obs = np.flip(obs, axis=2).copy()
        return obs, reward, terminated, truncated, info


class SnakeEpisodeStats(gym.Wrapper):
    """Episode stats wrapper that logs final Snake score correctly.

    PufferLib's generic `EpisodeStats` sums all per-step `info` values, which is
    wrong for cumulative fields like `score` and `length`. This wrapper emits
    episode-level aggregates only on termination/truncation:
      - `episode_return`: sum of rewards
      - `episode_length`: number of env steps
      - `episode_score`: final `info["score"]`
      - `episode_win`: 1 if `info["reason"] == "win"` else 0
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        elite_dir: str | None = None,
        elite_score_threshold: int | None = None,
        elite_min_fill: float = 0.90,
        elite_safe_action_labels: bool = False,
        elite_safe_action_fill_weight: float = 500.0,
    ):
        super().__init__(env)
        self.episode_return = 0.0
        self.episode_length = 0
        self.elite_dir = elite_dir
        self.elite_score_threshold = elite_score_threshold
        self.elite_min_fill = elite_min_fill
        self.elite_safe_action_labels = elite_safe_action_labels
        self.elite_safe_action_fill_weight = elite_safe_action_fill_weight
        self.last_obs = None
        self.elite_obs = []
        self.elite_actions = []
        if self.elite_dir is not None:
            os.makedirs(self.elite_dir, exist_ok=True)

    def _reset_elite_buffers(self):
        self.last_obs = None
        self.elite_obs = []
        self.elite_actions = []

    def _elite_fill_fraction(self) -> float:
        base_env = self.unwrapped
        snake_length = getattr(base_env, "snake_length", None)
        board_n = getattr(base_env, "n", None)
        if snake_length is None or board_n is None:
            return 0.0
        return float(snake_length) / float(board_n * board_n)

    def _save_elite_trajectory(self, score: int) -> str | None:
        if self.elite_dir is None or self.elite_score_threshold is None:
            return None
        if score < self.elite_score_threshold:
            return None
        if not self.elite_obs or len(self.elite_obs) != len(self.elite_actions):
            return None

        path = os.path.join(
            self.elite_dir,
            f"elite_{os.getpid()}_{time.time_ns()}_{score}.npz",
        )
        np.savez_compressed(
            path,
            observations=np.asarray(self.elite_obs, dtype=np.float32),
            actions=np.asarray(self.elite_actions, dtype=np.int64),
        )
        return path

    def reset(self, *, seed=None, options=None):
        self.episode_return = 0.0
        self.episode_length = 0
        self._reset_elite_buffers()
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = np.array(obs, copy=True)
        return obs, info

    def step(self, action):
        if (
            self.elite_dir is not None
            and self.elite_score_threshold is not None
            and self.last_obs is not None
            and self._elite_fill_fraction() >= self.elite_min_fill
        ):
            recorded_action = int(action)
            if self.elite_safe_action_labels:
                base_env = self.unwrapped
                if hasattr(base_env, "score_relative_actions"):
                    scores = base_env.score_relative_actions(
                        fill_weight=self.elite_safe_action_fill_weight,
                    )
                    if np.isfinite(np.max(scores)):
                        recorded_action = int(np.argmax(scores))
                        if getattr(self.env, "flipped", False):
                            if recorded_action == 0:
                                recorded_action = 2
                            elif recorded_action == 2:
                                recorded_action = 0
            self.elite_obs.append(np.array(self.last_obs, copy=True))
            self.elite_actions.append(recorded_action)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_return += float(reward)
        self.episode_length += 1

        if terminated or truncated:
            reason = info.get("reason", None)
            end_info = {
                "episode_return": self.episode_return,
                "episode_length": self.episode_length,
                "episode_score": int(info.get("score", 0)),
                "episode_win": 1 if reason == "win" else 0,
            }
            elite_path = self._save_elite_trajectory(int(info.get("score", 0)))
            if elite_path is not None:
                end_info["elite_path"] = elite_path
            self._reset_elite_buffers()
            return obs, reward, terminated, truncated, end_info

        self.last_obs = np.array(obs, copy=True)
        return obs, reward, terminated, truncated, {}


def make_snake_env(
    *,
    n: int,
    gamma: float,
    alpha: float,
    symmetric: bool = False,
    stall_penalty: float = -1.0,
    stall_terminates: bool = True,
    max_no_food_base: int = None,
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
    cycle_target_min_fill: float | None = None,
    safe_action_target_min_fill: float = 0.90,
    safe_action_soft_target_min_fill: float = 0.90,
    body_age_target_min_fill: float = 0.80,
    safe_action_fill_weight: float = 500.0,
    safe_action_soft_temperature: float = 1.0,
    safe_action_bonus: float = 0.0,
    safe_action_bonus_min_fill: float = 0.95,
    safe_action_bonus_fill_weight: float = 500.0,
    elite_dir: str | None = None,
    elite_score_threshold: int | None = None,
    elite_min_fill: float = 0.90,
    elite_safe_action_labels: bool = False,
    elite_safe_action_fill_weight: float = 500.0,
    head_centered: bool = False,
    buf=None,
    seed=None,
):
    seed = 0 if seed is None else int(seed)
    env = SnakeEnv(
        n=n,
        gamma=gamma,
        alpha=alpha,
        seed=seed,
        stall_penalty=stall_penalty,
        stall_terminates=stall_terminates,
        max_no_food_base=max_no_food_base,
        flood_fill_obs=flood_fill_obs,
        body_age_obs=body_age_obs,
        obs_history=obs_history,
        action_history_obs=action_history_obs,
        curriculum_prob=curriculum_prob,
        curriculum_min_fill=curriculum_min_fill,
        curriculum_max_fill=curriculum_max_fill,
        curriculum_follow_bonus=curriculum_follow_bonus,
        curriculum_follow_min_fill=curriculum_follow_min_fill,
        cycle_target_obs=cycle_target_obs,
        tail_target_obs=tail_target_obs,
        cycle_target_min_fill=cycle_target_min_fill,
        safe_action_target_obs=safe_action_target_obs,
        safe_action_soft_target_obs=safe_action_soft_target_obs,
        safe_action_target_min_fill=safe_action_target_min_fill,
        safe_action_soft_target_min_fill=safe_action_soft_target_min_fill,
        body_age_target_obs=body_age_target_obs,
        body_age_obs_min_fill=body_age_obs_min_fill,
        body_age_target_min_fill=body_age_target_min_fill,
        safe_action_fill_weight=safe_action_fill_weight,
        safe_action_soft_temperature=safe_action_soft_temperature,
        safe_action_bonus=safe_action_bonus,
        safe_action_bonus_min_fill=safe_action_bonus_min_fill,
        safe_action_bonus_fill_weight=safe_action_bonus_fill_weight,
        head_centered=head_centered,
    )
    if symmetric:
        env = SnakeSymmetricAugmentation(env, flip_prob=0.5, seed=seed)
    env = SnakeEpisodeStats(
        env,
        elite_dir=elite_dir,
        elite_score_threshold=elite_score_threshold,
        elite_min_fill=elite_min_fill,
        elite_safe_action_labels=elite_safe_action_labels,
        elite_safe_action_fill_weight=elite_safe_action_fill_weight,
    )
    return pufferlib.emulation.GymnasiumPufferEnv(env, buf=buf)


class SnakePolicy(nn.Module):
    """FC policy for Snake.

    Scale 1x (default): obs -> 1024 -> 512 -> 256 -> 128
    Scale 2x: obs -> 2048 -> 1024 -> 512 -> 256
    """

    def __init__(self, env, scale: int = 1, aux_flood_fill: bool = False,
                 aux_cycle_target: bool = False, aux_tail_target: bool = False,
                 aux_safe_action_target: bool = False, aux_safe_action_soft_target: bool = False,
                 aux_body_age_target: bool = False,
                 board_size: int = None,
                 head_centered: bool = False,
                 late_head_min_fill: float | None = None):
        super().__init__()

        obs_space = getattr(env, "single_observation_space", env.observation_space)
        act_space = getattr(env, "single_action_space", env.action_space)
        obs_shape = obs_space.shape
        total_channels = obs_shape[0]
        n_actions = act_space.n

        self.aux_flood_fill = aux_flood_fill
        self.aux_cycle_target = aux_cycle_target
        self.aux_tail_target = aux_tail_target
        self.aux_safe_action_target = aux_safe_action_target
        self.aux_safe_action_soft_target = aux_safe_action_soft_target
        self.aux_body_age_target = aux_body_age_target
        self.head_centered = head_centered
        self.late_head_min_fill = late_head_min_fill
        self.aux_target_channels = (
            int(aux_flood_fill)
            + int(aux_cycle_target)
            + int(aux_tail_target)
            + int(aux_safe_action_target)
            + 3 * int(aux_safe_action_soft_target)
            + int(aux_body_age_target)
        )
        self.encoder_channels = total_channels - self.aux_target_channels

        # Compute input size from encoder channels
        n_input = self.encoder_channels * obs_shape[1] * obs_shape[2]
        if board_size is not None:
            self.board_size = board_size
        else:
            self.board_size = obs_shape[1] - 2  # strip wall padding (non-head-centered)

        # Scale network width
        w = [1024, 512, 256, 128]
        if scale == 2:
            w = [2048, 1024, 512, 256]
        elif scale == 4:
            w = [4096, 2048, 1024, 512]

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input, w[0]),
            nn.LayerNorm(w[0]),
            nn.ReLU(),
            nn.Linear(w[0], w[1]),
            nn.LayerNorm(w[1]),
            nn.ReLU(),
            nn.Linear(w[1], w[2]),
            nn.LayerNorm(w[2]),
            nn.ReLU(),
            nn.Linear(w[2], w[3]),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(w[3], w[3] // 2),
            nn.ReLU(),
            nn.Linear(w[3] // 2, n_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(w[3], w[3]),
            nn.ReLU(),
            nn.Linear(w[3], w[3] // 2),
            nn.ReLU(),
            nn.Linear(w[3] // 2, 1),
        )

        if late_head_min_fill is not None:
            self.late_policy_head = nn.Sequential(
                nn.Linear(w[3], w[3] // 2),
                nn.ReLU(),
                nn.Linear(w[3] // 2, n_actions),
            )
            self.late_value_head = nn.Sequential(
                nn.Linear(w[3], w[3]),
                nn.ReLU(),
                nn.Linear(w[3], w[3] // 2),
                nn.ReLU(),
                nn.Linear(w[3] // 2, 1),
            )

        # Auxiliary flood-fill decoder (training only)
        if aux_flood_fill:
            if head_centered:
                # Head-centered: predict full obs grid (no fixed wall border to strip)
                self.flood_target_n = obs_shape[1]
            else:
                # Non-head-centered: predict inner board grid (strip wall padding)
                self.flood_target_n = self.board_size
            self.flood_decoder = nn.Sequential(
                nn.Linear(w[3], w[2]),
                nn.ReLU(),
                nn.Linear(w[2], self.flood_target_n * self.flood_target_n),
            )

        if aux_cycle_target:
            if head_centered:
                self.cycle_target_n = obs_shape[1]
            else:
                self.cycle_target_n = self.board_size
            self.cycle_target_decoder = nn.Sequential(
                nn.Linear(w[3], w[2]),
                nn.ReLU(),
                nn.Linear(w[2], self.cycle_target_n * self.cycle_target_n),
            )

        if aux_tail_target:
            if head_centered:
                self.tail_target_n = obs_shape[1]
            else:
                self.tail_target_n = self.board_size
            self.tail_target_decoder = nn.Sequential(
                nn.Linear(w[3], w[2]),
                nn.ReLU(),
                nn.Linear(w[2], self.tail_target_n * self.tail_target_n),
            )

        if aux_body_age_target:
            if head_centered:
                self.body_age_target_n = obs_shape[1]
            else:
                self.body_age_target_n = self.board_size
            self.body_age_target_decoder = nn.Sequential(
                nn.Linear(w[3], w[2]),
                nn.ReLU(),
                nn.Linear(w[2], self.body_age_target_n * self.body_age_target_n),
            )

        self._init_weights()
        self._sync_late_heads_from_base()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def _sync_late_heads_from_base(self):
        if self.late_head_min_fill is None:
            return
        self.late_policy_head.load_state_dict(self.policy_head.state_dict())
        self.late_value_head.load_state_dict(self.value_head.state_dict())

    def forward_eval(self, observations, state=None):
        obs_input = observations[:, :self.encoder_channels]
        features = self.features(obs_input)
        logits = self.policy_head(features)
        values = self.value_head(features)
        if self.late_head_min_fill is not None:
            fill_fraction = obs_input[:, 3].mean(dim=(1, 2))
            late_mask = fill_fraction >= self.late_head_min_fill
            if torch.any(late_mask):
                logits = logits.clone()
                values = values.clone()
                logits[late_mask] = self.late_policy_head(features[late_mask])
                values[late_mask] = self.late_value_head(features[late_mask])
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def forward_flood_predict(self, enc_input):
        """Predict flood-fill from encoder input. Returns (B, 1, n, n) logits."""
        features = self.features(enc_input)
        pred = self.flood_decoder(features)
        n = self.flood_target_n
        return pred.view(-1, 1, n, n)

    def forward_cycle_target_predict(self, enc_input):
        """Predict curriculum cycle target cell from encoder input."""
        features = self.features(enc_input)
        pred = self.cycle_target_decoder(features)
        n = self.cycle_target_n
        return pred.view(-1, 1, n, n)

    def forward_tail_target_predict(self, enc_input):
        """Predict tail position from encoder input."""
        features = self.features(enc_input)
        pred = self.tail_target_decoder(features)
        n = self.tail_target_n
        return pred.view(-1, 1, n, n)

    def forward_body_age_target_predict(self, enc_input):
        """Predict temporal occupancy horizon map from encoder input."""
        features = self.features(enc_input)
        pred = self.body_age_target_decoder(features)
        n = self.body_age_target_n
        return pred.view(-1, 1, n, n)


class SnakeCNNPolicy(nn.Module):
    """CNN policy for Snake - better at spatial patterns."""

    def __init__(self, env, scale: int = 1):
        super().__init__()

        obs_space = getattr(env, "single_observation_space", env.observation_space)
        act_space = getattr(env, "single_action_space", env.action_space)
        obs_shape = obs_space.shape  # (C, H, W)
        n_channels = obs_shape[0]
        n_actions = act_space.n

        # Scale conv channels: 1x=small, 2x=medium, 4x=large
        if scale >= 4:
            c = [128, 256, 256]
            hidden = 1024
        elif scale >= 2:
            c = [64, 128, 128]
            hidden = 512
        else:
            c = [32, 64, 64]
            hidden = 256

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, c[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c[0], c[1], kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(c[1], c[2], kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            conv_out = self.conv(dummy).shape[1]

        self.features = nn.Sequential(
            nn.Linear(conv_out, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward_eval(self, observations, state=None):
        x = self.conv(observations)
        features = self.features(x)
        logits = self.policy_head(features)
        values = self.value_head(features)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


class ResBlock(nn.Module):
    """Pre-activation residual block (BN -> ReLU -> Conv -> BN -> ReLU -> Conv)."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class SnakeResNetPolicy(nn.Module):
    """Deep ResNet policy — maintains spatial resolution for iterative reasoning.

    No pooling/striding: all conv layers see full 22x22 grid.
    With N residual blocks (2 conv layers each), receptive field ≈ 2*N+1.
    15 blocks → RF=31, enough to span the 22x22 grid.
    """

    def __init__(self, env, scale: int = 1):
        super().__init__()

        obs_space = getattr(env, "single_observation_space", env.observation_space)
        act_space = getattr(env, "single_action_space", env.action_space)
        obs_shape = obs_space.shape  # (C, H, W)
        n_channels = obs_shape[0]
        n_actions = act_space.n

        # Scale: channels and number of residual blocks
        if scale >= 4:
            channels = 128
            n_blocks = 19
        elif scale >= 2:
            channels = 64
            n_blocks = 15
        else:
            channels = 32
            n_blocks = 10

        # Initial projection
        self.input_conv = nn.Conv2d(n_channels, channels, 3, padding=1, bias=False)

        # Residual tower (no spatial downsampling)
        self.res_tower = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        self.post_bn = nn.BatchNorm2d(channels)
        self.post_relu = nn.ReLU()

        # Global average pool → small feature vector
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.policy_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, n_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_eval(self, observations, state=None):
        x = self.input_conv(observations)
        x = self.res_tower(x)
        x = self.post_relu(self.post_bn(x))
        x = self.gap(x).flatten(1)
        logits = self.policy_head(x)
        values = self.value_head(x)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


class IterativeBlock(nn.Module):
    """Weight-tied convolutional block for iterative spatial reasoning.

    Same local operation applied K times with shared weights, simulating
    BFS/flood-fill diffusion. K iterations of 3x3 conv covers RF of 2K+1.
    """

    def __init__(self, channels: int):
        super().__init__()
        n_groups = min(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(n_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(n_groups, channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return torch.relu(out + residual)


class SnakeIterativeCNNPolicy(nn.Module):
    """Weight-tied iterative CNN for learning spatial reasoning.

    A single conv block applied K times with shared weights provides the
    inductive bias for BFS/flood-fill-like computation. K iterations of
    3x3 convolution covers receptive field of 2K+1.

    With aux_flood_fill=True, the encoder only processes N-1 channels
    (excluding the last flood-fill channel) and a decoder head is trained
    to predict flood-fill from the learned spatial features.
    """

    def __init__(self, env=None, scale: int = 1, n_iterations: int = 12,
                 aux_flood_fill: bool = False, *,
                 board_size: int = None, n_channels: int = None):
        super().__init__()

        if env is not None:
            obs_space = getattr(env, "single_observation_space", env.observation_space)
            act_space = getattr(env, "single_action_space", env.action_space)
            total_channels = obs_space.shape[0]
            n_actions = act_space.n
        else:
            assert board_size is not None and n_channels is not None
            total_channels = n_channels
            n_actions = 3

        if scale >= 4:
            channels = 128
            hidden = 512
        elif scale >= 2:
            channels = 64
            hidden = 256
        else:
            channels = 32
            hidden = 128

        self.n_iterations = n_iterations
        self.aux_flood_fill = aux_flood_fill
        self.channels = channels

        # Encoder processes all channels, or strips flood-fill target
        if aux_flood_fill:
            self.encoder_channels = total_channels - 1
        else:
            self.encoder_channels = total_channels

        n_groups = min(8, channels)

        # Input projection
        self.input_conv = nn.Sequential(
            nn.Conv2d(self.encoder_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(n_groups, channels),
            nn.ReLU(),
        )

        # Weight-tied iterative block (shared weights across all iterations)
        self.iter_block = IterativeBlock(channels)

        # Post-iteration normalization
        self.post_norm = nn.GroupNorm(n_groups, channels)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Actor head
        self.policy_head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

        # Critic head
        self.value_head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Auxiliary flood-fill decoder (training only, dropped at eval)
        if aux_flood_fill:
            self.flood_decoder = nn.Sequential(
                nn.Conv2d(channels, channels // 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels // 2, 1, 1),
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_spatial(self, observations):
        """Compute spatial features via weight-tied iterative convolution."""
        x = self.input_conv(observations)
        for _ in range(self.n_iterations):
            x = self.iter_block(x)
        x = torch.relu(self.post_norm(x))
        return x

    def forward_eval(self, observations, state=None):
        obs_input = observations[:, :self.encoder_channels]
        spatial = self.forward_spatial(obs_input)
        features = self.gap(spatial).flatten(1)
        logits = self.policy_head(features)
        values = self.value_head(features)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def forward_flood_predict(self, enc_input):
        """Predict flood-fill from encoder input. Returns (B, 1, n, n) logits."""
        spatial = self.forward_spatial(enc_input)
        pred = self.flood_decoder(spatial)
        return pred[:, :, 1:-1, 1:-1]  # strip wall padding


def _auto_num_workers(num_envs: int) -> int:
    try:
        physical = psutil.cpu_count(logical=False) or 1
    except Exception:
        physical = 1
    max_workers = min(num_envs, physical)
    for w in range(max_workers, 0, -1):
        if num_envs % w == 0:
            return w
    return 1


@torch.no_grad()
def evaluate_policy(
    *,
    policy: nn.Module,
    device: str,
    board_size: int,
    episodes: int,
    seed: int,
    deterministic: bool,
    gamma: float,
    alpha: float,
    flood_fill_obs: bool = False,
    body_age_obs: bool = False,
    obs_history: int = 1,
    action_history_obs: int = 0,
    cycle_target_obs: bool = False,
    tail_target_obs: bool = False,
    safe_action_target_obs: bool = False,
    safe_action_soft_target_obs: bool = False,
    body_age_target_obs: bool = False,
    cycle_target_min_fill: float | None = None,
    safe_action_target_min_fill: float = 0.90,
    safe_action_soft_target_min_fill: float = 0.90,
    body_age_target_min_fill: float = 0.80,
    safe_action_fill_weight: float = 500.0,
    safe_action_soft_temperature: float = 1.0,
    safe_action_bonus: float = 0.0,
    safe_action_bonus_min_fill: float = 0.95,
    safe_action_bonus_fill_weight: float = 500.0,
    head_centered: bool = False,
) -> dict:
    env = SnakeEnv(
        n=board_size,
        gamma=gamma,
        alpha=alpha,
        seed=seed,
        flood_fill_obs=flood_fill_obs,
        body_age_obs=body_age_obs,
        obs_history=obs_history,
        action_history_obs=action_history_obs,
        cycle_target_obs=cycle_target_obs,
        tail_target_obs=tail_target_obs,
        cycle_target_min_fill=cycle_target_min_fill,
        safe_action_target_obs=safe_action_target_obs,
        safe_action_soft_target_obs=safe_action_soft_target_obs,
        safe_action_target_min_fill=safe_action_target_min_fill,
        safe_action_soft_target_min_fill=safe_action_soft_target_min_fill,
        body_age_target_obs=body_age_target_obs,
        body_age_target_min_fill=body_age_target_min_fill,
        safe_action_fill_weight=safe_action_fill_weight,
        safe_action_soft_temperature=safe_action_soft_temperature,
        safe_action_bonus=safe_action_bonus,
        safe_action_bonus_min_fill=safe_action_bonus_min_fill,
        safe_action_bonus_fill_weight=safe_action_bonus_fill_weight,
        head_centered=head_centered,
    )

    perfect_score = board_size * board_size - 3
    scores = []
    terminal_lengths = []
    reasons = []
    wins = 0

    policy.eval()
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        last_info = info
        while not done:
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
            logits, _ = policy.forward_eval(obs_t, state=None)
            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                action = int(torch.distributions.Categorical(logits=logits).sample().item())
            obs, _, terminated, truncated, last_info = env.step(action)
            done = terminated or truncated

        score = int(last_info.get("score", 0))
        terminal_lengths.append(int(last_info.get("length", score + 3)))
        reasons.append(str(last_info.get("reason", "unknown")))
        scores.append(score)
        if score >= perfect_score:
            wins += 1

    stats = {
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "median_score": float(np.median(scores)) if scores else 0.0,
        "win_rate": float(wins / max(1, episodes)),
        "perfect_score": int(perfect_score),
        "episodes": int(episodes),
    }
    stats.update(
        summarize_phase_metrics(
            scores=scores,
            terminal_lengths=terminal_lengths,
            reasons=reasons,
            perfect_score=perfect_score,
            episodes=episodes,
        )
    )
    return stats


def _format_command() -> str:
    parts = [sys.executable] + sys.argv
    return " ".join(shlex.quote(part) for part in parts)


def _safe_int(value, default: int = 0) -> int:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return default


def _get_agent_steps(logs, trainer) -> int:
    if logs is not None and "agent_steps" in logs:
        return _safe_int(logs["agent_steps"], 0)
    return _safe_int(getattr(trainer, "global_step", 0), 0)


def _load_policy_state(
    policy: nn.Module,
    state: dict,
    *,
    allow_missing_cycle_target: bool,
    allow_missing_tail_target: bool,
    allow_missing_body_age_target: bool,
    allow_missing_late_head: bool,
) -> None:
    first_layer_key = "features.1.weight"
    target_state = policy.state_dict()
    if first_layer_key in state and first_layer_key in target_state:
        source_weight = state[first_layer_key]
        target_weight = target_state[first_layer_key]
        if (
            isinstance(source_weight, torch.Tensor)
            and isinstance(target_weight, torch.Tensor)
            and source_weight.shape != target_weight.shape
            and source_weight.ndim == 2
            and target_weight.ndim == 2
            and source_weight.shape[0] == target_weight.shape[0]
        ):
            adapted = target_weight.clone()
            adapted.zero_()
            cols = min(source_weight.shape[1], target_weight.shape[1])
            adapted[:, :cols] = source_weight[:, :cols]
            state = dict(state)
            state[first_layer_key] = adapted

    if (
        not allow_missing_cycle_target
        and not allow_missing_tail_target
        and not allow_missing_body_age_target
        and not allow_missing_late_head
    ):
        policy.load_state_dict(state, strict=True)
        return

    incompatible = policy.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed_missing = {
        "cycle_target_decoder.0.weight",
        "cycle_target_decoder.0.bias",
        "cycle_target_decoder.2.weight",
        "cycle_target_decoder.2.bias",
        "tail_target_decoder.0.weight",
        "tail_target_decoder.0.bias",
        "tail_target_decoder.2.weight",
        "tail_target_decoder.2.bias",
        "body_age_target_decoder.0.weight",
        "body_age_target_decoder.0.bias",
        "body_age_target_decoder.2.weight",
        "body_age_target_decoder.2.bias",
        "late_policy_head.0.weight",
        "late_policy_head.0.bias",
        "late_policy_head.2.weight",
        "late_policy_head.2.bias",
        "late_value_head.0.weight",
        "late_value_head.0.bias",
        "late_value_head.2.weight",
        "late_value_head.2.bias",
        "late_value_head.4.weight",
        "late_value_head.4.bias",
    }
    disallowed_missing = [key for key in missing if key not in allowed_missing]
    disallowed_unexpected = [key for key in unexpected if key not in allowed_missing]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "Error(s) in loading state_dict for "
            f"{policy.__class__.__name__}: missing={disallowed_missing} "
            f"unexpected={disallowed_unexpected}"
        )
    if allow_missing_late_head and any(key.startswith("late_") for key in missing):
        sync_late = getattr(policy, "_sync_late_heads_from_base", None)
        if callable(sync_late):
            sync_late()


def _input_layer_shape_mismatch(policy: nn.Module, state: dict) -> bool:
    first_layer_key = "features.1.weight"
    target_state = policy.state_dict()
    if first_layer_key not in state or first_layer_key not in target_state:
        return False
    source_weight = state[first_layer_key]
    target_weight = target_state[first_layer_key]
    if not isinstance(source_weight, torch.Tensor) or not isinstance(target_weight, torch.Tensor):
        return False
    return source_weight.shape != target_weight.shape


def _extract_cycle_action_targets(observations: torch.Tensor, cycle_channel: int) -> tuple[torch.Tensor, torch.Tensor]:
    head = observations[:, 0]
    target = observations[:, cycle_channel]
    flat_head = head.flatten(1)
    flat_target = target.flatten(1)
    has_target = flat_target.amax(dim=1) > 0.5
    if not torch.any(has_target):
        empty = torch.zeros(observations.shape[0], device=observations.device, dtype=torch.bool)
        labels = torch.full((observations.shape[0],), -1, device=observations.device, dtype=torch.long)
        return empty, labels

    obs_n = head.shape[-1]
    head_idx = flat_head.argmax(dim=1)
    target_idx = flat_target.argmax(dim=1)
    head_r, head_c = torch.div(head_idx, obs_n, rounding_mode="floor"), head_idx % obs_n
    target_r, target_c = torch.div(target_idx, obs_n, rounding_mode="floor"), target_idx % obs_n
    dr = target_r - head_r
    dc = target_c - head_c

    labels = torch.full((observations.shape[0],), -1, device=observations.device, dtype=torch.long)
    labels[(dr == 0) & (dc == -1)] = 0  # left
    labels[(dr == -1) & (dc == 0)] = 1  # straight
    labels[(dr == 0) & (dc == 1)] = 2  # right
    valid = has_target & (labels >= 0)
    return valid, labels


def _extract_soft_action_targets(observations: torch.Tensor, start_channel: int) -> tuple[torch.Tensor, torch.Tensor]:
    target = observations[:, start_channel:start_channel + 3]
    probs = target.mean(dim=(2, 3))
    valid = probs.sum(dim=1) > 1e-6
    if torch.any(valid):
        probs = probs.clamp_min(0.0)
        probs_sum = probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        probs = probs / probs_sum
    return valid, probs


def _torch_save_atomic(payload, path: str) -> None:
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _build_resume_state(trainer, policy) -> dict:
    epoch = int(getattr(trainer, "epoch", 0))
    return {
        "model_name": None,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "global_step": int(getattr(trainer, "global_step", 0)),
        "epoch": epoch,
        "update": epoch,
    }


def main():
    parser = argparse.ArgumentParser(description="PufferLib PPO on Snake")
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0, help="0 = auto")
    parser.add_argument(
        "--backend",
        type=str,
        default="mp",
        choices=["mp", "serial"],
        help="Vector backend (default: mp)",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.0,
        help="Final LR ratio for cosine anneal (0.0 = decay to 0, 0.1 ~= 3e-4->3e-5)",
    )
    parser.add_argument("--no-anneal-lr", action="store_true", help="Disable LR annealing")
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=0,
        help="Decay LR over this many steps (0 = use total timesteps). "
             "After decay completes, LR stays at min_lr.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=128, help="Steps per env per epoch")
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=0,
        help="SGD minibatch size (0 = auto; must be divisible by --horizon)",
    )
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--ent-coef-final", type=float, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=0, help="0 = disable")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-deterministic", action="store_true")
    parser.add_argument("--perfect-patience", type=int, default=0, help="0 = disable early stop")
    parser.add_argument("--symmetric", action="store_true", help="Enable symmetric augmentation (50%% horizontal flip)")
    parser.add_argument("--network-scale", type=int, default=1, choices=[1, 2, 4], help="Network width multiplier (1=base, 2=2x, 4=4x)")
    parser.add_argument("--cnn", action="store_true", help="Use CNN policy instead of MLP")
    parser.add_argument("--resnet", action="store_true", help="Use deep ResNet policy (spatial reasoning)")
    parser.add_argument("--iterative-cnn", action="store_true", help="Use weight-tied iterative CNN policy")
    parser.add_argument("--n-iterations", type=int, default=12, help="Iterations for iterative CNN (RF = 2K+1)")
    parser.add_argument("--aux-flood-fill", action="store_true", help="Train auxiliary flood-fill decoder (requires --flood-fill)")
    parser.add_argument("--aux-flood-fill-coef", type=float, default=1.0, help="Weight for auxiliary flood-fill loss")
    parser.add_argument("--aux-cycle-target", action="store_true", help="Train auxiliary decoder to predict the curriculum cycle continuation target")
    parser.add_argument("--aux-cycle-target-coef", type=float, default=1.0, help="Weight for auxiliary curriculum cycle target loss")
    parser.add_argument("--aux-tail-target", action="store_true", help="Train auxiliary decoder to predict the tail position")
    parser.add_argument("--aux-tail-target-coef", type=float, default=1.0, help="Weight for auxiliary tail position loss")
    parser.add_argument("--aux-safe-action-target", action="store_true", help="Train auxiliary target for the late-game safe action heuristic")
    parser.add_argument("--aux-safe-action-target-coef", type=float, default=1.0, help="Weight for auxiliary safe action target loss")
    parser.add_argument("--aux-safe-action-soft-target", action="store_true", help="Train a soft late-game safe action target over the 3 relative actions")
    parser.add_argument("--aux-safe-action-soft-target-coef", type=float, default=1.0, help="Weight for auxiliary soft safe action target loss")
    parser.add_argument("--aux-body-age-target", action="store_true", help="Train auxiliary decoder to predict how long occupied cells will remain blocked")
    parser.add_argument("--aux-body-age-target-coef", type=float, default=1.0, help="Weight for auxiliary body-age target loss")
    parser.add_argument("--aux-cycle-target-min-fill", type=float, default=None, help="Minimum fill level before the auxiliary curriculum cycle target channel activates (default: match --curriculum-follow-min-fill)")
    parser.add_argument("--aux-safe-action-target-min-fill", type=float, default=0.90, help="Minimum fill level before the auxiliary safe action target channel activates")
    parser.add_argument("--aux-safe-action-soft-target-min-fill", type=float, default=0.98, help="Minimum fill level before the auxiliary soft safe action target activates")
    parser.add_argument("--aux-body-age-target-min-fill", type=float, default=0.80, help="Minimum fill level before the auxiliary body-age target channel activates")
    parser.add_argument("--aux-safe-action-fill-weight", type=float, default=500.0, help="Fill weight for the auxiliary safe action heuristic")
    parser.add_argument("--aux-safe-action-soft-temperature", type=float, default=1.0, help="Temperature used to soften late-game safe action scores into a target distribution")
    parser.add_argument("--safe-action-bonus", type=float, default=0.0, help="Small late-game reward bonus for matching the safe action heuristic")
    parser.add_argument("--safe-action-bonus-min-fill", type=float, default=0.95, help="Minimum fill level before the safe action bonus activates")
    parser.add_argument("--safe-action-bonus-fill-weight", type=float, default=500.0, help="Fill weight for the safe action bonus heuristic")
    parser.add_argument("--late-confidence-coef", type=float, default=0.0, help="Auxiliary loss to reduce action entropy on late-game states")
    parser.add_argument("--late-confidence-min-fill", type=float, default=0.95, help="Minimum fill level before late-game confidence sharpening activates")
    parser.add_argument("--late-head-min-fill", type=float, default=None, help="Route late-game states to a dedicated policy/value head starting at this fill fraction")
    parser.add_argument("--train-late-head-only", action="store_true", help="Freeze the incumbent trunk/base heads and train only the dedicated late head")
    parser.add_argument("--elite-bc-coef", type=float, default=0.0, help="Auxiliary behavior cloning loss on saved elite trajectories")
    parser.add_argument("--elite-score-threshold", type=int, default=0, help="Minimum terminal score needed to save a trajectory for elite BC replay (0 = disabled)")
    parser.add_argument("--elite-min-fill", type=float, default=0.90, help="Minimum fill level before transitions are recorded into elite trajectories")
    parser.add_argument("--elite-buffer-size", type=int, default=32768, help="Maximum number of elite transitions kept in replay memory")
    parser.add_argument("--elite-bc-start-steps", type=int, default=0, help="Delay elite BC replay until at least this many agent steps have elapsed")
    parser.add_argument("--elite-safe-action-labels", action="store_true", help="Relabel elite replay actions with the safe-action heuristic before saving")
    parser.add_argument("--elite-safe-action-fill-weight", type=float, default=500.0, help="Fill weight used when relabeling elite replay actions with the safe-action heuristic")
    parser.add_argument("--flood-fill", action="store_true", help="Add flood-fill reachability observation channel")
    parser.add_argument("--body-age-obs", action="store_true", help="Add a late-gated body-age observation channel that marks how long occupied cells remain blocked")
    parser.add_argument("--body-age-obs-min-fill", type=float, default=0.90, help="Minimum fill level before the body-age observation channel activates")
    parser.add_argument("--obs-history", type=int, default=1, help="Number of current+previous encoder-visible observations to stack (current frame first)")
    parser.add_argument("--action-history-obs", type=int, default=0, help="Number of previous relative actions to encode as one-hot broadcast planes")
    parser.add_argument("--curriculum-prob", type=float, default=0.0, help="Fraction of episodes starting at random fill level (0 = disabled)")
    parser.add_argument("--curriculum-min-fill", type=float, default=0.5, help="Minimum fill level for curriculum spawning")
    parser.add_argument("--curriculum-max-fill", type=float, default=0.85, help="Maximum fill level for curriculum spawning")
    parser.add_argument("--curriculum-follow-bonus", type=float, default=0.0, help="Small reward bonus/penalty for following the sampled curriculum Hamiltonian cycle")
    parser.add_argument("--curriculum-follow-min-fill", type=float, default=0.85, help="Minimum fill level before curriculum follow bonus activates")
    parser.add_argument("--head-centered", action="store_true", default=False, help="Head-centered observation (39x39 for 20x20 board)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--vf-clip-coef", type=float, default=0.2, help="Value function clip coefficient")
    parser.add_argument("--stall-penalty", type=float, default=-1.0, help="Penalty for stalling (default: -1.0, same as death)")
    parser.add_argument("--stall-terminates", action="store_true", default=True, help="Stall ends episode (terminated=True, not truncated)")
    parser.add_argument("--no-stall-terminates", action="store_false", dest="stall_terminates", help="Stall truncates instead of terminates")
    parser.add_argument("--max-no-food-base", type=int, default=None, help="Override max steps without food (default: dynamic based on length)")
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name (default: auto-generated)")
    parser.add_argument("--data-dir", type=str, default="experiments")
    parser.add_argument("--prio-alpha", type=float, default=0.8)
    parser.add_argument("--prio-beta0", type=float, default=0.2)
    parser.add_argument("--resume", type=str, default=None, help="Path to a saved .pt state_dict")
    parser.add_argument(
        "--resume-state",
        type=str,
        default=None,
        help="Path to trainer_state.pt (restores optimizer, epoch, global_step)",
    )
    parser.add_argument(
        "--resume-add-steps",
        action="store_true",
        help="When resuming, add --timesteps to the saved global_step",
    )
    parser.add_argument(
        "--override-resume-lr",
        action="store_true",
        help="When resuming, force optimizer LR to --lr after loading optimizer state",
    )
    parser.add_argument(
        "--policy-kl-anchor-path",
        type=str,
        default=None,
        help="Path to a reference policy checkpoint used for an explicit KL anchor during continuation",
    )
    parser.add_argument(
        "--policy-kl-anchor-coef",
        type=float,
        default=0.0,
        help="Weight for KL(reference || current) on rollout states after each PPO update",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Enable PufferLib rich dashboard output (off by default for benchmarking)",
    )
    args = parser.parse_args()

    if args.num_envs < 1:
        raise SystemExit("--num-envs must be >= 1")
    if args.horizon < 1:
        raise SystemExit("--horizon must be >= 1")
    if args.aux_flood_fill and not args.flood_fill:
        raise SystemExit("--aux-flood-fill requires --flood-fill (for ground truth)")
    if args.aux_flood_fill and not args.iterative_cnn:
        # MLP with aux flood-fill decoder - supported
        pass
    if args.aux_cycle_target and args.curriculum_prob <= 0.0:
        raise SystemExit("--aux-cycle-target requires --curriculum-prob > 0")
    if args.aux_cycle_target and (args.iterative_cnn or args.cnn or args.resnet):
        raise SystemExit("--aux-cycle-target is currently supported for the default MLP policy only")
    if args.aux_tail_target and (args.iterative_cnn or args.cnn or args.resnet):
        raise SystemExit("--aux-tail-target is currently supported for the default MLP policy only")
    if args.aux_safe_action_target and (args.iterative_cnn or args.cnn or args.resnet):
        raise SystemExit("--aux-safe-action-target is currently supported for the default MLP policy only")
    if args.aux_safe_action_soft_target and (args.iterative_cnn or args.cnn or args.resnet):
        raise SystemExit("--aux-safe-action-soft-target is currently supported for the default MLP policy only")
    if args.aux_body_age_target and (args.iterative_cnn or args.cnn or args.resnet):
        raise SystemExit("--aux-body-age-target is currently supported for the default MLP policy only")
    if args.policy_kl_anchor_coef < 0.0:
        raise SystemExit("--policy-kl-anchor-coef must be >= 0")
    if args.policy_kl_anchor_path is not None and not os.path.exists(args.policy_kl_anchor_path):
        raise SystemExit(f"--policy-kl-anchor-path not found: {args.policy_kl_anchor_path}")
    if args.aux_cycle_target_min_fill is not None and not (0.0 <= args.aux_cycle_target_min_fill <= 1.0):
        raise SystemExit("--aux-cycle-target-min-fill must be in [0, 1]")
    if not (0.0 <= args.aux_safe_action_target_min_fill <= 1.0):
        raise SystemExit("--aux-safe-action-target-min-fill must be in [0, 1]")
    if not (0.0 <= args.aux_safe_action_soft_target_min_fill <= 1.0):
        raise SystemExit("--aux-safe-action-soft-target-min-fill must be in [0, 1]")
    if not (0.0 <= args.aux_body_age_target_min_fill <= 1.0):
        raise SystemExit("--aux-body-age-target-min-fill must be in [0, 1]")
    if args.aux_safe_action_soft_temperature <= 0.0:
        raise SystemExit("--aux-safe-action-soft-temperature must be > 0")
    if not (0.0 <= args.body_age_obs_min_fill <= 1.0):
        raise SystemExit("--body-age-obs-min-fill must be in [0, 1]")
    if args.obs_history < 1:
        raise SystemExit("--obs-history must be >= 1")
    if args.action_history_obs < 0:
        raise SystemExit("--action-history-obs must be >= 0")
    if not (0.0 <= args.safe_action_bonus_min_fill <= 1.0):
        raise SystemExit("--safe-action-bonus-min-fill must be in [0, 1]")
    if not (0.0 <= args.late_confidence_min_fill <= 1.0):
        raise SystemExit("--late-confidence-min-fill must be in [0, 1]")
    if args.late_head_min_fill is not None and not (0.0 <= args.late_head_min_fill <= 1.0):
        raise SystemExit("--late-head-min-fill must be in [0, 1]")
    if args.train_late_head_only and args.late_head_min_fill is None:
        raise SystemExit("--train-late-head-only requires --late-head-min-fill")
    if not (0.0 <= args.elite_min_fill <= 1.0):
        raise SystemExit("--elite-min-fill must be in [0, 1]")
    if args.elite_score_threshold < 0:
        raise SystemExit("--elite-score-threshold must be >= 0")
    if args.elite_buffer_size < 1:
        raise SystemExit("--elite-buffer-size must be >= 1")

    resume_state = None
    resume_steps = 0
    resume_epoch = 0
    resume_state_path = args.resume_state
    if resume_state_path is not None:
        if not os.path.exists(resume_state_path):
            raise SystemExit(f"--resume-state not found: {resume_state_path}")
        try:
            resume_state = torch.load(resume_state_path, map_location="cpu")
        except Exception as exc:
            raise SystemExit(f"--resume-state load failed: {exc}") from exc
        resume_steps = int(resume_state.get("global_step", 0))
        resume_epoch = int(resume_state.get("epoch", resume_state.get("update", 0)))

    backend = pufferlib.vector.Multiprocessing if args.backend == "mp" else pufferlib.vector.Serial
    num_workers = int(args.num_workers)
    if backend is pufferlib.vector.Multiprocessing:
        if num_workers <= 0:
            num_workers = _auto_num_workers(args.num_envs)
        if args.num_envs % num_workers != 0:
            raise SystemExit(
                f"--num-envs ({args.num_envs}) must be divisible by --num-workers ({num_workers})"
            )

    torch.set_float32_matmul_precision("high")
    try:
        physical = psutil.cpu_count(logical=False) or 1
    except Exception:
        physical = 1
    torch.set_num_threads(max(1, physical - (num_workers or 0)))

    elite_dir = None
    if args.elite_bc_coef > 0 and args.elite_score_threshold > 0:
        elite_run_tag = args.exp_name or f"elite_{int(time.time() * 1_000_000)}"
        elite_dir = os.path.join(
            args.data_dir,
            "_elite",
            f"{elite_run_tag}_{int(time.time() * 1_000_000)}",
        )
        os.makedirs(elite_dir, exist_ok=True)

    env_kwargs = dict(
        n=args.board_size,
        gamma=args.gamma,
        alpha=args.alpha,
        symmetric=args.symmetric,
        stall_penalty=args.stall_penalty,
        stall_terminates=args.stall_terminates,
        max_no_food_base=args.max_no_food_base,
        flood_fill_obs=args.flood_fill,
        body_age_obs=args.body_age_obs,
        obs_history=args.obs_history,
        action_history_obs=args.action_history_obs,
        curriculum_prob=args.curriculum_prob,
        curriculum_min_fill=args.curriculum_min_fill,
        curriculum_max_fill=args.curriculum_max_fill,
        curriculum_follow_bonus=args.curriculum_follow_bonus,
        curriculum_follow_min_fill=args.curriculum_follow_min_fill,
        cycle_target_obs=args.aux_cycle_target,
        tail_target_obs=args.aux_tail_target,
        safe_action_target_obs=args.aux_safe_action_target,
        safe_action_soft_target_obs=args.aux_safe_action_soft_target,
        body_age_target_obs=args.aux_body_age_target,
        cycle_target_min_fill=args.aux_cycle_target_min_fill,
        safe_action_target_min_fill=args.aux_safe_action_target_min_fill,
        safe_action_soft_target_min_fill=args.aux_safe_action_soft_target_min_fill,
        body_age_obs_min_fill=args.body_age_obs_min_fill,
        body_age_target_min_fill=args.aux_body_age_target_min_fill,
        safe_action_fill_weight=args.aux_safe_action_fill_weight,
        safe_action_soft_temperature=args.aux_safe_action_soft_temperature,
        safe_action_bonus=args.safe_action_bonus,
        safe_action_bonus_min_fill=args.safe_action_bonus_min_fill,
        safe_action_bonus_fill_weight=args.safe_action_bonus_fill_weight,
        elite_dir=elite_dir,
        elite_score_threshold=(args.elite_score_threshold if args.elite_score_threshold > 0 else None),
        elite_min_fill=args.elite_min_fill,
        elite_safe_action_labels=args.elite_safe_action_labels,
        elite_safe_action_fill_weight=args.elite_safe_action_fill_weight,
        head_centered=args.head_centered,
    )
    vec_kwargs = dict(
        num_envs=args.num_envs,
        seed=args.seed,
        backend=backend,
        env_kwargs=env_kwargs,
    )
    if backend is pufferlib.vector.Multiprocessing:
        vec_kwargs["num_workers"] = num_workers
    vecenv = pufferlib.vector.make(make_snake_env, **vec_kwargs)

    if args.iterative_cnn:
        policy = SnakeIterativeCNNPolicy(
            vecenv.driver_env, scale=args.network_scale,
            n_iterations=args.n_iterations, aux_flood_fill=args.aux_flood_fill,
        ).to(args.device)
    elif args.resnet:
        policy = SnakeResNetPolicy(vecenv.driver_env, scale=args.network_scale).to(args.device)
    elif args.cnn:
        policy = SnakeCNNPolicy(vecenv.driver_env, scale=args.network_scale).to(args.device)
    else:
        policy = SnakePolicy(
            vecenv.driver_env,
            scale=args.network_scale,
            aux_flood_fill=args.aux_flood_fill,
            aux_cycle_target=args.aux_cycle_target,
            aux_tail_target=args.aux_tail_target,
            aux_safe_action_target=args.aux_safe_action_target,
            aux_safe_action_soft_target=args.aux_safe_action_soft_target,
            aux_body_age_target=args.aux_body_age_target,
            board_size=args.board_size,
            head_centered=args.head_centered,
            late_head_min_fill=args.late_head_min_fill,
        ).to(args.device)

    resume_optimizer_compatible = True
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        _load_policy_state(
                policy,
                state,
                allow_missing_cycle_target=args.aux_cycle_target,
                allow_missing_tail_target=args.aux_tail_target,
                allow_missing_body_age_target=args.aux_body_age_target,
                allow_missing_late_head=args.late_head_min_fill is not None,
            )
    elif resume_state is not None:
        inline_state = resume_state.get("model_state_dict")
        model_name = resume_state.get("model_name")
        if inline_state is not None:
            if _input_layer_shape_mismatch(policy, inline_state):
                resume_optimizer_compatible = False
            _load_policy_state(
                    policy,
                    inline_state,
                    allow_missing_cycle_target=args.aux_cycle_target,
                    allow_missing_tail_target=args.aux_tail_target,
                    allow_missing_body_age_target=args.aux_body_age_target,
                    allow_missing_late_head=args.late_head_min_fill is not None,
                )
        elif model_name:
            resume_model = os.path.join(os.path.dirname(resume_state_path), model_name)
            if os.path.exists(resume_model):
                state = torch.load(resume_model, map_location="cpu")
                if _input_layer_shape_mismatch(policy, state):
                    resume_optimizer_compatible = False
                _load_policy_state(
                    policy,
                    state,
                    allow_missing_cycle_target=args.aux_cycle_target,
                    allow_missing_tail_target=args.aux_tail_target,
                    allow_missing_body_age_target=args.aux_body_age_target,
                    allow_missing_late_head=args.late_head_min_fill is not None,
                )
            else:
                print(f"warning: resume model not found: {resume_model}", file=sys.stderr)

    if args.train_late_head_only:
        policy._late_head_grad_hooks = []
        for name, param in policy.named_parameters():
            if name.startswith("late_policy_head") or name.startswith("late_value_head"):
                continue
            policy._late_head_grad_hooks.append(
                param.register_hook(lambda grad: torch.zeros_like(grad))
            )

    reference_policy = None
    policy_kl_anchor_active = args.policy_kl_anchor_path is not None and args.policy_kl_anchor_coef > 0.0
    if policy_kl_anchor_active:
        reference_policy = copy.deepcopy(policy).to(args.device)
        anchor_state = torch.load(args.policy_kl_anchor_path, map_location="cpu")
        _load_policy_state(
            reference_policy,
            anchor_state,
            allow_missing_cycle_target=args.aux_cycle_target,
            allow_missing_tail_target=args.aux_tail_target,
            allow_missing_body_age_target=args.aux_body_age_target,
            allow_missing_late_head=args.late_head_min_fill is not None,
        )
        reference_policy.eval()
        for param in reference_policy.parameters():
            param.requires_grad_(False)

    batch_size = args.num_envs * args.horizon
    if args.minibatch_size > 0:
        if args.minibatch_size % args.horizon != 0:
            raise SystemExit("--minibatch-size must be divisible by --horizon")
        minibatch_size = min(int(args.minibatch_size), int(batch_size))
    else:
        minibatch_segments = max(1, 256 // args.horizon)
        minibatch_size = minibatch_segments * args.horizon
        if minibatch_size > batch_size:
            minibatch_size = batch_size

    total_timesteps = int(args.timesteps)
    if resume_state is not None and args.resume_add_steps:
        total_timesteps = resume_steps + total_timesteps
    elif resume_state is not None and resume_steps >= total_timesteps:
        print(
            f"warning: resume global_step ({resume_steps}) >= total_timesteps ({total_timesteps})",
            file=sys.stderr,
        )

    config = {
        "env": args.exp_name if args.exp_name else f"snake_{args.board_size}",
        "seed": args.seed,
        "torch_deterministic": True,
        "cpu_offload": False,
        "device": args.device,
        "optimizer": "adam",
        "precision": "float32",
        "total_timesteps": int(total_timesteps),
        "learning_rate": float(args.lr),
        "anneal_lr": not bool(args.no_anneal_lr),
        "min_lr_ratio": float(args.min_lr_ratio),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "update_epochs": int(args.update_epochs),
        "clip_coef": 0.1,
        "vf_coef": 1.0,
        "vf_clip_coef": float(args.vf_clip_coef),
        "max_grad_norm": 0.5,
        "ent_coef": float(args.ent_coef),
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "batch_size": int(batch_size),
        "minibatch_size": int(minibatch_size),
        "max_minibatch_size": 32768,
        "bptt_horizon": int(args.horizon),
        "compile": False,
        "compile_mode": "max-autotune-no-cudagraphs",
        "compile_fullgraph": True,
        "vtrace_rho_clip": 1.0,
        "vtrace_c_clip": 1.0,
        "prio_alpha": float(args.prio_alpha),
        "prio_beta0": float(args.prio_beta0),
        "use_rnn": False,
        "checkpoint_interval": int(args.checkpoint_interval),
        "data_dir": str(args.data_dir),
    }

    trainer = pufferl.PuffeRL(config, vecenv, policy)
    if not args.dashboard:
        trainer.print_dashboard = lambda *_, **__: None

    # Override scheduler if --lr-decay-steps is set
    if args.lr_decay_steps > 0:
        import math
        batch_size = config["batch_size"]
        decay_epochs = args.lr_decay_steps // batch_size
        min_ratio = float(args.min_lr_ratio)

        def lr_lambda(epoch):
            if epoch >= decay_epochs:
                return min_ratio
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * epoch / decay_epochs))

        trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer, lr_lambda=lr_lambda,
        )
        min_lr = float(args.lr) * min_ratio
        print(
            f"LR schedule: cosine decay over {args.lr_decay_steps/1e6:.0f}M steps "
            f"({decay_epochs} epochs), then constant at {min_lr:.1e}",
            file=sys.stderr,
        )

    if resume_state is not None:
        if "optimizer_state_dict" in resume_state and resume_optimizer_compatible:
            try:
                trainer.optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            except ValueError as exc:
                print(
                    "Warning: optimizer state resume skipped due to parameter mismatch: "
                    f"{exc}",
                    file=sys.stderr,
                )
        elif "optimizer_state_dict" in resume_state and not resume_optimizer_compatible:
            print(
                "Warning: optimizer state resume skipped due to adapted input layer shape",
                file=sys.stderr,
            )
        if args.override_resume_lr:
            for group in trainer.optimizer.param_groups:
                group["lr"] = float(args.lr)

        trainer.global_step = resume_steps
        trainer.epoch = resume_epoch
        trainer.last_log_step = resume_steps
        trainer.last_log_time = time.time()

        # Properly restore or recreate scheduler
        if args.resume_add_steps:
            if args.no_anneal_lr and args.lr_decay_steps <= 0:
                # Constant-LR runs should preserve the optimizer LR directly.
                try:
                    trainer.scheduler.last_epoch = resume_epoch
                except Exception:
                    pass
            else:
                # When extending an annealed run, recreate the schedule with
                # the new horizon and continue from the resumed epoch.
                trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    trainer.optimizer,
                    T_max=trainer.total_epochs,
                    last_epoch=resume_epoch - 1,  # -1 because step() will increment
                )
                trainer.scheduler.step()
        elif "scheduler_state_dict" in resume_state:
            # Normal resume - restore exact scheduler state
            try:
                trainer.scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            except Exception as exc:
                print(
                    "Warning: scheduler state resume skipped due to mismatch: "
                    f"{exc}",
                    file=sys.stderr,
                )
        else:
            # Fallback: just set last_epoch (may not work correctly)
            try:
                trainer.scheduler.last_epoch = resume_epoch
            except Exception:
                pass

        if args.override_resume_lr:
            if hasattr(trainer.scheduler, "base_lrs"):
                trainer.scheduler.base_lrs = [float(args.lr)] * len(trainer.scheduler.base_lrs)
            if hasattr(trainer.scheduler, "_last_lr"):
                trainer.scheduler._last_lr = [float(args.lr)] * len(trainer.optimizer.param_groups)

        lr = trainer.optimizer.param_groups[0]["lr"]
        scheduler_t_max = getattr(trainer.scheduler, "T_max", None)
        scheduler_desc = (
            f"{type(trainer.scheduler).__name__}(T_max={scheduler_t_max})"
            if scheduler_t_max is not None
            else type(trainer.scheduler).__name__
        )
        print(
            f"resume_state: steps={resume_steps} epoch={resume_epoch} "
            f"lr={lr:.2e} scheduler={scheduler_desc}",
            file=sys.stderr,
        )

    tracker = None
    try:
        tracker = ExperimentTracker(
            exp_name=config["env"],
            run_id=trainer.logger.run_id,
            data_dir=config["data_dir"],
            args=vars(args),
            config=config,
            command=_format_command(),
            cwd=os.getcwd(),
        )
    except Exception as exc:
        print(f"experiment_tracker_disabled: {exc}", file=sys.stderr)

    # Auxiliary training targets live in the observation but are excluded
    # from the policy encoder, so they supervise training without leaking
    # teacher signals into action selection.
    aux_flood_fill_active = args.aux_flood_fill and args.aux_flood_fill_coef > 0
    aux_cycle_target_active = args.aux_cycle_target and args.aux_cycle_target_coef > 0
    aux_tail_target_active = args.aux_tail_target and args.aux_tail_target_coef > 0
    aux_safe_action_target_active = args.aux_safe_action_target and args.aux_safe_action_target_coef > 0
    aux_safe_action_soft_target_active = args.aux_safe_action_soft_target and args.aux_safe_action_soft_target_coef > 0
    aux_body_age_target_active = args.aux_body_age_target and args.aux_body_age_target_coef > 0
    late_confidence_active = args.late_confidence_coef > 0
    elite_bc_active = args.elite_bc_coef > 0 and args.elite_score_threshold > 0
    if aux_flood_fill_active:
        obs_shape = vecenv.driver_env.observation_space.shape
        print(f"aux_flood_fill: coef={args.aux_flood_fill_coef}, "
              f"encoder_channels={policy.encoder_channels}, "
              f"total_channels={obs_shape[0]}", file=sys.stderr)
    if aux_cycle_target_active:
        obs_shape = vecenv.driver_env.observation_space.shape
        print(f"aux_cycle_target: coef={args.aux_cycle_target_coef}, "
              f"encoder_channels={policy.encoder_channels}, "
              f"total_channels={obs_shape[0]}", file=sys.stderr)
    if aux_tail_target_active:
        obs_shape = vecenv.driver_env.observation_space.shape
        print(f"aux_tail_target: coef={args.aux_tail_target_coef}, "
              f"encoder_channels={policy.encoder_channels}, "
              f"total_channels={obs_shape[0]}", file=sys.stderr)
    if aux_safe_action_target_active:
        obs_shape = vecenv.driver_env.observation_space.shape
        print(f"aux_safe_action_target: coef={args.aux_safe_action_target_coef}, "
              f"encoder_channels={policy.encoder_channels}, "
              f"total_channels={obs_shape[0]}", file=sys.stderr)
    if aux_safe_action_soft_target_active:
        obs_shape = vecenv.driver_env.observation_space.shape
        print(f"aux_safe_action_soft_target: coef={args.aux_safe_action_soft_target_coef}, "
              f"encoder_channels={policy.encoder_channels}, "
              f"total_channels={obs_shape[0]}, "
              f"min_fill={args.aux_safe_action_soft_target_min_fill}, "
              f"temp={args.aux_safe_action_soft_temperature}", file=sys.stderr)
    if aux_body_age_target_active:
        obs_shape = vecenv.driver_env.observation_space.shape
        print(f"aux_body_age_target: coef={args.aux_body_age_target_coef}, "
              f"encoder_channels={policy.encoder_channels}, "
              f"total_channels={obs_shape[0]}, "
              f"min_fill={args.aux_body_age_target_min_fill}", file=sys.stderr)
    if late_confidence_active:
        print(
            f"late_confidence: coef={args.late_confidence_coef}, "
            f"min_fill={args.late_confidence_min_fill}",
            file=sys.stderr,
        )
    if args.late_head_min_fill is not None:
        print(f"late_head: min_fill={args.late_head_min_fill}", file=sys.stderr)
    if args.train_late_head_only:
        print("late_head: training late head only", file=sys.stderr)
    if elite_bc_active:
        print(
            f"elite_bc: coef={args.elite_bc_coef}, "
            f"score_threshold={args.elite_score_threshold}, "
            f"min_fill={args.elite_min_fill}, "
            f"start_steps={args.elite_bc_start_steps}, "
            f"safe_labels={args.elite_safe_action_labels}",
            file=sys.stderr,
        )

    start_time = time.time()
    last_logs = None
    last_eval_at = resume_steps if resume_state is not None else 0
    perfect_streak = 0
    last_aux_flood_loss = None
    last_aux_cycle_loss = None
    last_aux_tail_loss = None
    last_aux_safe_action_loss = None
    last_aux_safe_action_soft_loss = None
    last_aux_body_age_loss = None
    last_late_confidence_loss = None
    last_elite_bc_loss = None
    last_policy_kl_anchor_loss = None
    elite_obs_buffer: deque[np.ndarray] = deque(maxlen=args.elite_buffer_size)
    elite_action_buffer: deque[int] = deque(maxlen=args.elite_buffer_size)
    seen_elite_paths: set[str] = set()
    while trainer.epoch < trainer.total_epochs:
        trainer.evaluate()

        if elite_bc_active:
            elite_paths = trainer.stats.get("elite_path", [])
            loaded_transitions = 0
            for path in elite_paths:
                if not isinstance(path, str) or path in seen_elite_paths or not os.path.exists(path):
                    continue
                try:
                    with np.load(path) as data:
                        obs_arr = data["observations"]
                        act_arr = data["actions"]
                    if len(obs_arr) == len(act_arr):
                        for obs_item, act_item in zip(obs_arr, act_arr):
                            elite_obs_buffer.append(np.asarray(obs_item, dtype=np.float32))
                            elite_action_buffer.append(int(act_item))
                            loaded_transitions += 1
                except Exception:
                    pass
                finally:
                    seen_elite_paths.add(path)
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            trainer.stats.pop("elite_path", None)
            if loaded_transitions > 0:
                print(
                    f"elite_buffer_loaded: +{loaded_transitions} transitions "
                    f"(buffer={len(elite_obs_buffer)})",
                    file=sys.stderr,
                )

        logs = trainer.train()

        aux_flood_loss = None
        aux_cycle_loss = None
        aux_tail_loss = None
        aux_safe_action_loss = None
        aux_safe_action_soft_loss = None
        aux_body_age_loss = None
        late_confidence_loss = None
        elite_bc_loss = None

        # Auxiliary training step
        if (
            aux_flood_fill_active
            or aux_cycle_target_active
            or aux_tail_target_active
            or aux_safe_action_target_active
            or aux_safe_action_soft_target_active
            or aux_body_age_target_active
            or late_confidence_active
            or elite_bc_active
            or policy_kl_anchor_active
        ) and logs is not None:
            obs_buf = trainer.observations
            # Reshape from (segments, horizon, C, H, W) to (N, C, H, W)
            flat_obs = obs_buf.reshape(-1, *obs_buf.shape[2:])
            n_total = flat_obs.shape[0]
            aux_batch = min(2048, n_total)
            idx = torch.randperm(n_total, device=flat_obs.device)[:aux_batch]
            mb = flat_obs[idx]
            if mb.device != torch.device(args.device):
                mb = mb.to(args.device)

            enc_input = mb[:, :policy.encoder_channels]
            aux_total_loss = None

            if aux_flood_fill_active:
                flood_target = mb[:, policy.encoder_channels:policy.encoder_channels + 1]
                if policy.head_centered:
                    flood_target_inner = flood_target
                else:
                    flood_target_inner = flood_target[:, :, 1:-1, 1:-1]
                flood_pred = policy.forward_flood_predict(enc_input)
                aux_flood_loss = F.binary_cross_entropy_with_logits(flood_pred, flood_target_inner)
                aux_total_loss = aux_flood_loss * args.aux_flood_fill_coef

            if aux_cycle_target_active:
                cycle_channel = policy.encoder_channels + int(args.aux_flood_fill)
                cycle_valid, cycle_labels = _extract_cycle_action_targets(mb, cycle_channel)
                if torch.any(cycle_valid):
                    cycle_mb = mb[cycle_valid]
                    cycle_labels = cycle_labels[cycle_valid]
                    cycle_logits, _ = policy.forward_eval(cycle_mb)
                    aux_cycle_loss = F.cross_entropy(cycle_logits, cycle_labels)
                    cycle_weighted = aux_cycle_loss * args.aux_cycle_target_coef
                    aux_total_loss = cycle_weighted if aux_total_loss is None else aux_total_loss + cycle_weighted

            if aux_tail_target_active:
                tail_channel = (
                    policy.encoder_channels
                    + int(args.aux_flood_fill)
                    + int(args.aux_cycle_target)
                )
                tail_target = mb[:, tail_channel:tail_channel + 1]
                if policy.head_centered:
                    tail_target_inner = tail_target
                else:
                    tail_target_inner = tail_target[:, :, 1:-1, 1:-1]
                tail_pred = policy.forward_tail_target_predict(enc_input)
                aux_tail_loss = F.binary_cross_entropy_with_logits(tail_pred, tail_target_inner)
                tail_weighted = aux_tail_loss * args.aux_tail_target_coef
                aux_total_loss = tail_weighted if aux_total_loss is None else aux_total_loss + tail_weighted

            if aux_safe_action_target_active:
                safe_channel = (
                    policy.encoder_channels
                    + int(args.aux_flood_fill)
                    + int(args.aux_cycle_target)
                    + int(args.aux_tail_target)
                )
                safe_valid, safe_labels = _extract_cycle_action_targets(mb, safe_channel)
                if torch.any(safe_valid):
                    safe_mb = mb[safe_valid]
                    safe_labels = safe_labels[safe_valid]
                    safe_logits, _ = policy.forward_eval(safe_mb)
                    aux_safe_action_loss = F.cross_entropy(safe_logits, safe_labels)
                    safe_weighted = aux_safe_action_loss * args.aux_safe_action_target_coef
                    aux_total_loss = safe_weighted if aux_total_loss is None else aux_total_loss + safe_weighted

            if aux_safe_action_soft_target_active:
                soft_channel = (
                    policy.encoder_channels
                    + int(args.aux_flood_fill)
                    + int(args.aux_cycle_target)
                    + int(args.aux_tail_target)
                    + int(args.aux_safe_action_target)
                )
                soft_valid, soft_targets = _extract_soft_action_targets(mb, soft_channel)
                if torch.any(soft_valid):
                    soft_mb = mb[soft_valid]
                    soft_targets = soft_targets[soft_valid]
                    soft_logits, _ = policy.forward_eval(soft_mb)
                    aux_safe_action_soft_loss = F.kl_div(
                        F.log_softmax(soft_logits, dim=-1),
                        soft_targets,
                        reduction="batchmean",
                    )
                    soft_weighted = aux_safe_action_soft_loss * args.aux_safe_action_soft_target_coef
                    aux_total_loss = soft_weighted if aux_total_loss is None else aux_total_loss + soft_weighted

            if aux_body_age_target_active:
                body_age_channel = (
                    policy.encoder_channels
                    + int(args.aux_flood_fill)
                    + int(args.aux_cycle_target)
                    + int(args.aux_tail_target)
                    + int(args.aux_safe_action_target)
                    + 3 * int(args.aux_safe_action_soft_target)
                )
                body_age_target = mb[:, body_age_channel:body_age_channel + 1]
                if policy.head_centered:
                    body_age_target_inner = body_age_target
                else:
                    body_age_target_inner = body_age_target[:, :, 1:-1, 1:-1]
                body_age_valid = body_age_target_inner.flatten(1).amax(dim=1) > 0.0
                if torch.any(body_age_valid):
                    body_age_enc = enc_input[body_age_valid]
                    body_age_target_inner = body_age_target_inner[body_age_valid]
                    body_age_pred = torch.sigmoid(policy.forward_body_age_target_predict(body_age_enc))
                    body_age_weights = 1.0 + 3.0 * (body_age_target_inner > 0.0).float()
                    aux_body_age_loss = (
                        body_age_weights * (body_age_pred - body_age_target_inner).pow(2)
                    ).sum() / body_age_weights.sum().clamp_min(1.0)
                    body_age_weighted = aux_body_age_loss * args.aux_body_age_target_coef
                    aux_total_loss = body_age_weighted if aux_total_loss is None else aux_total_loss + body_age_weighted

            if late_confidence_active:
                fill_fraction = mb[:, 1].sum(dim=(1, 2)) / float(policy.board_size * policy.board_size)
                late_mask = fill_fraction >= args.late_confidence_min_fill
                if torch.any(late_mask):
                    late_mb = mb[late_mask]
                    late_logits, _ = policy.forward_eval(late_mb)
                    late_log_probs = F.log_softmax(late_logits, dim=-1)
                    late_probs = late_log_probs.exp()
                    late_confidence_loss = -(late_probs * late_log_probs).sum(dim=-1).mean()
                    confidence_weighted = late_confidence_loss * args.late_confidence_coef
                    aux_total_loss = confidence_weighted if aux_total_loss is None else aux_total_loss + confidence_weighted

            current_agent_steps = _get_agent_steps(logs, trainer)

            if (
                elite_bc_active
                and elite_obs_buffer
                and current_agent_steps >= args.elite_bc_start_steps
            ):
                elite_batch = min(512, len(elite_obs_buffer))
                elite_indices = np.random.choice(len(elite_obs_buffer), size=elite_batch, replace=False)
                elite_obs = np.stack([elite_obs_buffer[int(i)] for i in elite_indices]).astype(np.float32)
                elite_actions = np.asarray([elite_action_buffer[int(i)] for i in elite_indices], dtype=np.int64)
                elite_obs_t = torch.as_tensor(elite_obs, device=args.device, dtype=torch.float32)
                elite_actions_t = torch.as_tensor(elite_actions, device=args.device, dtype=torch.long)
                elite_logits, _ = policy.forward_eval(elite_obs_t)
                elite_bc_loss = F.cross_entropy(elite_logits, elite_actions_t)
                elite_weighted = elite_bc_loss * args.elite_bc_coef
                aux_total_loss = elite_weighted if aux_total_loss is None else aux_total_loss + elite_weighted

            if policy_kl_anchor_active and reference_policy is not None:
                current_logits, _ = policy.forward_eval(mb)
                with torch.no_grad():
                    anchor_logits, _ = reference_policy.forward_eval(mb)
                policy_kl_anchor_loss = F.kl_div(
                    F.log_softmax(current_logits, dim=-1),
                    F.softmax(anchor_logits, dim=-1),
                    reduction="batchmean",
                )
                kl_weighted = policy_kl_anchor_loss * args.policy_kl_anchor_coef
                aux_total_loss = kl_weighted if aux_total_loss is None else aux_total_loss + kl_weighted
            else:
                policy_kl_anchor_loss = None

            if aux_total_loss is not None:
                trainer.optimizer.zero_grad()
                aux_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                trainer.optimizer.step()

            last_aux_flood_loss = aux_flood_loss.item() if aux_flood_loss is not None else None
            last_aux_cycle_loss = aux_cycle_loss.item() if aux_cycle_loss is not None else None
            last_aux_tail_loss = aux_tail_loss.item() if aux_tail_loss is not None else None
            last_aux_safe_action_loss = (
                aux_safe_action_loss.item() if aux_safe_action_loss is not None else None
            )
            last_aux_safe_action_soft_loss = (
                aux_safe_action_soft_loss.item() if aux_safe_action_soft_loss is not None else None
            )
            last_aux_body_age_loss = (
                aux_body_age_loss.item() if aux_body_age_loss is not None else None
            )
            last_late_confidence_loss = (
                late_confidence_loss.item() if late_confidence_loss is not None else None
            )
            last_elite_bc_loss = elite_bc_loss.item() if elite_bc_loss is not None else None
            last_policy_kl_anchor_loss = (
                policy_kl_anchor_loss.item() if policy_kl_anchor_loss is not None else None
            )

        if logs is not None:
            last_logs = logs
            sps = logs.get("SPS", 0)
            agent_steps = _get_agent_steps(logs, trainer)
            if tracker is not None:
                tracker.log_train(logs)
            ep_ret = logs.get("environment/episode_return", None)
            ep_len = logs.get("environment/episode_length", None)
            ep_score = logs.get("environment/episode_score", None)
            win_rate = logs.get("environment/episode_win", None)
            train_win_rate = _safe_float(win_rate, 0.0) or 0.0
            train_win = train_win_rate > 0.0

            if args.ent_coef_final is not None and args.timesteps > 0:
                progress = min(1.0, agent_steps / float(args.timesteps))
                ent = float(args.ent_coef + progress * (float(args.ent_coef_final) - float(args.ent_coef)))
                trainer.config["ent_coef"] = ent

            extra = []
            if ep_ret is not None:
                extra.append(f"ep_ret={ep_ret:.2f}")
            if ep_len is not None:
                extra.append(f"ep_len={ep_len:.1f}")
            if ep_score is not None:
                extra.append(f"ep_score={ep_score:.2f}")
            if win_rate is not None:
                extra.append(f"win={win_rate*100:.1f}%")
            if last_aux_flood_loss is not None:
                extra.append(f"aux_ff={last_aux_flood_loss:.4f}")
            if last_aux_cycle_loss is not None:
                extra.append(f"aux_cycle={last_aux_cycle_loss:.4f}")
            if last_aux_tail_loss is not None:
                extra.append(f"aux_tail={last_aux_tail_loss:.4f}")
            if last_aux_safe_action_loss is not None:
                extra.append(f"aux_safe={last_aux_safe_action_loss:.4f}")
            if last_aux_safe_action_soft_loss is not None:
                extra.append(f"aux_softsafe={last_aux_safe_action_soft_loss:.4f}")
            if last_aux_body_age_loss is not None:
                extra.append(f"aux_body_age={last_aux_body_age_loss:.4f}")
            if last_late_confidence_loss is not None:
                extra.append(f"aux_conf={last_late_confidence_loss:.4f}")
            if last_elite_bc_loss is not None:
                extra.append(f"elite_bc={last_elite_bc_loss:.4f}")
            if last_policy_kl_anchor_loss is not None:
                extra.append(f"anchor_kl={last_policy_kl_anchor_loss:.4f}")
            extra = (" | " + " ".join(extra)) if extra else ""
            print(f"steps={agent_steps:,} | SPS={sps:,.0f}{extra}")

            if tracker is not None and train_win:
                score_value = _safe_float(ep_score)
                length_value = _safe_float(ep_len)
                score_label = "nan" if score_value is None else f"{score_value:.2f}"
                win_rate_label = f"{train_win_rate * 100:.1f}%"
                first_captured = tracker.log_train_win(
                    logs,
                    agent_steps=agent_steps,
                    epoch=trainer.epoch,
                )
                if first_captured:
                    first_train_win_path = os.path.join(tracker.run_dir, "first_train_win.pt")
                    _torch_save_atomic(policy.state_dict(), first_train_win_path)
                    tracker.log_train_win_checkpoint(
                        first_train_win_path,
                        epoch=trainer.epoch,
                        agent_steps=agent_steps,
                        kind="first",
                        score=score_value,
                        episode_length=length_value,
                        win_rate=train_win_rate,
                    )
                    print(
                        f"train_win_checkpoint: kind=first score={score_label} win_rate={win_rate_label} "
                        f"path={first_train_win_path}"
                    )

                latest_train_win_path = os.path.join(tracker.run_dir, "latest_train_win.pt")
                _torch_save_atomic(policy.state_dict(), latest_train_win_path)
                tracker.log_train_win_checkpoint(
                    latest_train_win_path,
                    epoch=trainer.epoch,
                    agent_steps=agent_steps,
                    kind="latest",
                    score=score_value,
                    episode_length=length_value,
                    win_rate=train_win_rate,
                )
                print(
                    f"train_win_checkpoint: kind=latest score={score_label} win_rate={win_rate_label} "
                    f"path={latest_train_win_path}"
                )

            if args.eval_every_steps > 0 and agent_steps - last_eval_at >= args.eval_every_steps:
                last_eval_at = agent_steps

                stats = evaluate_policy(
                    policy=policy,
                    device=args.device,
                    board_size=args.board_size,
                    episodes=args.eval_episodes,
                    seed=args.seed + 10_000,
                    deterministic=args.eval_deterministic,
                    gamma=float(args.gamma),
                    alpha=float(args.alpha),
                    flood_fill_obs=args.flood_fill,
                    body_age_obs=args.body_age_obs,
                    obs_history=args.obs_history,
                    action_history_obs=args.action_history_obs,
                    cycle_target_obs=args.aux_cycle_target,
                    tail_target_obs=args.aux_tail_target,
                    safe_action_target_obs=args.aux_safe_action_target,
                    safe_action_soft_target_obs=args.aux_safe_action_soft_target,
                    body_age_target_obs=args.aux_body_age_target,
                    cycle_target_min_fill=args.aux_cycle_target_min_fill,
                    safe_action_target_min_fill=args.aux_safe_action_target_min_fill,
                    safe_action_soft_target_min_fill=args.aux_safe_action_soft_target_min_fill,
                    body_age_target_min_fill=args.aux_body_age_target_min_fill,
                    safe_action_fill_weight=args.aux_safe_action_fill_weight,
                    safe_action_soft_temperature=args.aux_safe_action_soft_temperature,
                    safe_action_bonus=args.safe_action_bonus,
                    safe_action_bonus_min_fill=args.safe_action_bonus_min_fill,
                    safe_action_bonus_fill_weight=args.safe_action_bonus_fill_weight,
                    head_centered=args.head_centered,
                )
                mean_score = stats["mean_score"]
                win_rate = stats["win_rate"]
                perfect_score = stats["perfect_score"]
                print(
                    f"eval: mean_score={mean_score:.2f}/{perfect_score} median={stats['median_score']:.0f} "
                    f"win_rate={win_rate*100:.1f}% lt20={stats['phase_lt20_rate']*100:.1f}% "
                    f"95+={stats['phase_gte95_rate']*100:.1f}% ({stats['episodes']} eps)"
                )
                if tracker is not None:
                    improved = tracker.log_eval(
                        stats,
                        agent_steps=agent_steps,
                        epoch=trainer.epoch,
                        deterministic=args.eval_deterministic,
                    )
                    if improved:
                        best_eval_path = os.path.join(tracker.run_dir, "best_eval.pt")
                        _torch_save_atomic(policy.state_dict(), best_eval_path)
                        best_eval_resume_path = os.path.join(
                            tracker.run_dir, "best_eval_resume.pt"
                        )
                        _torch_save_atomic(
                            _build_resume_state(trainer, policy),
                            best_eval_resume_path,
                        )
                        tracker.log_best_eval_checkpoint(
                            best_eval_path,
                            epoch=trainer.epoch,
                            agent_steps=agent_steps,
                        )
                        print(
                            f"best_eval_checkpoint: mean_score={mean_score:.2f}/{perfect_score} "
                            f"win_rate={win_rate*100:.1f}% path={best_eval_path}"
                        )

                if args.perfect_patience > 0 and win_rate >= 1.0:
                    perfect_streak += 1
                    if perfect_streak >= args.perfect_patience:
                        print(
                            f"early_stop: perfect win_rate for {perfect_streak}/{args.perfect_patience} evals"
                        )
                        break
                else:
                    perfect_streak = 0

        if tracker is not None:
            checkpoint_interval = int(args.checkpoint_interval)
            if checkpoint_interval > 0 and (
                trainer.epoch % checkpoint_interval == 0 or trainer.epoch >= trainer.total_epochs
            ):
                agent_steps = _get_agent_steps(last_logs, trainer)
                checkpoint_path = os.path.join(
                    tracker.run_dir, f"model_{config['env']}_{trainer.epoch:06d}.pt"
                )
                torch.save(policy.state_dict(), checkpoint_path)
                tracker.log_checkpoint(
                    checkpoint_path,
                    epoch=trainer.epoch,
                    agent_steps=agent_steps,
                )

    final_checkpoint = trainer.close()
    elapsed = time.time() - start_time
    if tracker is not None:
        tracker.finalize(
            status="completed",
            final_checkpoint=final_checkpoint,
            elapsed_seconds=elapsed,
        )
    if last_logs is not None and elapsed > 0:
        agent_steps = _safe_int(last_logs.get("agent_steps", 0), 0)
        print(f"avg_SPS={agent_steps/elapsed:,.0f} (steps={agent_steps:,}, seconds={elapsed:.1f})")


if __name__ == "__main__":
    main()
