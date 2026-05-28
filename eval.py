"""Evaluate a trained Snake policy checkpoint."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_metrics import PHASE_BUCKETS, summarize_phase_metrics
from snake_env import SnakeEnv


class SnakePolicy(nn.Module):
    """FC policy for Snake (must match train.py architecture)."""

    def __init__(self, board_size: int, scale: int = 1, n_channels: int = 5,
                 aux_flood_fill: bool = False, aux_cycle_target: bool = False,
                 aux_tail_target: bool = False, aux_safe_action_target: bool = False,
                 aux_safe_action_soft_target: bool = False,
                 aux_body_age_target: bool = False,
                 head_centered: bool = False,
                 late_head_min_fill: float | None = None):
        super().__init__()

        total_channels = n_channels
        n_actions = 3

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

        if head_centered:
            obs_n = 2 * (board_size - 1) + 1  # 39 for board_size=20
        else:
            obs_n = board_size + 2
        obs_shape = (self.encoder_channels, obs_n, obs_n)
        n_input = int(np.prod(obs_shape))
        self.board_size = board_size

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

        if aux_flood_fill:
            if head_centered:
                flood_target_n = obs_n
            else:
                flood_target_n = board_size
            self.flood_decoder = nn.Sequential(
                nn.Linear(w[3], w[2]),
                nn.ReLU(),
                nn.Linear(w[2], flood_target_n * flood_target_n),
            )

        if aux_cycle_target:
            if head_centered:
                cycle_target_n = obs_n
            else:
                cycle_target_n = board_size
            self.cycle_target_decoder = nn.Sequential(
                nn.Linear(w[3], w[2]),
                nn.ReLU(),
                nn.Linear(w[2], cycle_target_n * cycle_target_n),
            )

        if aux_tail_target:
            if head_centered:
                tail_target_n = obs_n
            else:
                tail_target_n = board_size
            self.tail_target_decoder = nn.Sequential(
                nn.Linear(w[3], w[2]),
                nn.ReLU(),
                nn.Linear(w[2], tail_target_n * tail_target_n),
            )

        if aux_body_age_target:
            if head_centered:
                body_age_target_n = obs_n
            else:
                body_age_target_n = board_size
            self.body_age_target_decoder = nn.Sequential(
                nn.Linear(w[3], w[2]),
                nn.ReLU(),
                nn.Linear(w[2], body_age_target_n * body_age_target_n),
            )

        self._sync_late_heads_from_base()

    def _sync_late_heads_from_base(self):
        if self.late_head_min_fill is None:
            return
        self.late_policy_head.load_state_dict(self.policy_head.state_dict())
        self.late_value_head.load_state_dict(self.value_head.state_dict())

    def forward(self, observations, state=None):
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


class SnakeRecurrentPPOPolicy(nn.Module):
    """LSTM policy for recurrent PPO checkpoints."""

    def __init__(
        self,
        board_size: int,
        scale: int = 1,
        n_channels: int = 5,
        hidden_size: int = 256,
        embed_size: int | None = None,
        head_centered: bool = False,
        full_ff_encoder: bool = False,
        residual_recurrent: bool = False,
    ):
        super().__init__()

        self.encoder_channels = n_channels
        self.hidden_size = int(hidden_size)
        self.board_size = board_size
        self.head_centered = head_centered
        self.full_ff_encoder = bool(full_ff_encoder)
        self.residual_recurrent = bool(residual_recurrent)

        if head_centered:
            obs_n = 2 * (board_size - 1) + 1
        else:
            obs_n = board_size + 2
        self.obs_shape = (n_channels, obs_n, obs_n)
        n_input = int(np.prod(self.obs_shape))

        if embed_size is None:
            embed_size = 256 if scale <= 1 else 512
        trunk = 1024 if scale <= 1 else 2048
        if scale >= 4:
            trunk = 4096

        if self.full_ff_encoder:
            w = [1024, 512, 256, 128]
            if scale == 2:
                w = [2048, 1024, 512, 256]
            elif scale == 4:
                w = [4096, 2048, 1024, 512]
            recurrent_input_size = w[-1]
            self.encoder = nn.Sequential(
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
        else:
            recurrent_input_size = int(embed_size)
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_input, trunk),
                nn.LayerNorm(trunk),
                nn.ReLU(),
                nn.Linear(trunk, int(embed_size)),
                nn.LayerNorm(int(embed_size)),
                nn.ReLU(),
            )
        if self.residual_recurrent and recurrent_input_size != self.hidden_size:
            raise ValueError("--recurrent-residual requires recurrent input size to match hidden size")

        self.lstm = nn.LSTM(recurrent_input_size, self.hidden_size)
        self.cell = nn.LSTMCell(recurrent_input_size, self.hidden_size)
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0

        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 3),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
        )

    def initial_state(self, batch_size: int, device: str | torch.device) -> dict[str, torch.Tensor]:
        return {
            "lstm_h": torch.zeros(batch_size, self.hidden_size, device=device),
            "lstm_c": torch.zeros(batch_size, self.hidden_size, device=device),
        }

    def _state_tensors(
        self,
        state: dict | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None or state.get("lstm_h") is None or state.get("lstm_c") is None:
            zeros = self.initial_state(batch_size, device)
            return zeros["lstm_h"], zeros["lstm_c"]
        h = state["lstm_h"]
        c = state["lstm_c"]
        if h.ndim == 3:
            h = h.squeeze(0)
        if c.ndim == 3:
            c = c.squeeze(0)
        return h, c

    def _encode_flat(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations[:, :self.encoder_channels])

    def _decode(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy_head(hidden), self.value_head(hidden)

    def forward_eval(self, observations, state=None):
        batch_size = observations.shape[0]
        encoded = self._encode_flat(observations)
        h, c = self._state_tensors(state, batch_size, observations.device)
        h, c = self.cell(encoded, (h, c))
        if state is not None:
            state["lstm_h"] = h.detach()
            state["lstm_c"] = c.detach()
        features = encoded + h if self.residual_recurrent else h
        return self._decode(features)

    def forward(self, observations, state=None):
        obs_dims = len(self.obs_shape)
        if observations.ndim == obs_dims + 1:
            return self.forward_eval(observations, state)
        if observations.ndim != obs_dims + 2:
            raise ValueError(f"invalid observation shape: {tuple(observations.shape)}")

        batch_size, timesteps = observations.shape[:2]
        flat_obs = observations.reshape(batch_size * timesteps, *self.obs_shape)
        encoded = self._encode_flat(flat_obs).reshape(batch_size, timesteps, -1)
        encoded = encoded.transpose(0, 1)
        h, c = self._state_tensors(state, batch_size, observations.device)
        outputs, (h_n, c_n) = self.lstm(encoded, (h.unsqueeze(0), c.unsqueeze(0)))
        outputs = outputs.transpose(0, 1)
        if self.residual_recurrent:
            outputs = outputs + encoded.transpose(0, 1)
        outputs = outputs.reshape(batch_size * timesteps, self.hidden_size)
        logits, values = self._decode(outputs)
        if state is not None:
            state["lstm_h"] = h_n.detach()
            state["lstm_c"] = c_n.detach()
        return logits, values.reshape(batch_size, timesteps)


class IterativeBlock(nn.Module):
    """Weight-tied convolutional block (must match train.py)."""

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
    """Weight-tied iterative CNN (must match train.py)."""

    def __init__(self, board_size: int, scale: int = 1, n_channels: int = 5,
                 n_iterations: int = 12, aux_flood_fill: bool = False,
                 aux_tail_target: bool = False, aux_safe_action_target: bool = False):
        super().__init__()

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
        self.aux_tail_target = aux_tail_target
        self.aux_safe_action_target = aux_safe_action_target
        self.channels = channels
        self.encoder_channels = (
            total_channels
            - int(aux_flood_fill)
            - int(aux_tail_target)
            - int(aux_safe_action_target)
        )

        n_groups = min(8, channels)

        self.input_conv = nn.Sequential(
            nn.Conv2d(self.encoder_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(n_groups, channels),
            nn.ReLU(),
        )
        self.iter_block = IterativeBlock(channels)
        self.post_norm = nn.GroupNorm(n_groups, channels)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.policy_head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        if aux_flood_fill:
            self.flood_decoder = nn.Sequential(
                nn.Conv2d(channels, channels // 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels // 2, 1, 1),
            )

        if aux_tail_target:
            self.tail_target_decoder = nn.Sequential(
                nn.Conv2d(channels, channels // 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels // 2, 1, 1),
            )

    def forward_spatial(self, observations):
        x = self.input_conv(observations)
        for _ in range(self.n_iterations):
            x = self.iter_block(x)
        x = torch.relu(self.post_norm(x))
        return x

    def forward(self, observations, state=None):
        obs_input = observations[:, :self.encoder_channels]
        spatial = self.forward_spatial(obs_input)
        features = self.gap(spatial).flatten(1)
        logits = self.policy_head(features)
        values = self.value_head(features)
        return logits, values


def _load_policy_state(
    policy: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    aux_flood_fill: bool,
    aux_cycle_target: bool,
    aux_tail_target: bool,
    aux_body_age_target: bool,
    late_head_min_fill: float | None,
) -> None:
    first_layer_key = "features.1.weight"
    target_state = policy.state_dict()
    if first_layer_key in state_dict and first_layer_key in target_state:
        source_weight = state_dict[first_layer_key]
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
            state_dict = dict(state_dict)
            state_dict[first_layer_key] = adapted

    filtered = dict(state_dict)
    if not aux_flood_fill:
        filtered = {k: v for k, v in filtered.items() if not k.startswith("flood_decoder")}
    if not aux_cycle_target:
        filtered = {k: v for k, v in filtered.items() if not k.startswith("cycle_target_decoder")}
    if not aux_tail_target:
        filtered = {k: v for k, v in filtered.items() if not k.startswith("tail_target_decoder")}
    if not aux_body_age_target:
        filtered = {k: v for k, v in filtered.items() if not k.startswith("body_age_target_decoder")}
    if late_head_min_fill is None:
        filtered = {
            k: v for k, v in filtered.items()
            if not k.startswith("late_policy_head") and not k.startswith("late_value_head")
        }

    incompatible = policy.load_state_dict(filtered, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    allowed_missing = set()
    if aux_cycle_target:
        allowed_missing.update({
            "cycle_target_decoder.0.weight",
            "cycle_target_decoder.0.bias",
            "cycle_target_decoder.2.weight",
            "cycle_target_decoder.2.bias",
        })
    if aux_tail_target:
        allowed_missing.update({
            "tail_target_decoder.0.weight",
            "tail_target_decoder.0.bias",
            "tail_target_decoder.2.weight",
            "tail_target_decoder.2.bias",
        })
    if aux_body_age_target:
        allowed_missing.update({
            "body_age_target_decoder.0.weight",
            "body_age_target_decoder.0.bias",
            "body_age_target_decoder.2.weight",
            "body_age_target_decoder.2.bias",
        })
    if late_head_min_fill is not None:
        allowed_missing.update({
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
        })

    if unexpected or any(key not in allowed_missing for key in missing):
        raise RuntimeError(
            "Error(s) in loading state_dict for "
            f"{policy.__class__.__name__}: missing={missing} unexpected={unexpected}"
        )
    if late_head_min_fill is not None and any(key.startswith("late_") for key in missing):
        sync_late = getattr(policy, "_sync_late_heads_from_base", None)
        if callable(sync_late):
            sync_late()


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    board_size: int,
    episodes: int,
    seed: int,
    deterministic: bool,
    device: str,
    network_scale: int = 1,
    verbose: bool = False,
    flood_fill: bool = False,
    body_age_obs: bool = False,
    obs_history: int = 1,
    action_history_obs: int = 0,
    iterative_cnn: bool = False,
    n_iterations: int = 12,
    recurrent: bool = False,
    recurrent_hidden_size: int = 256,
    recurrent_embed_size: int = 0,
    recurrent_full_ff_encoder: bool = False,
    recurrent_residual: bool = False,
    aux_flood_fill: bool = False,
    aux_cycle_target: bool = False,
    aux_tail_target: bool = False,
    aux_safe_action_target: bool = False,
    aux_safe_action_soft_target: bool = False,
    aux_body_age_target: bool = False,
    late_head_min_fill: float | None = None,
    aux_cycle_target_min_fill: float | None = None,
    aux_safe_action_target_min_fill: float = 0.90,
    aux_safe_action_soft_target_min_fill: float = 0.90,
    body_age_obs_min_fill: float = 0.90,
    aux_body_age_target_min_fill: float = 0.80,
    aux_safe_action_fill_weight: float = 500.0,
    aux_safe_action_soft_temperature: float = 1.0,
    safe_action_bonus: float = 0.0,
    safe_action_bonus_min_fill: float = 0.95,
    safe_action_bonus_fill_weight: float = 500.0,
    head_centered: bool = False,
    num_envs: int = 1,
    return_episode_data: bool = False,
) -> dict:
    """Load checkpoint and evaluate."""

    # Load policy
    base_obs_channels = 5 + int(flood_fill) + int(body_age_obs) + 3 * int(action_history_obs)
    n_channels = (
        base_obs_channels * int(obs_history)
        + int(aux_cycle_target)
        + int(aux_tail_target)
        + int(aux_safe_action_target)
        + 3 * int(aux_safe_action_soft_target)
        + int(aux_body_age_target)
    )
    state_dict = torch.load(checkpoint_path, map_location=device)

    if recurrent:
        if iterative_cnn:
            raise ValueError("recurrent evaluation cannot be combined with iterative_cnn")
        policy = SnakeRecurrentPPOPolicy(
            board_size=board_size,
            scale=network_scale,
            n_channels=n_channels,
            hidden_size=recurrent_hidden_size,
            embed_size=(recurrent_embed_size or None),
            head_centered=head_centered,
            full_ff_encoder=recurrent_full_ff_encoder,
            residual_recurrent=recurrent_residual,
        ).to(device)
        _load_policy_state(
            policy,
            state_dict,
            aux_flood_fill=False,
            aux_cycle_target=False,
            aux_tail_target=False,
            aux_body_age_target=False,
            late_head_min_fill=None,
        )
    elif iterative_cnn:
        policy = SnakeIterativeCNNPolicy(
            board_size=board_size, scale=network_scale, n_channels=n_channels,
            n_iterations=n_iterations, aux_flood_fill=aux_flood_fill,
            aux_tail_target=aux_tail_target,
            aux_safe_action_target=aux_safe_action_target,
        ).to(device)
        _load_policy_state(
            policy,
            state_dict,
            aux_flood_fill=aux_flood_fill,
            aux_cycle_target=False,
            aux_tail_target=aux_tail_target,
            aux_body_age_target=False,
            late_head_min_fill=None,
        )
    else:
        policy = SnakePolicy(board_size, scale=network_scale, n_channels=n_channels,
                             aux_flood_fill=aux_flood_fill,
                             aux_cycle_target=aux_cycle_target,
                             aux_tail_target=aux_tail_target,
                             aux_safe_action_target=aux_safe_action_target,
                             aux_safe_action_soft_target=aux_safe_action_soft_target,
                             aux_body_age_target=aux_body_age_target,
                             head_centered=head_centered,
                             late_head_min_fill=late_head_min_fill).to(device)
        _load_policy_state(
            policy,
            state_dict,
            aux_flood_fill=aux_flood_fill,
            aux_cycle_target=aux_cycle_target,
            aux_tail_target=aux_tail_target,
            aux_body_age_target=aux_body_age_target,
            late_head_min_fill=late_head_min_fill,
        )
    policy.eval()

    perfect_score = board_size * board_size - 3

    scores = []
    wins = 0
    lengths = []
    death_lengths = []
    reasons = []
    death_reasons = {}

    if verbose or num_envs <= 1:
        env = SnakeEnv(
            n=board_size,
            gamma=0.99,
            alpha=0.2,
            seed=seed,
            flood_fill_obs=flood_fill,
            body_age_obs=body_age_obs,
            obs_history=obs_history,
            action_history_obs=action_history_obs,
            cycle_target_obs=aux_cycle_target,
            tail_target_obs=aux_tail_target,
            safe_action_target_obs=aux_safe_action_target,
            safe_action_soft_target_obs=aux_safe_action_soft_target,
            body_age_target_obs=aux_body_age_target,
            cycle_target_min_fill=aux_cycle_target_min_fill,
            safe_action_target_min_fill=aux_safe_action_target_min_fill,
            safe_action_soft_target_min_fill=aux_safe_action_soft_target_min_fill,
            body_age_obs_min_fill=body_age_obs_min_fill,
            body_age_target_min_fill=aux_body_age_target_min_fill,
            safe_action_fill_weight=aux_safe_action_fill_weight,
            safe_action_soft_temperature=aux_safe_action_soft_temperature,
            safe_action_bonus=safe_action_bonus,
            safe_action_bonus_min_fill=safe_action_bonus_min_fill,
            safe_action_bonus_fill_weight=safe_action_bonus_fill_weight,
            head_centered=head_centered,
        )

        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            state = policy.initial_state(1, device) if recurrent else None
            done = False
            last_info = info
            steps = 0

            while not done:
                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                if recurrent:
                    logits, _ = policy.forward_eval(obs_t, state)
                else:
                    logits, _ = policy(obs_t)
                if deterministic:
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    action = int(torch.distributions.Categorical(logits=logits).sample().item())
                obs, _, terminated, truncated, last_info = env.step(action)
                done = terminated or truncated
                steps += 1

            score = int(last_info.get("score", 0))
            snake_len = int(last_info.get("length", score + 3))
            reason = last_info.get("reason", "unknown")
            scores.append(score)
            lengths.append(steps)
            death_lengths.append(snake_len)
            reasons.append(str(reason))
            death_reasons[reason] = death_reasons.get(reason, 0) + 1

            if score >= perfect_score:
                wins += 1

            if verbose:
                fill_pct = snake_len / (board_size * board_size) * 100
                win_str = "WIN" if score >= perfect_score else ""
                print(f"  Ep {ep+1:3d}: score={score:3d}/{perfect_score}  len={snake_len:3d}  fill={fill_pct:4.1f}%  steps={steps:5d}  {reason:6s} {win_str}")
    else:
        active_slots = min(num_envs, episodes)
        envs = []
        obs_batch = []
        episode_steps = []
        next_seed = seed
        recurrent_state = policy.initial_state(active_slots, device) if recurrent else None

        for _ in range(active_slots):
            env = SnakeEnv(
                n=board_size,
                gamma=0.99,
                alpha=0.2,
                seed=next_seed,
                flood_fill_obs=flood_fill,
                body_age_obs=body_age_obs,
                obs_history=obs_history,
                action_history_obs=action_history_obs,
                cycle_target_obs=aux_cycle_target,
                tail_target_obs=aux_tail_target,
                safe_action_target_obs=aux_safe_action_target,
                safe_action_soft_target_obs=aux_safe_action_soft_target,
                body_age_target_obs=aux_body_age_target,
                cycle_target_min_fill=aux_cycle_target_min_fill,
                safe_action_target_min_fill=aux_safe_action_target_min_fill,
                safe_action_soft_target_min_fill=aux_safe_action_soft_target_min_fill,
                body_age_obs_min_fill=body_age_obs_min_fill,
                body_age_target_min_fill=aux_body_age_target_min_fill,
                safe_action_fill_weight=aux_safe_action_fill_weight,
                safe_action_soft_temperature=aux_safe_action_soft_temperature,
                safe_action_bonus=safe_action_bonus,
                safe_action_bonus_min_fill=safe_action_bonus_min_fill,
                safe_action_bonus_fill_weight=safe_action_bonus_fill_weight,
                head_centered=head_centered,
            )
            obs, _ = env.reset(seed=next_seed)
            envs.append(env)
            obs_batch.append(obs)
            episode_steps.append(0)
            next_seed += 1

        while len(scores) < episodes:
            active_indices = [i for i, obs in enumerate(obs_batch) if obs is not None]
            batch_obs = np.stack([obs_batch[i] for i in active_indices]).astype(np.float32)
            obs_t = torch.as_tensor(batch_obs, device=device, dtype=torch.float32)
            if recurrent:
                state = {
                    "lstm_h": recurrent_state["lstm_h"][active_indices],
                    "lstm_c": recurrent_state["lstm_c"][active_indices],
                }
                logits, _ = policy.forward_eval(obs_t, state)
                recurrent_state["lstm_h"][active_indices] = state["lstm_h"]
                recurrent_state["lstm_c"][active_indices] = state["lstm_c"]
            else:
                logits, _ = policy(obs_t)
            if deterministic:
                batch_actions = torch.argmax(logits, dim=-1).cpu().numpy()
            else:
                batch_actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()

            for slot, action in zip(active_indices, batch_actions):
                obs, _, terminated, truncated, info = envs[slot].step(int(action))
                episode_steps[slot] += 1

                if terminated or truncated:
                    score = int(info.get("score", 0))
                    snake_len = int(info.get("length", score + 3))
                    reason = info.get("reason", "unknown")
                    scores.append(score)
                    lengths.append(episode_steps[slot])
                    death_lengths.append(snake_len)
                    reasons.append(str(reason))
                    death_reasons[reason] = death_reasons.get(reason, 0) + 1

                    if score >= perfect_score:
                        wins += 1

                    if len(scores) >= episodes:
                        obs_batch[slot] = None
                        continue

                    obs, _ = envs[slot].reset(seed=next_seed)
                    next_seed += 1
                    episode_steps[slot] = 0
                    if recurrent:
                        recurrent_state["lstm_h"][slot].zero_()
                        recurrent_state["lstm_c"][slot].zero_()

                obs_batch[slot] = obs

    board_area = board_size * board_size
    bucket_size = board_area // 10
    bucket_counts = [0] * 10
    for dl in death_lengths:
        bucket = min((dl - 1) // bucket_size, 9)
        bucket_counts[bucket] += 1

    stats = {
        "checkpoint": checkpoint_path,
        "board_size": board_size,
        "perfect_score": perfect_score,
        "episodes": episodes,
        "deterministic": deterministic,
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "std_score": float(np.std(scores)),
        "min_score": int(np.min(scores)),
        "max_score": int(np.max(scores)),
        "win_rate": float(wins / episodes),
        "wins": wins,
        "mean_length": float(np.mean(lengths)),
        "death_lengths": death_lengths,
        "mean_death_length": float(np.mean(death_lengths)),
        "median_death_length": float(np.median(death_lengths)),
        "death_reasons": death_reasons,
        "death_fill_buckets": bucket_counts,
        "board_area": board_area,
    }
    stats.update(
        summarize_phase_metrics(
            scores=scores,
            terminal_lengths=death_lengths,
            reasons=reasons,
            perfect_score=perfect_score,
            episodes=episodes,
        )
    )
    if return_episode_data:
        stats["scores"] = [int(score) for score in scores]
        stats["lengths"] = [int(length) for length in lengths]
        stats["death_lengths_raw"] = [int(length) for length in death_lengths]
        stats["reasons"] = [str(reason) for reason in reasons]
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Snake checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint file")
    parser.add_argument("--board-size", type=int, default=10, help="Board size (default: 10)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes (default: 100)")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for evaluation")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--network-scale", type=int, default=1, choices=[1, 2, 4], help="Network width multiplier (must match training)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-episode results")
    parser.add_argument("--flood-fill", action="store_true", help="Use flood-fill observation channel")
    parser.add_argument("--body-age-obs", action="store_true", help="Use the late-gated body-age observation channel")
    parser.add_argument("--obs-history", type=int, default=1, help="Number of current+previous encoder-visible observations to stack (current frame first)")
    parser.add_argument("--action-history-obs", type=int, default=0, help="Number of previous relative actions to encode as one-hot broadcast planes")
    parser.add_argument("--iterative-cnn", action="store_true", help="Use iterative CNN policy")
    parser.add_argument("--n-iterations", type=int, default=12, help="Iterations for iterative CNN")
    parser.add_argument("--recurrent", action="store_true", help="Use recurrent PPO LSTM policy")
    parser.add_argument("--recurrent-hidden-size", type=int, default=256, help="Hidden size for --recurrent policy")
    parser.add_argument("--recurrent-embed-size", type=int, default=0, help="Encoder size for --recurrent policy (0 = scale default)")
    parser.add_argument("--recurrent-full-ff-encoder", action="store_true", help="Use the feed-forward PPO trunk before the recurrent adapter")
    parser.add_argument("--recurrent-residual", action="store_true", help="Add LSTM output as a residual correction to encoder features")
    parser.add_argument("--aux-flood-fill", action="store_true", help="Model was trained with aux flood-fill decoder")
    parser.add_argument("--aux-cycle-target", action="store_true", help="Model was trained with auxiliary curriculum cycle target decoder")
    parser.add_argument("--aux-tail-target", action="store_true", help="Model was trained with auxiliary tail target decoder")
    parser.add_argument("--aux-body-age-target", action="store_true", help="Model was trained with auxiliary body-age target decoder")
    parser.add_argument("--aux-cycle-target-min-fill", type=float, default=None, help="Minimum fill level before the auxiliary curriculum cycle target channel activates")
    parser.add_argument("--late-head-min-fill", type=float, default=None, help="Route late-game states to a dedicated policy/value head starting at this fill fraction")
    parser.add_argument("--aux-safe-action-target", action="store_true", help="Model was trained with auxiliary safe action target channel")
    parser.add_argument("--aux-safe-action-soft-target", action="store_true", help="Model was trained with auxiliary soft safe action target channels")
    parser.add_argument("--aux-safe-action-target-min-fill", type=float, default=0.90, help="Minimum fill level before the auxiliary safe action target channel activates")
    parser.add_argument("--aux-safe-action-soft-target-min-fill", type=float, default=0.90, help="Minimum fill level before the auxiliary soft safe action target activates")
    parser.add_argument("--body-age-obs-min-fill", type=float, default=0.90, help="Minimum fill level before the body-age observation channel activates")
    parser.add_argument("--aux-body-age-target-min-fill", type=float, default=0.80, help="Minimum fill level before the auxiliary body-age target channel activates")
    parser.add_argument("--aux-safe-action-fill-weight", type=float, default=500.0, help="Fill weight for the auxiliary safe action heuristic")
    parser.add_argument("--aux-safe-action-soft-temperature", type=float, default=1.0, help="Temperature used to soften safe action scores into a target distribution")
    parser.add_argument("--safe-action-bonus", type=float, default=0.0, help="Late-game safe action bonus used during training/eval env setup")
    parser.add_argument("--safe-action-bonus-min-fill", type=float, default=0.95, help="Minimum fill level before the safe action bonus activates")
    parser.add_argument("--safe-action-bonus-fill-weight", type=float, default=500.0, help="Fill weight for the safe action bonus heuristic")
    parser.add_argument("--head-centered", action="store_true", help="Head-centered observation (39x39 for 20x20 board)")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel envs for evaluation (ignored when --verbose)")
    args = parser.parse_args()

    print(f"Evaluating: {args.checkpoint}")
    print(f"  board_size={args.board_size}, episodes={args.episodes}, deterministic={args.deterministic}")
    print()

    try:
        stats = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            board_size=args.board_size,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=args.deterministic,
            device=args.device,
            network_scale=args.network_scale,
            verbose=args.verbose,
            flood_fill=args.flood_fill,
            body_age_obs=args.body_age_obs,
            obs_history=args.obs_history,
            action_history_obs=args.action_history_obs,
            iterative_cnn=args.iterative_cnn,
            n_iterations=args.n_iterations,
            recurrent=args.recurrent,
            recurrent_hidden_size=args.recurrent_hidden_size,
            recurrent_embed_size=args.recurrent_embed_size,
            recurrent_full_ff_encoder=args.recurrent_full_ff_encoder,
            recurrent_residual=args.recurrent_residual,
            aux_flood_fill=args.aux_flood_fill,
            aux_cycle_target=args.aux_cycle_target,
            aux_tail_target=args.aux_tail_target,
            aux_body_age_target=args.aux_body_age_target,
            late_head_min_fill=args.late_head_min_fill,
            aux_cycle_target_min_fill=args.aux_cycle_target_min_fill,
            aux_safe_action_target=args.aux_safe_action_target,
            aux_safe_action_soft_target=args.aux_safe_action_soft_target,
            aux_safe_action_target_min_fill=args.aux_safe_action_target_min_fill,
            aux_safe_action_soft_target_min_fill=args.aux_safe_action_soft_target_min_fill,
            body_age_obs_min_fill=args.body_age_obs_min_fill,
            aux_body_age_target_min_fill=args.aux_body_age_target_min_fill,
            aux_safe_action_fill_weight=args.aux_safe_action_fill_weight,
            aux_safe_action_soft_temperature=args.aux_safe_action_soft_temperature,
            safe_action_bonus=args.safe_action_bonus,
            safe_action_bonus_min_fill=args.safe_action_bonus_min_fill,
            safe_action_bonus_fill_weight=args.safe_action_bonus_fill_weight,
            head_centered=args.head_centered,
            num_envs=args.num_envs,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 60)
    print(f"Results for {args.checkpoint}")
    print("=" * 60)
    print(f"  Board size:    {stats['board_size']}x{stats['board_size']}")
    print(f"  Perfect score: {stats['perfect_score']}")
    print(f"  Episodes:      {stats['episodes']}")
    print(f"  Deterministic: {stats['deterministic']}")
    print()
    print(f"  Win rate:      {stats['win_rate']*100:.1f}% ({stats['wins']}/{stats['episodes']})")
    print(f"  Mean score:    {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
    print(f"  Median score:  {stats['median_score']:.0f}")
    print(f"  Score range:   [{stats['min_score']}, {stats['max_score']}]")
    print(f"  Mean length:   {stats['mean_length']:.1f} steps")
    print()

    print(f"  --- Phase Buckets ---")
    phase_labels = {
        "phase_lt20": "<20%",
        "phase_20_80": "20-80%",
        "phase_80_95": "80-95%",
        "phase_gte95": "95-100%",
    }
    for bucket_name, _, _ in PHASE_BUCKETS:
        count_key = f"{bucket_name}_count"
        rate_key = f"{bucket_name}_rate"
        label = phase_labels[bucket_name]
        print(f"  {label:12s}: {stats[count_key]:3d} ({stats[rate_key]*100:.1f}%)")
    print(f"  win         : {stats['win_count']:3d} ({stats['win_rate']*100:.1f}%)")
    print()

    # Death analysis
    board_area = stats["board_area"]
    print(f"  --- Death Analysis ---")
    print(f"  Mean death length:   {stats['mean_death_length']:.1f}/{board_area} ({stats['mean_death_length']/board_area*100:.1f}% fill)")
    print(f"  Median death length: {stats['median_death_length']:.0f}/{board_area} ({stats['median_death_length']/board_area*100:.1f}% fill)")
    print()

    # Death reasons
    print(f"  Death reasons:")
    for reason, count in sorted(stats["death_reasons"].items(), key=lambda x: -x[1]):
        print(f"    {reason:8s}: {count:3d} ({count/stats['episodes']*100:.1f}%)")
    print()

    # Histogram
    bucket_size = board_area // 10
    buckets = stats["death_fill_buckets"]
    max_count = max(buckets) if max(buckets) > 0 else 1
    print(f"  Death length distribution:")
    for i, count in enumerate(buckets):
        lo = i * bucket_size + 1
        hi = (i + 1) * bucket_size
        if i == 9:
            hi = board_area
        bar = "#" * int(count / max_count * 30)
        pct = count / stats["episodes"] * 100
        print(f"    {lo:3d}-{hi:3d} ({lo/board_area*100:4.0f}-{hi/board_area*100:3.0f}%): {bar:30s} {count:3d} ({pct:4.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    main()
