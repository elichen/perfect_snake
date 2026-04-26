"""Recurrent policy for standalone expert distillation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SnakeRNNPolicy(nn.Module):
    """GRU policy for Snake distillation."""

    def __init__(
        self,
        *,
        board_size: int,
        n_channels: int = 5,
        flood_fill: bool = False,
        head_centered: bool = False,
        hidden_size: int = 256,
        embed_size: int = 256,
        prev_action_input: bool = False,
        fill_input: bool = False,
        future_action_horizon: int = 0,
        early_head_max_fill: float | None = None,
    ):
        super().__init__()
        total_channels = n_channels
        self.encoder_channels = total_channels
        self.hidden_size = hidden_size
        self.prev_action_input = prev_action_input
        self.fill_input = fill_input
        self.prev_action_vocab = 4
        self.future_action_horizon = future_action_horizon
        self.early_head_max_fill = early_head_max_fill

        if head_centered:
            obs_n = 2 * (board_size - 1) + 1
        else:
            obs_n = board_size + 2
        n_input = int(np.prod((self.encoder_channels, obs_n, obs_n)))

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input, 1024 if hidden_size >= 256 else 512),
            nn.LayerNorm(1024 if hidden_size >= 256 else 512),
            nn.ReLU(),
            nn.Linear(1024 if hidden_size >= 256 else 512, embed_size),
            nn.LayerNorm(embed_size),
            nn.ReLU(),
        )
        extra_dim = 0
        if prev_action_input:
            self.prev_action_embed = nn.Embedding(self.prev_action_vocab, 16)
            extra_dim += 16
        if fill_input:
            extra_dim += 1
        self.gru_cell = nn.GRUCell(embed_size + extra_dim, hidden_size)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3),
        )
        if early_head_max_fill is not None:
            self.early_policy_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 3),
            )
        if future_action_horizon > 0:
            self.future_action_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, future_action_horizon * 3),
            )
        self._sync_early_head_from_base()

    def initial_state(self, batch_size: int, device: str | torch.device):
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def _sync_early_head_from_base(self) -> None:
        if self.early_head_max_fill is None:
            return
        self.early_policy_head.load_state_dict(self.policy_head.state_dict())

    def _augment_encoded(self, encoded, prev_actions=None, fill_values=None):
        pieces = [encoded]
        if self.prev_action_input:
            if prev_actions is None:
                raise ValueError("prev_actions is required when prev_action_input=True")
            pieces.append(self.prev_action_embed(prev_actions))
        if self.fill_input:
            if fill_values is None:
                raise ValueError("fill_values is required when fill_input=True")
            pieces.append(fill_values.unsqueeze(-1))
        if len(pieces) == 1:
            return encoded
        return torch.cat(pieces, dim=-1)

    def forward_sequence(
        self,
        observations,
        hidden=None,
        reset_mask=None,
        prev_actions=None,
        fill_values=None,
        return_features: bool = False,
    ):
        # observations: [T, B, C, H, W]
        t, b = observations.shape[:2]
        if hidden is None:
            hidden = self.initial_state(b, observations.device)
        flat = observations.reshape(t * b, *observations.shape[2:])
        encoded = self.encoder(flat).reshape(t, b, -1)
        encoded = self._augment_encoded(encoded, prev_actions=prev_actions, fill_values=fill_values)
        outputs = []
        for step in range(t):
            if reset_mask is not None:
                hidden = hidden * reset_mask[step].unsqueeze(-1)
            hidden = self.gru_cell(encoded[step], hidden)
            outputs.append(hidden)
        outputs_t = torch.stack(outputs, dim=0)
        logits = self.policy_head(outputs_t.reshape(t * b, -1)).reshape(t, b, 3)
        if self.early_head_max_fill is not None:
            if fill_values is None:
                raise ValueError("fill_values is required when early_head_max_fill is set")
            early_mask = fill_values < self.early_head_max_fill
            if torch.any(early_mask):
                early_logits = self.early_policy_head(outputs_t[early_mask])
                logits = logits.clone()
                logits[early_mask] = early_logits
        if return_features:
            return logits, hidden, outputs_t
        return logits, hidden

    def forward_step(self, observations, hidden=None, prev_actions=None, fill_values=None):
        # observations: [B, C, H, W]
        batch_size = observations.shape[0]
        if hidden is None:
            hidden = self.initial_state(batch_size, observations.device)
        encoded = self.encoder(observations)
        encoded = self._augment_encoded(encoded, prev_actions=prev_actions, fill_values=fill_values).unsqueeze(0)
        hidden = self.gru_cell(encoded.squeeze(0), hidden)
        logits = self.policy_head(hidden)
        if self.early_head_max_fill is not None:
            if fill_values is None:
                raise ValueError("fill_values is required when early_head_max_fill is set")
            early_mask = fill_values < self.early_head_max_fill
            if torch.any(early_mask):
                early_logits = self.early_policy_head(hidden[early_mask])
                logits = logits.clone()
                logits[early_mask] = early_logits
        return logits, hidden

    def forward_future_logits(self, hidden_states):
        if self.future_action_horizon < 1:
            raise ValueError("future_action_horizon must be > 0 to request future logits")
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = self.future_action_head(flat)
        return logits.reshape(*hidden_states.shape[:2], self.future_action_horizon, 3)


def load_rnn_policy_state(policy: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    incompatible = policy.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed_missing = set()
    if getattr(policy, "future_action_horizon", 0) > 0:
        allowed_missing.update({
            "future_action_head.0.weight",
            "future_action_head.0.bias",
            "future_action_head.2.weight",
            "future_action_head.2.bias",
        })
    if getattr(policy, "early_head_max_fill", None) is not None:
        allowed_missing.update({
            "early_policy_head.0.weight",
            "early_policy_head.0.bias",
            "early_policy_head.2.weight",
            "early_policy_head.2.bias",
        })
    if unexpected or any(key not in allowed_missing for key in missing):
        raise RuntimeError(
            f"Error(s) in loading state_dict: missing={missing} unexpected={unexpected}"
        )
    if getattr(policy, "early_head_max_fill", None) is not None and any(key.startswith("early_policy_head") for key in missing):
        sync_early = getattr(policy, "_sync_early_head_from_base", None)
        if callable(sync_early):
            sync_early()


def freeze_except_early_head(policy: nn.Module) -> None:
    if getattr(policy, "early_head_max_fill", None) is None:
        raise ValueError("freeze_except_early_head requires early_head_max_fill to be set")
    for name, param in policy.named_parameters():
        param.requires_grad = name.startswith("early_policy_head")
