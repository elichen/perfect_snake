"""Standalone MLP policy definition for distillation experiments."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SnakePolicy(nn.Module):
    """FC policy for Snake distillation (checkpoint-compatible with MLP PPO line)."""

    def __init__(
        self,
        *,
        board_size: int,
        scale: int = 1,
        n_channels: int = 5,
        aux_flood_fill: bool = False,
        head_centered: bool = False,
        late_head_min_fill: float | None = None,
    ):
        super().__init__()

        total_channels = n_channels
        n_actions = 3

        self.aux_flood_fill = aux_flood_fill
        self.head_centered = head_centered
        self.late_head_min_fill = late_head_min_fill
        self.aux_target_channels = int(aux_flood_fill)
        self.encoder_channels = total_channels - self.aux_target_channels

        if head_centered:
            obs_n = 2 * (board_size - 1) + 1
        else:
            obs_n = board_size + 2
        obs_shape = (self.encoder_channels, obs_n, obs_n)
        n_input = int(np.prod(obs_shape))
        self.board_size = board_size

        widths = [1024, 512, 256, 128]
        if scale == 2:
            widths = [2048, 1024, 512, 256]
        elif scale == 4:
            widths = [4096, 2048, 1024, 512]

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input, widths[0]),
            nn.LayerNorm(widths[0]),
            nn.ReLU(),
            nn.Linear(widths[0], widths[1]),
            nn.LayerNorm(widths[1]),
            nn.ReLU(),
            nn.Linear(widths[1], widths[2]),
            nn.LayerNorm(widths[2]),
            nn.ReLU(),
            nn.Linear(widths[2], widths[3]),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(widths[3], widths[3] // 2),
            nn.ReLU(),
            nn.Linear(widths[3] // 2, n_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(widths[3], widths[3]),
            nn.ReLU(),
            nn.Linear(widths[3], widths[3] // 2),
            nn.ReLU(),
            nn.Linear(widths[3] // 2, 1),
        )

        if late_head_min_fill is not None:
            self.late_policy_head = nn.Sequential(
                nn.Linear(widths[3], widths[3] // 2),
                nn.ReLU(),
                nn.Linear(widths[3] // 2, n_actions),
            )
            self.late_value_head = nn.Sequential(
                nn.Linear(widths[3], widths[3]),
                nn.ReLU(),
                nn.Linear(widths[3], widths[3] // 2),
                nn.ReLU(),
                nn.Linear(widths[3] // 2, 1),
            )

        if aux_flood_fill:
            self.flood_target_n = obs_n if head_centered else board_size
            self.flood_decoder = nn.Sequential(
                nn.Linear(widths[3], widths[2]),
                nn.ReLU(),
                nn.Linear(widths[2], self.flood_target_n * self.flood_target_n),
            )

        self._sync_late_heads_from_base()

    def _sync_late_heads_from_base(self) -> None:
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
        features = self.features(enc_input)
        pred = self.flood_decoder(features)
        n = self.flood_target_n
        return pred.view(-1, 1, n, n)


def load_policy_state(
    policy: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    aux_flood_fill: bool,
    late_head_min_fill: float | None,
) -> None:
    filtered = {
        k: v for k, v in state_dict.items()
        if not k.startswith("cycle_target_decoder")
        and not k.startswith("tail_target_decoder")
        and not k.startswith("safe_action")
    }
    if not aux_flood_fill:
        filtered = {k: v for k, v in filtered.items() if not k.startswith("flood_decoder")}
    if late_head_min_fill is None:
        filtered = {
            k: v for k, v in filtered.items()
            if not k.startswith("late_policy_head") and not k.startswith("late_value_head")
        }

    incompatible = policy.load_state_dict(filtered, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    allowed_missing = set()
    if aux_flood_fill:
        allowed_missing.update({
            "flood_decoder.0.weight",
            "flood_decoder.0.bias",
            "flood_decoder.2.weight",
            "flood_decoder.2.bias",
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


def freeze_except_late_heads(policy: nn.Module) -> None:
    policy._late_head_grad_hooks = []
    for name, param in policy.named_parameters():
        if name.startswith("late_policy_head") or name.startswith("late_value_head"):
            continue
        policy._late_head_grad_hooks.append(
            param.register_hook(lambda grad: torch.zeros_like(grad))
        )

