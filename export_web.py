"""Export PyTorch Snake model to compact JSON for web deployment."""

import argparse
import json
import gzip
import os
import numpy as np
import torch
import torch.nn as nn


class SnakePolicy(nn.Module):
    def __init__(self, board_size: int, scale: int = 1):
        super().__init__()
        n_channels = 5
        obs_shape = (n_channels, board_size + 2, board_size + 2)
        n_input = int(np.prod(obs_shape))
        n_actions = 3

        w = [1024, 512, 256, 128]
        if scale == 2:
            w = [2048, 1024, 512, 256]
        elif scale == 4:
            w = [4096, 2048, 1024, 512]

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input, w[0]), nn.LayerNorm(w[0]), nn.ReLU(),
            nn.Linear(w[0], w[1]), nn.LayerNorm(w[1]), nn.ReLU(),
            nn.Linear(w[1], w[2]), nn.LayerNorm(w[2]), nn.ReLU(),
            nn.Linear(w[2], w[3]), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(w[3], w[3] // 2), nn.ReLU(),
            nn.Linear(w[3] // 2, n_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(w[3], w[3]), nn.ReLU(),
            nn.Linear(w[3], w[3] // 2), nn.ReLU(),
            nn.Linear(w[3] // 2, 1),
        )

    def forward(self, observations, state=None):
        features = self.features(observations)
        return self.policy_head(features), self.value_head(features)


def export_compact(checkpoint_path: str, output_path: str, board_size: int, network_scale: int):
    policy = SnakePolicy(board_size, scale=network_scale)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state_dict, strict=True)
    
    # Convert to float16 and flatten for compact storage
    weights = {}
    for name, param in policy.named_parameters():
        # Round to 4 decimal places and store as flat list
        arr = param.detach().numpy().astype(np.float32)
        weights[name] = {
            "shape": list(arr.shape),
            "data": [round(float(x), 4) for x in arr.flatten()]
        }
    
    output = {
        "metadata": {
            "board_size": board_size,
            "network_scale": network_scale,
            "n_channels": 5,
            "n_actions": 3,
            "obs_size": board_size + 2,
        },
        "weights": weights,
    }
    
    # Write compact JSON (no pretty printing)
    with open(output_path, "w") as f:
        json.dump(output, f, separators=(',', ':'))
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--network-scale", type=int, default=2)
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.checkpoint.replace(".pt", "_web.json")
    
    export_compact(args.checkpoint, args.output, args.board_size, args.network_scale)
