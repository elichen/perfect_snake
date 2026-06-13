"""Build a web deployment of a trained 5-channel egocentric Snake policy.

Outputs:
  1. weights.bin   -- binary weights in the format agent-egocentric.js expects.
  2. verify_episode.json -- a full deterministic episode (board states + chosen
     actions + logits) recorded from the SAME policy, used to verify the JS port
     matches PyTorch action-for-action.

The model uses encoder_channels = 5 (head, body, food, length, walls). flood_fill is
an auxiliary TRAINING target only (forward_eval slices observations[:, :5]); the network
never sees it at inference, so the web obs builder needs no flood-fill.
"""
import argparse
import json
import struct

import numpy as np
import torch

from snake_env import SnakeEnv
from train import SnakePolicy

WEIGHT_ORDER = [
    "features.1.weight", "features.1.bias",
    "features.2.weight", "features.2.bias",
    "features.4.weight", "features.4.bias",
    "features.5.weight", "features.5.bias",
    "features.7.weight", "features.7.bias",
    "features.8.weight", "features.8.bias",
    "features.10.weight", "features.10.bias",
    "policy_head.0.weight", "policy_head.0.bias",
    "policy_head.2.weight", "policy_head.2.bias",
]


def load_policy(ckpt_path, board_size, scale, device="cpu"):
    env = SnakeEnv(n=board_size, flood_fill_obs=True, head_centered=True)
    policy = SnakePolicy(env, scale=scale, aux_flood_fill=True)
    sd = torch.load(ckpt_path, map_location=device)
    # Load only encoder + policy head (the inference path). Aux decoders/value head in
    # the model stay at init — they are never exported or used. strict=False alone
    # raises on aux-head shape mismatches, so filter to the keys we need.
    needed = {k: v for k, v in sd.items()
              if k.startswith("features.") or k.startswith("policy_head.")}
    missing, _ = policy.load_state_dict(needed, strict=False)
    crit = [k for k in missing if k.startswith("features.") or k.startswith("policy_head.")]
    assert not crit, f"critical params missing from checkpoint: {crit}"
    for k in WEIGHT_ORDER:
        assert k in needed, f"checkpoint missing required export key: {k}"
    policy.eval()
    return env, policy


def export_weights(policy, out_path, board_size, scale):
    sd = policy.state_dict()
    obs_n = 2 * (board_size - 1) + 1
    meta = {
        "board_size": board_size,
        "network_scale": scale,
        "n_channels": 5,
        "n_actions": 3,
        "obs_size": obs_n,
        "head_centered": True,
    }
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    buf = bytearray()
    buf += struct.pack("<I", len(meta_bytes))
    buf += meta_bytes
    buf += struct.pack("<H", len(WEIGHT_ORDER))
    for name in WEIGHT_ORDER:
        arr = sd[name].detach().cpu().numpy().astype(np.float32)
        name_bytes = name.encode("utf-8")
        buf += struct.pack("<H", len(name_bytes))
        buf += name_bytes
        buf += struct.pack("<B", arr.ndim)
        for dim in arr.shape:
            buf += struct.pack("<I", int(dim))
        buf += np.ascontiguousarray(arr).tobytes()  # C-order, little-endian float32
    with open(out_path, "wb") as f:
        f.write(buf)
    print(f"wrote {out_path} ({len(buf) / 1e6:.1f} MB), {len(WEIGHT_ORDER)} tensors")


def record_episode(env, policy, seed, want="win", max_seed_scan=400):
    """Play deterministic episodes from fresh seeds until one matches `want`
    (win/loss/any); record its full trajectory."""
    board_area = env.n * env.n
    perfect = board_area - 3
    s = seed
    for _ in range(max_seed_scan):
        obs, _ = env.reset(seed=s)
        frames = []
        done = False
        score = 0
        while not done:
            # Record pre-move state (env coords)
            snake = [[int(r), int(c)] for (r, c) in env.snake]
            food = [int(env.food_pos[0]), int(env.food_pos[1])]
            direction = int(env.direction)
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = policy.forward_eval(obs_t)
            logits = logits.squeeze(0).tolist()
            action = int(np.argmax(logits))
            frames.append({
                "snake": snake, "food": food, "dir": direction,
                "action": action, "logits": [round(x, 5) for x in logits],
            })
            obs, _, terminated, truncated, info = env.step(action)
            score = int(info.get("score", score))
            done = terminated or truncated
            reason = info.get("reason")
        is_win = score >= perfect
        if want == "any" or (want == "win" and is_win) or (want == "loss" and not is_win):
            return {
                "board_size": env.n, "seed": s, "won": is_win, "score": score,
                "steps": len(frames), "reason": reason, "frames": frames,
            }, s
        s += 1
    return None, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--board-size", type=int, default=20)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--weights-out", default="web_build/weights.bin")
    ap.add_argument("--episode-out", default="web_build/verify_episode.json")
    ap.add_argument("--record-seed", type=int, default=500001)
    ap.add_argument("--want", default="win", choices=["win", "loss", "any"])
    args = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(args.weights_out), exist_ok=True)

    env, policy = load_policy(args.checkpoint, args.board_size, args.scale)
    export_weights(policy, args.weights_out, args.board_size, args.scale)

    ep, found = record_episode(env, policy, args.record_seed, want=args.want)
    if ep is None:
        raise SystemExit(f"no '{args.want}' episode found scanning from {args.record_seed}")
    with open(args.episode_out, "w") as f:
        json.dump(ep, f)
    print(f"recorded {args.want} episode seed={found}: won={ep['won']} "
          f"score={ep['score']}/{env.n*env.n-3} steps={ep['steps']} -> {args.episode_out}")


if __name__ == "__main__":
    main()
