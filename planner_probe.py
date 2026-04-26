"""Probe a shallow late-game planner on top of a trained policy."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from eval import SnakePolicy
from eval_metrics import summarize_phase_metrics
from snake_env import SnakeEnv


def load_policy(checkpoint: str, *, board_size: int, device: str, network_scale: int,
                flood_fill: bool, aux_flood_fill: bool, head_centered: bool) -> SnakePolicy:
    n_channels = 5 + int(flood_fill)
    policy = SnakePolicy(
        board_size=board_size,
        scale=network_scale,
        n_channels=n_channels,
        aux_flood_fill=aux_flood_fill,
        aux_cycle_target=False,
        aux_tail_target=False,
        head_centered=head_centered,
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    state = {k: v for k, v in state.items()
             if not k.startswith("cycle_target_decoder")
             and not k.startswith("tail_target_decoder")}
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy


@torch.no_grad()
def run_probe(*, checkpoint: str, board_size: int, episodes: int, seed: int,
              threshold: float, fill_weight: float, logit_weight: float, depth: int,
              device: str, network_scale: int, flood_fill: bool,
              aux_flood_fill: bool, head_centered: bool) -> dict:
    policy = load_policy(
        checkpoint,
        board_size=board_size,
        device=device,
        network_scale=network_scale,
        flood_fill=flood_fill,
        aux_flood_fill=aux_flood_fill,
        head_centered=head_centered,
    )
    perfect_score = board_size * board_size - 3
    scores = []
    reasons = []
    lengths = []
    planner_steps = 0

    def score_actions(env: SnakeEnv, search_depth: int) -> list[float]:
        if search_depth <= 1:
            return env.score_relative_actions(fill_weight=fill_weight)

        scores = []
        snapshot = env._snapshot_state()
        for action in range(3):
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info.get("reason") == "win":
                    score = float("inf")
                else:
                    score = float("-inf")
            else:
                score = max(score_actions(env, search_depth - 1))
            scores.append(score)
            env._restore_state(snapshot)
        return scores

    for ep in range(episodes):
        env = SnakeEnv(
            n=board_size,
            gamma=0.99,
            alpha=0.2,
            seed=seed + ep,
            flood_fill_obs=flood_fill,
            head_centered=head_centered,
        )
        obs, _ = env.reset(seed=seed + ep)
        done = False
        last_info = {}

        while not done:
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            logits, _ = policy(obs_t)
            logits_np = logits[0].cpu().numpy()
            action = int(np.argmax(logits_np))

            fill = env.snake_length / float(board_size * board_size)
            if fill >= threshold:
                planner_steps += 1
                planner_scores = score_actions(env, depth)
                combined = [
                    planner_scores[a] + logit_weight * float(logits_np[a])
                    for a in range(3)
                ]
                action = int(np.argmax(combined))

            obs, _, terminated, truncated, last_info = env.step(action)
            done = terminated or truncated

        score = int(last_info.get("score", 0))
        scores.append(score)
        reasons.append(str(last_info.get("reason", "unknown")))
        lengths.append(int(last_info.get("length", score + 3)))
        print({
            "episode": ep + 1,
            "score": score,
            "reason": last_info.get("reason"),
            "length": lengths[-1],
        })

    wins = sum(int(score >= perfect_score) for score in scores)
    stats = {
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "win_rate": float(wins / max(1, episodes)),
        "wins": wins,
        "episodes": episodes,
        "planner_steps": planner_steps,
    }
    stats.update(
        summarize_phase_metrics(
            scores=scores,
            terminal_lengths=lengths,
            reasons=reasons,
            perfect_score=perfect_score,
            episodes=episodes,
        )
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a shallow late-game planner")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--fill-weight", type=float, default=500.0)
    parser.add_argument("--logit-weight", type=float, default=0.01)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--network-scale", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--flood-fill", action="store_true")
    parser.add_argument("--aux-flood-fill", action="store_true")
    parser.add_argument("--head-centered", action="store_true")
    args = parser.parse_args()

    stats = run_probe(
        checkpoint=args.checkpoint,
        board_size=args.board_size,
        episodes=args.episodes,
        seed=args.seed,
        threshold=args.threshold,
        fill_weight=args.fill_weight,
        logit_weight=args.logit_weight,
        depth=args.depth,
        device=args.device,
        network_scale=args.network_scale,
        flood_fill=args.flood_fill,
        aux_flood_fill=args.aux_flood_fill,
        head_centered=args.head_centered,
    )
    print(stats)


if __name__ == "__main__":
    main()
