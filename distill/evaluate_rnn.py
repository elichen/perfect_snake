"""Standalone evaluator for recurrent distillation checkpoints."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from eval_metrics import summarize_phase_metrics
from snake_env import SnakeEnv

from .conditioning import augment_observation, conditioning_channels, find_cycle_condition
from .rnn_model import SnakeRNNPolicy, load_rnn_policy_state


@torch.no_grad()
def evaluate_policy(
    policy: SnakeRNNPolicy,
    *,
    board_size: int,
    episodes: int,
    seed: int,
    seeds: list[int] | None = None,
    deterministic: bool,
    device: str,
    flood_fill: bool,
    head_centered: bool,
    cycle_conditioning: bool = False,
    use_prev_action_input: bool = False,
    use_fill_input: bool = False,
) -> dict:
    perfect_score = board_size * board_size - 3
    scores: list[int] = []
    lengths: list[int] = []
    reasons: list[str] = []
    steps: list[int] = []

    eval_seeds = seeds if seeds is not None else [seed + ep for ep in range(episodes)]
    for ep_seed in eval_seeds:
        env = SnakeEnv(
            n=board_size,
            gamma=0.999,
            alpha=0.2,
            seed=ep_seed,
            flood_fill_obs=flood_fill,
            head_centered=head_centered,
        )
        obs, _ = env.reset(seed=ep_seed)
        hidden = policy.initial_state(1, device)
        prev_action = 3
        cycle_idx = None
        if cycle_conditioning:
            cycle_idx, _ = find_cycle_condition(env)
        done = False
        info = {}
        while not done:
            obs_eval = obs
            if cycle_conditioning:
                obs_eval = augment_observation(obs, env, cycle_idx)
            obs_t = torch.as_tensor(obs_eval, dtype=torch.float32, device=device).unsqueeze(0)
            prev_t = None
            if use_prev_action_input:
                prev_t = torch.as_tensor([prev_action], dtype=torch.long, device=device)
            fill_t = None
            if use_fill_input or getattr(policy, "early_head_max_fill", None) is not None:
                fill_ratio = env.snake_length / float(board_size * board_size)
                fill_t = torch.as_tensor([fill_ratio], dtype=torch.float32, device=device)
            logits, hidden = policy.forward_step(
                obs_t,
                hidden,
                prev_actions=prev_t,
                fill_values=fill_t,
            )
            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                action = int(torch.multinomial(probs[0], num_samples=1).item())
            prev_action = action
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        score = int(info.get("score", 0))
        length = int(info.get("length", score + 3))
        reason = str(info.get("reason", "unknown"))
        scores.append(score)
        lengths.append(length)
        reasons.append(reason)
        steps.append(int(info.get("steps", 0)))

    wins = sum(int(score >= perfect_score) for score in scores)
    win_steps = [step for score, step in zip(scores, steps) if score >= perfect_score]
    stats = {
        "episodes": len(eval_seeds),
        "seeds": eval_seeds,
        "deterministic": deterministic,
        "wins": wins,
        "win_rate": float(wins / max(1, len(eval_seeds))),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "min_score": int(min(scores)) if scores else 0,
        "max_score": int(max(scores)) if scores else 0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "mean_win_steps": float(np.mean(win_steps)) if win_steps else None,
        "median_win_steps": float(np.median(win_steps)) if win_steps else None,
        "p95_win_steps": float(np.percentile(win_steps, 95)) if win_steps else None,
        "steps_per_food": float(np.mean(win_steps) / perfect_score) if win_steps else None,
    }
    stats.update(
        summarize_phase_metrics(
            scores=scores,
            terminal_lengths=lengths,
            reasons=reasons,
            perfect_score=perfect_score,
            episodes=len(eval_seeds),
        )
    )
    return stats


@torch.no_grad()
def evaluate_checkpoint(
    *,
    checkpoint_path: str,
    board_size: int,
    episodes: int,
    seed: int,
    seeds: list[int] | None = None,
    deterministic: bool,
    device: str,
    flood_fill: bool,
    head_centered: bool,
    hidden_size: int = 256,
    cycle_conditioning: bool = False,
    use_prev_action_input: bool = False,
    use_fill_input: bool = False,
    future_action_horizon: int = 0,
    early_head_max_fill: float | None = None,
    residual_policy_head: bool = False,
) -> dict:
    base_channels = 5 + int(flood_fill)
    n_channels = base_channels
    if cycle_conditioning:
        probe_env = SnakeEnv(
            n=board_size,
            gamma=0.999,
            alpha=0.2,
            seed=seed,
            flood_fill_obs=flood_fill,
            head_centered=head_centered,
        )
        n_channels += conditioning_channels(probe_env)
    policy = SnakeRNNPolicy(
        board_size=board_size,
        n_channels=n_channels,
        flood_fill=flood_fill,
        head_centered=head_centered,
        hidden_size=hidden_size,
        prev_action_input=use_prev_action_input,
        fill_input=use_fill_input,
        future_action_horizon=future_action_horizon,
        early_head_max_fill=early_head_max_fill,
        residual_policy_head=residual_policy_head,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    load_rnn_policy_state(policy, state_dict)
    policy.eval()

    stats = evaluate_policy(
        policy,
        board_size=board_size,
        episodes=episodes,
        seed=seed,
        seeds=seeds,
        deterministic=deterministic,
        device=device,
        flood_fill=flood_fill,
        head_centered=head_centered,
        cycle_conditioning=cycle_conditioning,
        use_prev_action_input=use_prev_action_input,
        use_fill_input=use_fill_input,
    )
    stats["checkpoint"] = checkpoint_path
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RNN distillation checkpoint")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated exact episode seeds. Overrides --episodes/--seed range.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--flood-fill", action="store_true")
    parser.add_argument("--head-centered", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--cycle-conditioning", action="store_true")
    parser.add_argument("--prev-action-input", action="store_true")
    parser.add_argument("--fill-input", action="store_true")
    parser.add_argument("--future-action-horizon", type=int, default=0)
    parser.add_argument("--early-head-max-fill", type=float, default=None)
    parser.add_argument("--residual-policy-head", action="store_true")
    args = parser.parse_args()
    seed_list = None
    if args.seeds:
        seed_list = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        if not seed_list:
            raise SystemExit("--seeds produced an empty seed list")

    stats = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        board_size=args.board_size,
        episodes=args.episodes,
        seed=args.seed,
        seeds=seed_list,
        deterministic=args.deterministic,
        device=args.device,
        flood_fill=args.flood_fill,
        head_centered=args.head_centered,
        hidden_size=args.hidden_size,
        cycle_conditioning=args.cycle_conditioning,
        use_prev_action_input=args.prev_action_input,
        use_fill_input=args.fill_input,
        future_action_horizon=args.future_action_horizon,
        early_head_max_fill=args.early_head_max_fill,
        residual_policy_head=args.residual_policy_head,
    )
    print(stats)


if __name__ == "__main__":
    main()
