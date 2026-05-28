"""Online BC for PPO-compatible LSTM Snake policies.

The teacher is used only during training. Saved checkpoints are plain
SnakeRecurrentPPOPolicy state_dicts and run greedily from the base observation
stream at inference.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from distill.expert import expert_action, find_aligned_cycle
from eval_metrics import summarize_phase_metrics
from eval import SnakeRecurrentPPOPolicy
from snake_env import SnakeEnv


def _parse_seeds(value: str | None) -> list[int] | None:
    if value is None:
        return None
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("seed list cannot be empty")
    return seeds


def _save_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _save_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _make_policy(
    *,
    board_size: int,
    network_scale: int,
    hidden_size: int,
    embed_size: int | None,
    device: str,
    seed: int,
) -> SnakeRecurrentPPOPolicy:
    del seed
    policy = SnakeRecurrentPPOPolicy(
        board_size=board_size,
        scale=network_scale,
        n_channels=5,
        hidden_size=hidden_size,
        embed_size=embed_size,
    )
    for module in policy.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (torch.nn.LSTM, torch.nn.LSTMCell)):
            for name, param in module.named_parameters():
                if "weight" in name and param.ndim >= 2:
                    torch.nn.init.orthogonal_(param, gain=1.0)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)
    return policy.to(device)


def _load_state(policy: torch.nn.Module, path: Path) -> None:
    state = torch.load(path, map_location="cpu")
    incompatible = policy.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )


def _selector(stats: dict[str, Any]) -> tuple[float, float, float]:
    mean_win_steps = stats["mean_win_steps"]
    step_term = -float(mean_win_steps) if mean_win_steps is not None else 0.0
    return (float(stats["win_rate"]), float(stats["mean_score"]), step_term)


def _detach_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach() for key, value in state.items()}


def _train_chunk(
    *,
    policy: SnakeRecurrentPPOPolicy,
    anchor_policy: SnakeRecurrentPPOPolicy | None,
    optimizer: torch.optim.Optimizer,
    observations: list[np.ndarray],
    actions: list[int],
    state: dict[str, torch.Tensor],
    anchor_state: dict[str, torch.Tensor] | None,
    device: str,
    grad_clip: float,
    kl_anchor_coef: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, dict[str, float]]:
    obs_t = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=device).unsqueeze(0)
    action_t = torch.as_tensor(actions, dtype=torch.long, device=device)

    logits, _ = policy(obs_t, state)
    ce_loss = F.cross_entropy(logits, action_t)
    loss = ce_loss
    kl_loss = None

    next_anchor_state = anchor_state
    if anchor_policy is not None and anchor_state is not None and kl_anchor_coef > 0.0:
        with torch.no_grad():
            anchor_logits, _ = anchor_policy(obs_t, anchor_state)
            next_anchor_state = _detach_state(anchor_state)
        kl_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(anchor_logits, dim=-1),
            reduction="batchmean",
        )
        loss = loss + kl_anchor_coef * kl_loss

    pred = torch.argmax(logits, dim=-1)
    accuracy = (pred == action_t).float().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    optimizer.step()

    return _detach_state(state), next_anchor_state, {
        "loss": float(loss.item()),
        "ce_loss": float(ce_loss.item()),
        "kl_loss": None if kl_loss is None else float(kl_loss.item()),
        "accuracy": float(accuracy.item()),
    }


def train_episode(
    *,
    policy: SnakeRecurrentPPOPolicy,
    anchor_policy: SnakeRecurrentPPOPolicy | None,
    optimizer: torch.optim.Optimizer,
    board_size: int,
    seed: int,
    device: str,
    seq_len: int,
    max_steps: int,
    grad_clip: float,
    kl_anchor_coef: float,
) -> dict[str, Any]:
    env = SnakeEnv(n=board_size, gamma=0.99, alpha=0.2, seed=seed)
    obs, _ = env.reset(seed=seed)
    initial_direction = int(env.direction)
    cycle, head_idx = find_aligned_cycle(env)
    state = policy.initial_state(1, device)
    anchor_state = None if anchor_policy is None else anchor_policy.initial_state(1, device)

    obs_buf: list[np.ndarray] = []
    action_buf: list[int] = []
    losses: list[float] = []
    ce_losses: list[float] = []
    kl_losses: list[float] = []
    accuracies: list[float] = []
    chunks = 0
    info: dict[str, Any] = {}

    for step in range(max_steps):
        action, head_idx = expert_action(env, cycle, head_idx)
        obs_buf.append(obs.astype(np.float32, copy=True))
        action_buf.append(int(action))
        obs, _, terminated, truncated, info = env.step(action)

        if len(obs_buf) >= seq_len or terminated or truncated:
            state, anchor_state, metrics = _train_chunk(
                policy=policy,
                anchor_policy=anchor_policy,
                optimizer=optimizer,
                observations=obs_buf,
                actions=action_buf,
                state=state,
                anchor_state=anchor_state,
                device=device,
                grad_clip=grad_clip,
                kl_anchor_coef=kl_anchor_coef,
            )
            losses.append(metrics["loss"])
            ce_losses.append(metrics["ce_loss"])
            if metrics["kl_loss"] is not None:
                kl_losses.append(metrics["kl_loss"])
            accuracies.append(metrics["accuracy"])
            chunks += 1
            obs_buf = []
            action_buf = []

        if terminated or truncated:
            break

    score = int(info.get("score", env.score))
    return {
        "seed": seed,
        "initial_direction": initial_direction,
        "teacher_score": score,
        "teacher_reason": str(info.get("reason", "timeout")),
        "teacher_steps": int(info.get("steps", step + 1)),
        "chunks": chunks,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "mean_ce_loss": float(np.mean(ce_losses)) if ce_losses else 0.0,
        "mean_kl_loss": float(np.mean(kl_losses)) if kl_losses else None,
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
    }


@torch.no_grad()
def evaluate_policy(
    *,
    policy: SnakeRecurrentPPOPolicy,
    board_size: int,
    seeds: list[int],
    device: str,
    max_steps: int,
) -> dict[str, Any]:
    perfect_score = board_size * board_size - 3
    results: list[dict[str, Any]] = []

    policy.eval()
    for seed in seeds:
        env = SnakeEnv(n=board_size, gamma=0.99, alpha=0.2, seed=seed)
        obs, _ = env.reset(seed=seed)
        state = policy.initial_state(1, device)
        info: dict[str, Any] = {}
        for step in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = policy.forward_eval(obs_t, state)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        score = int(info.get("score", env.score))
        result = {
            "seed": seed,
            "score": score,
            "length": int(info.get("length", score + 3)),
            "reason": str(info.get("reason", "timeout")),
            "steps": int(info.get("steps", step + 1)),
            "win": score >= perfect_score,
        }
        results.append(result)
    policy.train()

    scores = [result["score"] for result in results]
    lengths = [result["length"] for result in results]
    reasons = [result["reason"] for result in results]
    wins = [result for result in results if result["win"]]
    win_steps = [result["steps"] for result in wins]
    summary = {
        "episodes": len(results),
        "seeds": seeds,
        "wins": len(wins),
        "win_rate": float(len(wins) / max(1, len(results))),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "median_score": float(np.median(scores)) if scores else 0.0,
        "min_score": int(min(scores)) if scores else 0,
        "max_score": int(max(scores)) if scores else 0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "failures": [result for result in results if not result["win"]],
        "mean_win_steps": float(np.mean(win_steps)) if win_steps else None,
        "median_win_steps": float(np.median(win_steps)) if win_steps else None,
        "p95_win_steps": float(np.percentile(win_steps, 95)) if win_steps else None,
        "steps_per_food": float(np.mean(win_steps) / perfect_score) if win_steps else None,
    }
    summary.update(
        summarize_phase_metrics(
            scores=scores,
            terminal_lengths=lengths,
            reasons=reasons,
            perfect_score=perfect_score,
            episodes=len(results),
        )
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Online BC for recurrent PPO LSTM Snake policies")
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--network-scale", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--embed-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--train-seeds", type=_parse_seeds, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--kl-anchor-coef", type=float, default=0.0)
    parser.add_argument("--train-policy-head-only", action="store_true")
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--eval-every-episodes", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=20001)
    parser.add_argument("--eval-seeds", type=_parse_seeds, default=None)
    args = parser.parse_args()

    if args.train_episodes < 1:
        raise SystemExit("--train-episodes must be >= 1")
    if args.seq_len < 1:
        raise SystemExit("--seq-len must be >= 1")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    if args.eval_every_episodes < 1:
        raise SystemExit("--eval-every-episodes must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy = _make_policy(
        board_size=args.board_size,
        network_scale=args.network_scale,
        hidden_size=args.hidden_size,
        embed_size=args.embed_size or None,
        device=args.device,
        seed=args.seed,
    )
    if args.resume is not None:
        _load_state(policy, args.resume)

    anchor_policy = None
    if args.kl_anchor_coef > 0.0:
        if args.resume is None:
            raise SystemExit("--kl-anchor-coef requires --resume")
        anchor_policy = _make_policy(
            board_size=args.board_size,
            network_scale=args.network_scale,
            hidden_size=args.hidden_size,
            embed_size=args.embed_size or None,
            device=args.device,
            seed=args.seed,
        )
        _load_state(anchor_policy, args.resume)
        anchor_policy.eval()
        for param in anchor_policy.parameters():
            param.requires_grad = False

    if args.train_policy_head_only:
        for name, param in policy.named_parameters():
            param.requires_grad = name.startswith("policy_head")
    optimizer = torch.optim.Adam([param for param in policy.parameters() if param.requires_grad], lr=args.lr)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    _save_json_atomic({"args": vars(args)}, args.save_path.parent / "run.json")
    started = time.time()
    best_key: tuple[float, float, float] | None = None
    events: list[dict[str, Any]] = []
    train_events: list[dict[str, Any]] = []
    train_seeds = args.train_seeds
    eval_seeds = args.eval_seeds
    if eval_seeds is None:
        eval_seeds = list(range(args.eval_seed, args.eval_seed + args.eval_episodes))

    policy.train()
    for episode in range(1, args.train_episodes + 1):
        if train_seeds is None:
            train_seed = args.seed + episode - 1
        else:
            train_seed = train_seeds[(episode - 1) % len(train_seeds)]
        train_event = train_episode(
            policy=policy,
            anchor_policy=anchor_policy,
            optimizer=optimizer,
            board_size=args.board_size,
            seed=train_seed,
            device=args.device,
            seq_len=args.seq_len,
            max_steps=args.max_steps,
            grad_clip=args.grad_clip,
            kl_anchor_coef=max(0.0, args.kl_anchor_coef),
        )
        train_event["episode"] = episode
        train_event["elapsed_sec"] = round(time.time() - started, 1)
        train_events.append(train_event)
        print({"train": train_event}, flush=True)

        if episode % args.eval_every_episodes == 0 or episode == args.train_episodes:
            stats = evaluate_policy(
                policy=policy,
                board_size=args.board_size,
                seeds=eval_seeds,
                device=args.device,
                max_steps=args.max_steps,
            )
            key = _selector(stats)
            if best_key is None or key > best_key:
                best_key = key
                best_path = args.save_path.with_name(f"{args.save_path.stem}.best_eval.pt")
                _save_atomic(policy.state_dict(), best_path)
                print({"best_checkpoint": str(best_path), "selector": best_key}, flush=True)
            event = {"episode": episode, "eval": stats, "elapsed_sec": round(time.time() - started, 1)}
            events.append(event)
            print({"eval": event}, flush=True)
            if stats["win_rate"] >= 1.0:
                break

    _save_atomic(policy.state_dict(), args.save_path)
    summary = {
        "save_path": str(args.save_path),
        "best_eval_path": str(args.save_path.with_name(f"{args.save_path.stem}.best_eval.pt")),
        "elapsed_sec": time.time() - started,
        "train_events": train_events,
        "eval_events": events,
        "args": vars(args),
    }
    _save_json_atomic(summary, args.save_path.with_name(f"{args.save_path.stem}.summary.json"))
    print({"saved": str(args.save_path), "summary": str(args.save_path.with_name(f"{args.save_path.stem}.summary.json"))}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
