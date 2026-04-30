"""Online recurrent BC on full Hamiltonian expert trajectories.

This trainer is intentionally inference-pure: the policy sees only the normal
Snake observation stream. The Hamiltonian cycle is used only as a train-time
teacher for labels, and hidden state is carried forward from reset so the RNN
can learn phase rather than being asked to classify ambiguous random windows.
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

from distill.evaluate_rnn import evaluate_policy
from distill.expert import expert_action, find_aligned_cycle
from distill.rnn_model import SnakeRNNPolicy, load_rnn_policy_state
from rnn_cycle_shortcut_patch import _teacher_action
from snake_env import SnakeEnv


START_ACTION_TOKEN = 3


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


def _selector(stats: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(stats["win_rate"]),
        float(stats["mean_score"]),
        float(stats["phase_gte95_rate"]),
        -float(stats["phase_lt20_rate"]),
    )


def _make_env(*, board_size: int, flood_fill: bool, head_centered: bool, seed: int) -> SnakeEnv:
    return SnakeEnv(
        n=board_size,
        gamma=0.999,
        alpha=0.2,
        seed=seed,
        flood_fill_obs=flood_fill,
        head_centered=head_centered,
    )


def _train_chunk(
    *,
    policy: SnakeRNNPolicy,
    anchor_policy: SnakeRNNPolicy | None,
    optimizer: torch.optim.Optimizer,
    observations: list[np.ndarray],
    actions: list[int],
    prev_actions: list[int],
    fill_values: list[float],
    hidden: torch.Tensor,
    anchor_hidden: torch.Tensor | None,
    device: str,
    grad_clip: float,
    kl_anchor_coef: float,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
    obs_t = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=device).unsqueeze(1)
    act_t = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    prev_t = None
    if policy.prev_action_input:
        prev_t = torch.as_tensor(prev_actions, dtype=torch.long, device=device).unsqueeze(1)
    fill_t = None
    if policy.fill_input:
        fill_t = torch.as_tensor(fill_values, dtype=torch.float32, device=device).unsqueeze(1)

    logits, next_hidden = policy.forward_sequence(
        obs_t,
        hidden=hidden,
        prev_actions=prev_t,
        fill_values=fill_t,
    )
    loss = F.cross_entropy(logits.reshape(-1, 3), act_t.reshape(-1))
    kl_loss = None
    next_anchor_hidden = anchor_hidden
    if anchor_policy is not None and anchor_hidden is not None and kl_anchor_coef > 0.0:
        with torch.no_grad():
            anchor_logits, next_anchor_hidden = anchor_policy.forward_sequence(
                obs_t,
                hidden=anchor_hidden,
                prev_actions=prev_t,
                fill_values=fill_t,
            )
        kl_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(anchor_logits, dim=-1),
            reduction="none",
        ).sum(dim=-1).mean()
        loss = loss + kl_anchor_coef * kl_loss
    pred = torch.argmax(logits, dim=-1)
    accuracy = (pred == act_t).float().mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    optimizer.step()

    return next_hidden.detach(), None if next_anchor_hidden is None else next_anchor_hidden.detach(), {
        "loss": float(loss.item()),
        "kl_loss": None if kl_loss is None else float(kl_loss.item()),
        "accuracy": float(accuracy.item()),
    }


def _maybe_eval(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    eval_episodes: int,
    eval_seed: int,
    eval_seeds: list[int] | None,
    device: str,
    flood_fill: bool,
    head_centered: bool,
    use_prev_action_input: bool,
    use_fill_input: bool,
    save_path: Path,
    best: tuple[float, float, float, float] | None,
    events: list[dict[str, Any]],
    label: str,
) -> tuple[tuple[float, float, float, float] | None, dict[str, Any]]:
    policy.eval()
    stats = evaluate_policy(
        policy,
        board_size=board_size,
        episodes=eval_episodes,
        seed=eval_seed,
        seeds=eval_seeds,
        deterministic=True,
        device=device,
        flood_fill=flood_fill,
        head_centered=head_centered,
        cycle_conditioning=False,
        use_prev_action_input=use_prev_action_input,
        use_fill_input=use_fill_input,
    )
    policy.train()
    current = _selector(stats)
    if best is None or current > best:
        best = current
        best_path = save_path.with_name(f"{save_path.stem}.best_eval.pt")
        _save_atomic(policy.state_dict(), best_path)
        print(
            {
                "best_checkpoint": str(best_path),
                "label": label,
                "mean_score": round(float(stats["mean_score"]), 3),
                "win_rate": round(float(stats["win_rate"]), 4),
            },
            flush=True,
        )
    events.append({"label": label, "eval": stats})
    return best, stats


def train_episode(
    *,
    policy: SnakeRNNPolicy,
    anchor_policy: SnakeRNNPolicy | None,
    optimizer: torch.optim.Optimizer,
    board_size: int,
    flood_fill: bool,
    head_centered: bool,
    seed: int,
    device: str,
    seq_len: int,
    max_steps: int,
    grad_clip: float,
    teacher_mode: str,
    max_plan_nodes: int,
    max_plan_candidates: int,
    kl_anchor_coef: float,
) -> dict[str, Any]:
    env = _make_env(
        board_size=board_size,
        flood_fill=flood_fill,
        head_centered=head_centered,
        seed=seed,
    )
    obs, _ = env.reset(seed=seed)
    initial_direction = int(env.direction)
    cycle, head_idx = find_aligned_cycle(env)
    cycle_index = {pos: idx for idx, pos in enumerate(cycle)}
    hidden = policy.initial_state(1, device)
    anchor_hidden = None if anchor_policy is None else anchor_policy.initial_state(1, device)
    prev_action = START_ACTION_TOKEN

    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    prev_buf: list[int] = []
    fill_buf: list[float] = []
    chunk_losses: list[float] = []
    chunk_kls: list[float] = []
    chunk_accs: list[float] = []
    chunks = 0
    done = False
    info: dict[str, Any] = {}
    steps = 0

    while not done and steps < max_steps:
        if teacher_mode == "hamiltonian":
            action, next_head_idx = expert_action(env, cycle, head_idx)
        else:
            action = _teacher_action(
                env,
                cycle,
                cycle_index,
                teacher_mode,
                max_plan_nodes=max_plan_nodes,
                max_plan_candidates=max_plan_candidates,
            )
            next_head_idx = cycle_index.get(_target_head_after_action(env, action), head_idx)
        obs_buf.append(obs.astype(np.float32, copy=True))
        act_buf.append(int(action))
        prev_buf.append(int(prev_action))
        fill_buf.append(env.snake_length / float(board_size * board_size))

        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        head_idx = next_head_idx
        prev_action = int(action)
        steps += 1

        if len(obs_buf) >= seq_len or done:
            hidden, anchor_hidden, metrics = _train_chunk(
                policy=policy,
                anchor_policy=anchor_policy,
                optimizer=optimizer,
                observations=obs_buf,
                actions=act_buf,
                prev_actions=prev_buf,
                fill_values=fill_buf,
                hidden=hidden,
                anchor_hidden=anchor_hidden,
                device=device,
                grad_clip=grad_clip,
                kl_anchor_coef=kl_anchor_coef,
            )
            chunk_losses.append(metrics["loss"])
            if metrics["kl_loss"] is not None:
                chunk_kls.append(metrics["kl_loss"])
            chunk_accs.append(metrics["accuracy"])
            chunks += 1
            obs_buf = []
            act_buf = []
            prev_buf = []
            fill_buf = []

    reason = str(info.get("reason", "timeout" if steps >= max_steps else "unknown"))
    score = int(info.get("score", env.score))
    return {
        "seed": seed,
        "initial_direction": initial_direction,
        "teacher_score": score,
        "teacher_reason": reason,
        "teacher_steps": steps,
        "chunks": chunks,
        "mean_loss": float(np.mean(chunk_losses)) if chunk_losses else 0.0,
        "mean_kl_loss": float(np.mean(chunk_kls)) if chunk_kls else None,
        "mean_accuracy": float(np.mean(chunk_accs)) if chunk_accs else 0.0,
    }


def _target_head_after_action(env: SnakeEnv, action: int) -> tuple[int, int]:
    new_dir = (env.direction + {0: -1, 1: 0, 2: 1}[int(action)]) % 4
    dr, dc = env.DIRECTIONS[new_dir]
    hr, hc = env.snake_head
    return hr + dr, hc + dc


def main() -> int:
    parser = argparse.ArgumentParser(description="Online recurrent BC from full Hamiltonian expert episodes")
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=80_000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--flood-fill", action="store_true")
    parser.add_argument("--head-centered", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--prev-action-input", action="store_true")
    parser.add_argument("--fill-input", action="store_true")
    parser.add_argument("--teacher-mode", choices=["hamiltonian", "cycle", "grid_path"], default="hamiltonian")
    parser.add_argument("--max-plan-nodes", type=int, default=2000)
    parser.add_argument("--max-plan-candidates", type=int, default=64)
    parser.add_argument("--kl-anchor-coef", type=float, default=0.0)
    parser.add_argument(
        "--train-policy-head-only",
        action="store_true",
        help="Freeze encoder/recurrent dynamics and update only policy_head parameters.",
    )
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument(
        "--train-seeds",
        type=str,
        default=None,
        help="Optional comma-separated seed list. If provided, cycles through these seeds for training.",
    )
    parser.add_argument(
        "--initial-directions",
        type=str,
        default=None,
        help="Optional comma-separated reset directions to train on, e.g. '2,3'",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--eval-every-episodes", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=10001)
    parser.add_argument(
        "--eval-seeds",
        type=str,
        default=None,
        help="Optional comma-separated exact eval seeds. Overrides the contiguous --eval-seed/--eval-episodes gate.",
    )
    args = parser.parse_args()

    if args.train_episodes < 1:
        raise SystemExit("--train-episodes must be >= 1")
    if args.seq_len < 1:
        raise SystemExit("--seq-len must be >= 1")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    if args.eval_every_episodes < 1:
        raise SystemExit("--eval-every-episodes must be >= 1")
    eval_seed_list = None
    if args.eval_seeds:
        eval_seed_list = [
            int(part.strip())
            for part in args.eval_seeds.split(",")
            if part.strip()
        ]
        if not eval_seed_list:
            raise SystemExit("--eval-seeds produced an empty seed list")
    train_seed_list = None
    if args.train_seeds:
        train_seed_list = [
            int(part.strip())
            for part in args.train_seeds.split(",")
            if part.strip()
        ]
        if not train_seed_list:
            raise SystemExit("--train-seeds produced an empty seed list")
    initial_directions = None
    if args.initial_directions:
        initial_directions = {
            int(part.strip())
            for part in args.initial_directions.split(",")
            if part.strip()
        }
        if not initial_directions or any(direction not in (0, 1, 2, 3) for direction in initial_directions):
            raise SystemExit("--initial-directions must contain comma-separated values from {0,1,2,3}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    probe_env = _make_env(
        board_size=args.board_size,
        flood_fill=args.flood_fill,
        head_centered=args.head_centered,
        seed=args.seed,
    )
    policy = SnakeRNNPolicy(
        board_size=args.board_size,
        n_channels=probe_env.observation_space.shape[0],
        flood_fill=args.flood_fill,
        head_centered=args.head_centered,
        hidden_size=args.hidden_size,
        prev_action_input=args.prev_action_input,
        fill_input=args.fill_input,
    ).to(args.device)
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu")
        load_rnn_policy_state(policy, state)
    anchor_policy = None
    if args.kl_anchor_coef > 0.0:
        if args.resume is None:
            raise SystemExit("--kl-anchor-coef requires --resume")
        anchor_policy = SnakeRNNPolicy(
            board_size=args.board_size,
            n_channels=probe_env.observation_space.shape[0],
            flood_fill=args.flood_fill,
            head_centered=args.head_centered,
            hidden_size=args.hidden_size,
            prev_action_input=args.prev_action_input,
            fill_input=args.fill_input,
        ).to(args.device)
        load_rnn_policy_state(anchor_policy, torch.load(args.resume, map_location="cpu"))
        anchor_policy.eval()
        for param in anchor_policy.parameters():
            param.requires_grad = False
    if args.train_policy_head_only:
        for name, param in policy.named_parameters():
            param.requires_grad = name.startswith("policy_head")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    policy.train()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    _save_json_atomic({"args": vars(args)}, args.save_path.parent / "run.json")

    started = time.time()
    best: tuple[float, float, float, float] | None = None
    events: list[dict[str, Any]] = []
    train_events: list[dict[str, Any]] = []

    seed_cursor = args.seed
    for ep_idx in range(1, args.train_episodes + 1):
        if train_seed_list is not None:
            train_seed = train_seed_list[(ep_idx - 1) % len(train_seed_list)]
        else:
            train_seed = seed_cursor
        if initial_directions is not None and train_seed_list is None:
            while True:
                probe = _make_env(
                    board_size=args.board_size,
                    flood_fill=args.flood_fill,
                    head_centered=args.head_centered,
                    seed=train_seed,
                )
                probe.reset(seed=train_seed)
                if int(probe.direction) in initial_directions:
                    break
                train_seed += 1
        if train_seed_list is None:
            seed_cursor = train_seed + 1
        event = train_episode(
            policy=policy,
            anchor_policy=anchor_policy,
            optimizer=optimizer,
            board_size=args.board_size,
            flood_fill=args.flood_fill,
            head_centered=args.head_centered,
            seed=train_seed,
            device=args.device,
            seq_len=args.seq_len,
            max_steps=args.max_steps,
            grad_clip=args.grad_clip,
            teacher_mode=args.teacher_mode,
            max_plan_nodes=max(1, args.max_plan_nodes),
            max_plan_candidates=max(1, args.max_plan_candidates),
            kl_anchor_coef=max(0.0, args.kl_anchor_coef),
        )
        event["episode"] = ep_idx
        event["elapsed_sec"] = round(time.time() - started, 1)
        train_events.append(event)
        print(
            {
                "episode": ep_idx,
                "seed": event["seed"],
                "initial_direction": event["initial_direction"],
                "teacher_score": event["teacher_score"],
                "teacher_reason": event["teacher_reason"],
                "teacher_steps": event["teacher_steps"],
                "chunks": event["chunks"],
                "mean_loss": round(event["mean_loss"], 6),
                "mean_kl_loss": None if event["mean_kl_loss"] is None else round(event["mean_kl_loss"], 6),
                "mean_accuracy": round(event["mean_accuracy"], 4),
                "elapsed_sec": event["elapsed_sec"],
            },
            flush=True,
        )
        if ep_idx % args.eval_every_episodes == 0 or ep_idx == args.train_episodes:
            best, stats = _maybe_eval(
                policy=policy,
                board_size=args.board_size,
                eval_episodes=args.eval_episodes,
                eval_seed=args.eval_seed,
                eval_seeds=eval_seed_list,
                device=args.device,
                flood_fill=args.flood_fill,
                head_centered=args.head_centered,
                use_prev_action_input=args.prev_action_input,
                use_fill_input=args.fill_input,
                save_path=args.save_path,
                best=best,
                events=events,
                label=f"episode_{ep_idx}",
            )
            print(
                {
                    "eval_after_episode": ep_idx,
                    "mean_score": round(float(stats["mean_score"]), 3),
                    "win_rate": round(float(stats["win_rate"]), 4),
                    "phase_lt20_rate": round(float(stats["phase_lt20_rate"]), 4),
                },
                flush=True,
            )

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
    print(
        {
            "saved": str(args.save_path),
            "summary": str(args.save_path.with_name(f"{args.save_path.stem}.summary.json")),
            "elapsed_sec": round(time.time() - started, 1),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
