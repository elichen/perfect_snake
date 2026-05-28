"""Sequence-level late-failure repair for recurrent Snake checkpoints.

This keeps inference as the same pure RNN policy. The script only uses stored
training-time deviation labels and reference-policy anchor windows.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from distill.rnn_model import SnakeRNNPolicy, load_rnn_policy_state
from rnn_cycle_shortcut_patch import _parse_floats, _parse_ints, _save_atomic, _score_results
from rnn_eval_seeds_batch import eval_seed_batch
from rnn_targeted_action_patch import _load_improved_records
from snake_env import SnakeEnv

START_ACTION_TOKEN = 3


@dataclass
class Window:
    observations: np.ndarray
    actions: np.ndarray
    prev_actions: np.ndarray
    fill_values: np.ndarray
    ce_weights: np.ndarray
    kl_weights: np.ndarray
    meta: dict[str, Any]


def _make_policy(
    *,
    board_size: int,
    hidden_size: int,
    early_head_max_fill: float | None,
    residual_policy_head: bool,
    device: str,
    state: dict[str, Any],
) -> SnakeRNNPolicy:
    policy = SnakeRNNPolicy(
        board_size=board_size,
        n_channels=5,
        hidden_size=hidden_size,
        early_head_max_fill=early_head_max_fill,
        residual_policy_head=residual_policy_head,
    ).to(device)
    load_rnn_policy_state(policy, state)
    return policy


def _parse_ranges(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        spec = part.strip()
        if not spec:
            continue
        if ":" not in spec:
            raise argparse.ArgumentTypeError("ranges must use start:count")
        start_s, count_s = spec.split(":", 1)
        start = int(start_s)
        count = int(count_s)
        if count < 1:
            raise argparse.ArgumentTypeError("range count must be >= 1")
        seeds.extend(range(start, start + count))
    if not seeds:
        raise argparse.ArgumentTypeError("expected at least one range")
    return seeds


def _unique_ordered(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _empty_window(window_len: int, obs_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observations = np.zeros((window_len, *obs_shape), dtype=np.float32)
    actions = np.zeros((window_len,), dtype=np.int64)
    prev_actions = np.full((window_len,), START_ACTION_TOKEN, dtype=np.int64)
    fill_values = np.zeros((window_len,), dtype=np.float32)
    return observations, actions, prev_actions, fill_values


def _pack_window(
    *,
    transitions: list[dict[str, Any]],
    window_len: int,
    obs_shape: tuple[int, ...],
    ce_weight_fn,
    kl_weight_fn,
    meta: dict[str, Any],
) -> Window:
    observations, actions, prev_actions, fill_values = _empty_window(window_len, obs_shape)
    ce_weights = np.zeros((window_len,), dtype=np.float32)
    kl_weights = np.zeros((window_len,), dtype=np.float32)
    kept = transitions[-window_len:]
    for offset, transition in enumerate(kept):
        observations[offset] = transition["obs"]
        actions[offset] = int(transition["action"])
        prev_actions[offset] = int(transition["prev_action"])
        fill_values[offset] = float(transition["fill"])
        ce_weights[offset] = float(ce_weight_fn(transition))
        kl_weights[offset] = float(kl_weight_fn(transition))
    return Window(
        observations=observations,
        actions=actions,
        prev_actions=prev_actions,
        fill_values=fill_values,
        ce_weights=ce_weights,
        kl_weights=kl_weights,
        meta=meta,
    )


def _collect_target_window(
    *,
    board_size: int,
    record: dict[str, Any],
    window_len: int,
    context_before: int,
    context_after: int,
    target_weight: float,
    trajectory_weight: float,
    target_kl_weight: float,
    trajectory_kl_weight: float,
) -> Window:
    seed = int(record["seed"])
    actions = [int(action) for action in record["actions"]]
    deviation_step = int(record["deviation_step"])
    collect_start = max(0, deviation_step - context_before)
    collect_end = min(len(actions) - 1, deviation_step + context_after)

    env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
    obs, _ = env.reset(seed=seed)
    obs_shape = tuple(obs.shape)
    prev_action = START_ACTION_TOKEN
    transitions: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}

    for step, action in enumerate(actions):
        if collect_start <= step <= collect_end:
            transitions.append(
                {
                    "step": step,
                    "obs": obs.astype(np.float32, copy=True),
                    "action": action,
                    "prev_action": prev_action,
                    "fill": env.snake_length / float(board_size * board_size),
                }
            )
        obs, _, terminated, truncated, final_info = env.step(action)
        prev_action = action
        if terminated or truncated:
            break
        if step >= collect_end:
            break

    if not transitions:
        raise RuntimeError(f"target record for seed {seed} produced no transitions")

    def ce_weight(transition: dict[str, Any]) -> float:
        return target_weight if int(transition["step"]) == deviation_step else trajectory_weight

    def kl_weight(transition: dict[str, Any]) -> float:
        return target_kl_weight if int(transition["step"]) == deviation_step else trajectory_kl_weight

    return _pack_window(
        transitions=transitions,
        window_len=window_len,
        obs_shape=obs_shape,
        ce_weight_fn=ce_weight,
        kl_weight_fn=kl_weight,
        meta={
            "role": "target",
            "seed": seed,
            "deviation_step": deviation_step,
            "deviation_score": int(record["deviation_score"]),
            "base_action": int(record["base_action"]),
            "alt_action": int(record["alt_action"]),
            "score": int(final_info.get("score", env.score)),
            "reason": final_info.get("reason"),
            "steps": int(final_info.get("steps", env.total_steps)),
            "transitions": len(transitions),
        },
    )


@torch.no_grad()
def _collect_anchor_windows(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seeds: list[int],
    window_len: int,
    device: str,
    max_steps: int,
    anchor_stride: int,
    anchor_min_fill: float,
    anchor_weight: float,
    anchor_kl_weight: float,
    max_anchor_windows: int,
    early_head_max_fill: float | None,
) -> list[Window]:
    windows: list[Window] = []
    for seed in seeds:
        seed_window_start = len(windows)
        env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
        obs, _ = env.reset(seed=seed)
        obs_shape = tuple(obs.shape)
        hidden = policy.initial_state(1, device)
        prev_action = START_ACTION_TOKEN
        transitions: list[dict[str, Any]] = []
        info: dict[str, Any] = {}

        for _ in range(max_steps):
            fill = env.snake_length / float(board_size * board_size)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            fill_t = None
            if early_head_max_fill is not None:
                fill_t = torch.as_tensor([fill], dtype=torch.float32, device=device)
            logits, hidden = policy.forward_step(obs_t, hidden, fill_values=fill_t)
            action = int(torch.argmax(logits, dim=-1).item())
            step = int(env.total_steps)
            transitions.append(
                {
                    "step": step,
                    "obs": obs.astype(np.float32, copy=True),
                    "action": action,
                    "prev_action": prev_action,
                    "fill": fill,
                }
            )
            if (
                len(transitions) >= 1
                and fill >= anchor_min_fill
                and step % anchor_stride == 0
                and (max_anchor_windows <= 0 or len(windows) < max_anchor_windows)
            ):
                windows.append(
                    _pack_window(
                        transitions=transitions,
                        window_len=window_len,
                        obs_shape=obs_shape,
                        ce_weight_fn=lambda _transition: anchor_weight,
                        kl_weight_fn=lambda _transition: anchor_kl_weight,
                        meta={"role": "anchor", "seed": seed, "end_step": step, "fill": fill},
                    )
                )
            obs, _, terminated, truncated, info = env.step(action)
            prev_action = action
            if terminated or truncated:
                break
            if max_anchor_windows > 0 and len(windows) >= max_anchor_windows:
                break

        if (
            len(windows) == seed_window_start
            and transitions
            and float(transitions[-1]["fill"]) >= anchor_min_fill
            and (max_anchor_windows <= 0 or len(windows) < max_anchor_windows)
        ):
            windows.append(
                _pack_window(
                    transitions=transitions,
                    window_len=window_len,
                    obs_shape=obs_shape,
                    ce_weight_fn=lambda _transition: anchor_weight,
                    kl_weight_fn=lambda _transition: anchor_kl_weight,
                    meta={
                        "role": "anchor",
                        "seed": seed,
                        "end_step": int(transitions[-1]["step"]),
                        "fill": float(transitions[-1]["fill"]),
                        "fallback": True,
                    },
                )
            )

        print(
            {
                "anchor_seed": seed,
                "score": int(info.get("score", env.score)),
                "reason": info.get("reason"),
                "steps": int(info.get("steps", env.total_steps)),
                "anchor_windows_total": len(windows),
            },
            flush=True,
        )
        if max_anchor_windows > 0 and len(windows) >= max_anchor_windows:
            break
    return windows


def _stack_windows(windows: list[Window], indexes: torch.Tensor) -> dict[str, torch.Tensor]:
    ids = indexes.detach().cpu().tolist()
    return {
        "observations": torch.as_tensor(np.stack([windows[i].observations for i in ids], axis=1), dtype=torch.float32),
        "actions": torch.as_tensor(np.stack([windows[i].actions for i in ids], axis=1), dtype=torch.long),
        "prev_actions": torch.as_tensor(np.stack([windows[i].prev_actions for i in ids], axis=1), dtype=torch.long),
        "fill_values": torch.as_tensor(np.stack([windows[i].fill_values for i in ids], axis=1), dtype=torch.float32),
        "ce_weights": torch.as_tensor(np.stack([windows[i].ce_weights for i in ids], axis=1), dtype=torch.float32),
        "kl_weights": torch.as_tensor(np.stack([windows[i].kl_weights for i in ids], axis=1), dtype=torch.float32),
    }


def _train_sequence_candidate(
    *,
    policy: SnakeRNNPolicy,
    anchor_policy: SnakeRNNPolicy,
    windows: list[Window],
    lr: float,
    epochs: int,
    batch_size: int,
    device: str,
    early_head_max_fill: float | None,
    kl_coef: float,
    train_residual_head: bool,
) -> dict[str, Any]:
    if train_residual_head:
        for param in policy.parameters():
            param.requires_grad = False
        for param in policy.residual_policy_head.parameters():
            param.requires_grad = True
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    anchor_policy.eval()
    for param in anchor_policy.parameters():
        param.requires_grad_(False)

    n_windows = len(windows)
    losses: list[float] = []
    ce_losses: list[float] = []
    kl_losses: list[float] = []
    for _ in range(epochs):
        order = torch.randperm(n_windows)
        for start in range(0, n_windows, batch_size):
            index = order[start : start + batch_size]
            batch = {key: value.to(device) for key, value in _stack_windows(windows, index).items()}
            fill_for_policy = batch["fill_values"] if early_head_max_fill is not None else None
            logits, _ = policy.forward_sequence(batch["observations"], fill_values=fill_for_policy)
            with torch.no_grad():
                anchor_logits, _ = anchor_policy.forward_sequence(batch["observations"], fill_values=fill_for_policy)
                anchor_log_probs = F.log_softmax(anchor_logits, dim=-1)
                anchor_probs = anchor_log_probs.exp()

            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch["actions"].reshape(-1),
                reduction="none",
            ).reshape_as(batch["actions"])
            ce_denom = batch["ce_weights"].sum().clamp_min(1e-6)
            ce_loss = (ce * batch["ce_weights"]).sum() / ce_denom
            current_log_probs = F.log_softmax(logits, dim=-1)
            kl = (anchor_probs * (anchor_log_probs - current_log_probs)).sum(dim=-1)
            kl_denom = batch["kl_weights"].sum().clamp_min(1e-6)
            kl_loss = (kl * batch["kl_weights"]).sum() / kl_denom
            loss = ce_loss + kl_coef * kl_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            ce_losses.append(float(ce_loss.item()))
            kl_losses.append(float(kl_loss.item()))

    return {
        "windows": n_windows,
        "epochs": epochs,
        "lr": lr,
        "kl_coef": kl_coef,
        "train_residual_head": train_residual_head,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "mean_ce": float(np.mean(ce_losses)) if ce_losses else 0.0,
        "mean_kl": float(np.mean(kl_losses)) if kl_losses else 0.0,
    }


def _eval_gate(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seeds: list[int],
    device: str,
    max_steps: int,
    fail_fast: bool,
    early_head_max_fill: float | None,
) -> list[dict[str, Any]]:
    policy.eval()
    return eval_seed_batch(
        policy=policy,
        board_size=board_size,
        seeds=seeds,
        device=device,
        max_steps=max_steps,
        use_fill_values=early_head_max_fill is not None,
        progress_every=10,
        stop_after_failures=int(fail_fast),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequence-level RNN late-failure repair search")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--early-head-max-fill", type=float, default=None)
    parser.add_argument("--train-residual-head", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--anchor-seeds", type=_parse_ints, default=None)
    parser.add_argument("--anchor-ranges", type=_parse_ranges, default=None)
    parser.add_argument("--target-seeds", type=_parse_ints, default=None)
    parser.add_argument("--gate-seeds", type=_parse_ints, required=True)
    parser.add_argument("--lrs", type=_parse_floats, required=True)
    parser.add_argument("--epochs", type=_parse_ints, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-len", type=int, default=544)
    parser.add_argument("--target-context-before", type=int, default=512)
    parser.add_argument("--target-context-after", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--anchor-stride", type=int, default=1000)
    parser.add_argument("--anchor-min-fill", type=float, default=0.90)
    parser.add_argument("--max-anchor-windows", type=int, default=200)
    parser.add_argument("--target-weight", type=float, default=50.0)
    parser.add_argument("--trajectory-weight", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--target-kl-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-kl-weight", type=float, default=0.1)
    parser.add_argument("--anchor-kl-weight", type=float, default=1.0)
    parser.add_argument("--kl-coef", type=float, default=1.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--keep-all-records", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    if args.window_len < 1:
        raise SystemExit("--window-len must be >= 1")
    if args.target_context_before < 0 or args.target_context_after < 0:
        raise SystemExit("--target-context-before/after must be >= 0")
    if args.anchor_stride < 1:
        raise SystemExit("--anchor-stride must be >= 1")
    if not (0.0 <= args.anchor_min_fill <= 1.0):
        raise SystemExit("--anchor-min-fill must be in [0, 1]")
    if args.kl_coef < 0.0:
        raise SystemExit("--kl-coef must be >= 0")
    anchor_seeds = _unique_ordered((args.anchor_seeds or []) + (args.anchor_ranges or []))
    if not anchor_seeds:
        raise SystemExit("provide --anchor-seeds, --anchor-ranges, or both")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_state = torch.load(args.base, map_location="cpu")
    base_policy = _make_policy(
        board_size=args.board_size,
        hidden_size=args.hidden_size,
        early_head_max_fill=args.early_head_max_fill,
        residual_policy_head=args.train_residual_head,
        device=args.device,
        state=base_state,
    )
    base_policy.eval()

    windows: list[Window] = []
    target_seed_filter = set(args.target_seeds or [])
    target_records = [
        record
        for record in _load_improved_records(args.trajectory_json, keep_all=args.keep_all_records)
        if not target_seed_filter or int(record["seed"]) in target_seed_filter
    ]
    if not target_records:
        raise SystemExit("--target-seeds filtered out all trajectory records")

    for record in target_records:
        window = _collect_target_window(
            board_size=args.board_size,
            record=record,
            window_len=args.window_len,
            context_before=args.target_context_before,
            context_after=args.target_context_after,
            target_weight=args.target_weight,
            trajectory_weight=args.trajectory_weight,
            target_kl_weight=args.target_kl_weight,
            trajectory_kl_weight=args.trajectory_kl_weight,
        )
        windows.append(window)
        print({"target_window": window.meta}, flush=True)

    windows.extend(
        _collect_anchor_windows(
            policy=base_policy,
            board_size=args.board_size,
            seeds=anchor_seeds,
            window_len=args.window_len,
            device=args.device,
            max_steps=args.max_steps,
            anchor_stride=args.anchor_stride,
            anchor_min_fill=args.anchor_min_fill,
            anchor_weight=args.anchor_weight,
            anchor_kl_weight=args.anchor_kl_weight,
            max_anchor_windows=args.max_anchor_windows,
            early_head_max_fill=args.early_head_max_fill,
        )
    )
    dataset_summary = {
        "base": str(args.base),
        "trajectory_json": str(args.trajectory_json),
        "target_seeds": args.target_seeds,
        "anchor_seeds": anchor_seeds,
        "windows": len(windows),
        "target_windows": sum(1 for window in windows if window.meta.get("role") == "target"),
        "anchor_windows": sum(1 for window in windows if window.meta.get("role") == "anchor"),
        "window_len": args.window_len,
        "window_meta": [window.meta for window in windows],
    }
    (args.out_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print({"dataset": {key: value for key, value in dataset_summary.items() if key != "window_meta"}}, flush=True)
    if args.collect_only:
        return 0

    best_key: tuple[int, float, float] | None = None
    best_record: dict[str, Any] | None = None
    started = time.time()
    with (args.out_dir / "search.jsonl").open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"dataset": dataset_summary}, sort_keys=True) + "\n")
        log_file.flush()
        for lr in args.lrs:
            for epoch_count in args.epochs:
                candidate_started = time.time()
                policy = _make_policy(
                    board_size=args.board_size,
                    hidden_size=args.hidden_size,
                    early_head_max_fill=args.early_head_max_fill,
                    residual_policy_head=args.train_residual_head,
                    device=args.device,
                    state=base_state,
                )
                anchor_policy = _make_policy(
                    board_size=args.board_size,
                    hidden_size=args.hidden_size,
                    early_head_max_fill=args.early_head_max_fill,
                    residual_policy_head=args.train_residual_head,
                    device=args.device,
                    state=base_state,
                )
                train_stats = _train_sequence_candidate(
                    policy=policy,
                    anchor_policy=anchor_policy,
                    windows=windows,
                    lr=lr,
                    epochs=epoch_count,
                    batch_size=args.batch_size,
                    device=args.device,
                    early_head_max_fill=args.early_head_max_fill,
                    kl_coef=args.kl_coef,
                    train_residual_head=args.train_residual_head,
                )
                gate_results = _eval_gate(
                    policy=policy,
                    board_size=args.board_size,
                    seeds=args.gate_seeds,
                    device=args.device,
                    max_steps=args.max_steps,
                    fail_fast=args.fail_fast,
                    early_head_max_fill=args.early_head_max_fill,
                )
                key = _score_results(gate_results)
                lr_label = f"{lr:.2e}".replace("+", "").replace(".", "p")
                candidate_path = args.out_dir / f"lr{lr_label}_ep{epoch_count}.pt"
                saved_path = None
                if args.save_all or best_key is None or key > best_key:
                    _save_atomic(policy.state_dict(), candidate_path)
                    saved_path = str(candidate_path)
                if best_key is None or key > best_key:
                    best_key = key
                    best_record = {
                        "checkpoint": str(candidate_path),
                        "key": key,
                        "gate_results": gate_results,
                    }
                win_steps = [
                    int(result["steps"])
                    for result in gate_results
                    if result["win"] and result.get("steps") is not None
                ]
                record = {
                    "lr": lr,
                    "epochs": epoch_count,
                    "train_stats": train_stats,
                    "gate_results": gate_results,
                    "wins": key[0],
                    "gate_count": len(gate_results),
                    "mean_score": float(np.mean([result["score"] for result in gate_results])),
                    "mean_win_steps": float(np.mean(win_steps)) if win_steps else None,
                    "saved_path": saved_path,
                    "elapsed_sec": round(time.time() - candidate_started, 1),
                    "total_elapsed_sec": round(time.time() - started, 1),
                }
                print(record, flush=True)
                log_file.write(json.dumps(record, sort_keys=True) + "\n")
                log_file.flush()

    summary = {
        "dataset": dataset_summary,
        "best": best_record,
        "elapsed_sec": round(time.time() - started, 1),
        "args": vars(args),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print({"summary": str(args.out_dir / "summary.json"), "best": best_record}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
