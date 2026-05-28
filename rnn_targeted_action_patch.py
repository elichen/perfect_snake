"""Patch an RNN policy head toward harvested shorter action trajectories."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distill.rnn_model import SnakeRNNPolicy, load_rnn_policy_state
from rnn_cycle_shortcut_patch import (
    _parse_floats,
    _parse_ints,
    _pop_samples,
    _save_atomic,
    _score_results,
    _train_head,
)
from rnn_eval_seeds_batch import eval_seed_batch
from snake_env import SnakeEnv


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


def _load_improved_records(path: Path, *, keep_all: bool = False) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "results" in payload:
        records = [
            record
            for seed_result in payload["results"]
            for record in seed_result.get("improved", [])
        ]
    else:
        records = payload.get("improved", [])
    if not records:
        raise ValueError(f"{path} does not contain improved records")
    if keep_all:
        return records
    best_by_seed: dict[int, dict[str, Any]] = {}
    for record in records:
        seed = int(record["seed"])
        current = best_by_seed.get(seed)
        if current is None or int(record["improvement"]) > int(current["improvement"]):
            best_by_seed[seed] = record
    return [best_by_seed[seed] for seed in sorted(best_by_seed)]


def _make_policy(
    *,
    board_size: int,
    hidden_size: int,
    early_head_max_fill: float | None,
    residual_policy_head: bool,
    residual_policy_min_fill: float | None,
    device: str,
    state: dict[str, Any],
) -> SnakeRNNPolicy:
    policy = SnakeRNNPolicy(
        board_size=board_size,
        n_channels=5,
        hidden_size=hidden_size,
        early_head_max_fill=early_head_max_fill,
        residual_policy_head=residual_policy_head,
        residual_policy_min_fill=residual_policy_min_fill,
    ).to(device)
    load_rnn_policy_state(policy, state)
    return policy


@torch.no_grad()
def _collect_trajectory_record(
    *,
    policy,
    board_size: int,
    device: str,
    record: dict[str, Any],
    sample_stride: int,
    target_weight: float,
    trajectory_weight: float,
    target_kl_weight: float,
    trajectory_kl_weight: float,
    train_residual_head: bool,
) -> dict[str, Any]:
    seed = int(record["seed"])
    actions = [int(action) for action in record["actions"]]
    deviation_step = int(record["deviation_step"])
    env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
    obs, _ = env.reset(seed=seed)
    hidden = policy.initial_state(1, device)
    features: list[torch.Tensor] = []
    labels: list[int] = []
    weights: list[float] = []
    base_logits: list[torch.Tensor] = []
    kl_weights: list[float] = []
    samples = 0
    info: dict[str, Any] = {}

    for step, action in enumerate(actions):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        encoded = policy.encoder(obs_t)
        hidden = policy.gru_cell(encoded, hidden)
        logits = policy.policy_head(hidden)
        if train_residual_head and getattr(policy, "residual_policy_head_enabled", False):
            logits = logits + policy.residual_policy_head(hidden)
        if step == deviation_step or step % sample_stride == 0:
            is_target = step == deviation_step
            features.append(hidden.squeeze(0).detach().cpu())
            labels.append(action)
            weights.append(float(target_weight if is_target else trajectory_weight))
            base_logits.append(logits.squeeze(0).detach().cpu())
            kl_weights.append(float(target_kl_weight if is_target else trajectory_kl_weight))
            samples += 1
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    score = int(info.get("score", env.score))
    return {
        "seed": seed,
        "deviation_step": deviation_step,
        "deviation_score": int(record["deviation_score"]),
        "features": features,
        "labels": labels,
        "weights": weights,
        "base_logits": base_logits,
        "kl_weights": kl_weights,
        "samples": samples,
        "score": score,
        "reason": info.get("reason"),
        "steps": int(info.get("steps", env.total_steps)),
    }


@torch.no_grad()
def _collect_anchor_seed(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seed: int,
    device: str,
    max_steps: int,
    anchor_stride: int,
    anchor_weight: float,
    anchor_kl_weight: float,
    early_head_max_fill: float | None,
    train_residual_head: bool,
) -> dict[str, Any]:
    env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
    obs, _ = env.reset(seed=seed)
    hidden = policy.initial_state(1, device)
    features: list[torch.Tensor] = []
    labels: list[int] = []
    weights: list[float] = []
    base_logits: list[torch.Tensor] = []
    kl_weights: list[float] = []
    samples = 0
    info: dict[str, Any] = {}

    for step in range(max_steps):
        fill_value = env.snake_length / float(board_size * board_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        fill_t = None
        if early_head_max_fill is not None:
            fill_t = torch.as_tensor([fill_value], dtype=torch.float32, device=device)
        logits, hidden = policy.forward_step(obs_t, hidden, fill_values=fill_t)
        action = int(torch.argmax(logits, dim=-1).item())

        main_head_active = early_head_max_fill is None or fill_value >= early_head_max_fill
        should_anchor = train_residual_head or main_head_active
        if should_anchor and step % anchor_stride == 0:
            main_logits = logits if train_residual_head else policy.policy_head(hidden)
            features.append(hidden.squeeze(0).detach().cpu())
            labels.append(action)
            weights.append(float(anchor_weight))
            base_logits.append(main_logits.squeeze(0).detach().cpu())
            kl_weights.append(float(anchor_kl_weight))
            samples += 1

        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "features": features,
        "labels": labels,
        "weights": weights,
        "base_logits": base_logits,
        "kl_weights": kl_weights,
        "samples": samples,
        "score": int(info.get("score", env.score)),
        "reason": info.get("reason"),
        "steps": int(info.get("steps", env.total_steps)),
    }


@torch.no_grad()
def _collect_anchor_seeds_batch(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seeds: list[int],
    device: str,
    max_steps: int,
    anchor_stride: int,
    anchor_weight: float,
    anchor_kl_weight: float,
    early_head_max_fill: float | None,
    train_residual_head: bool,
) -> list[dict[str, Any]]:
    envs = [SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed) for seed in seeds]
    observations = []
    for env, seed in zip(envs, seeds):
        obs, _ = env.reset(seed=seed)
        observations.append(obs)

    hidden = policy.initial_state(len(seeds), device)
    done = [False for _ in seeds]
    records = [
        {
            "seed": int(seed),
            "features": [],
            "labels": [],
            "weights": [],
            "base_logits": [],
            "kl_weights": [],
            "samples": 0,
            "score": 0,
            "reason": None,
            "steps": 0,
        }
        for seed in seeds
    ]

    for _ in range(max_steps):
        active = [idx for idx, is_done in enumerate(done) if not is_done]
        if not active:
            break

        active_t = torch.as_tensor(active, dtype=torch.long, device=device)
        obs_t = torch.as_tensor(
            np.stack([observations[idx] for idx in active], axis=0),
            dtype=torch.float32,
            device=device,
        )
        fill_t = None
        fill_values = None
        if early_head_max_fill is not None:
            fill_values = [
                envs[idx].snake_length / float(board_size * board_size)
                for idx in active
            ]
            fill_t = torch.as_tensor(fill_values, dtype=torch.float32, device=device)

        logits, active_hidden = policy.forward_step(
            obs_t,
            hidden.index_select(0, active_t),
            fill_values=fill_t,
        )
        hidden[active_t] = active_hidden
        actions = torch.argmax(logits, dim=-1).detach().cpu().tolist()

        for local_idx, (idx, action) in enumerate(zip(active, actions)):
            env = envs[idx]
            step = int(env.total_steps)
            main_head_active = early_head_max_fill is None
            if early_head_max_fill is not None and fill_values is not None:
                main_head_active = fill_values[local_idx] >= early_head_max_fill
            should_anchor = train_residual_head or main_head_active
            if should_anchor and step % anchor_stride == 0:
                feature = active_hidden[local_idx].detach().cpu()
                main_logits = logits[local_idx]
                if not train_residual_head:
                    main_logits = policy.policy_head(active_hidden[local_idx].unsqueeze(0)).squeeze(0)
                record = records[idx]
                record["features"].append(feature)
                record["labels"].append(int(action))
                record["weights"].append(float(anchor_weight))
                record["base_logits"].append(main_logits.detach().cpu())
                record["kl_weights"].append(float(anchor_kl_weight))
                record["samples"] += 1

            obs, _, terminated, truncated, info = env.step(int(action))
            observations[idx] = obs
            if terminated or truncated:
                done[idx] = True
                records[idx]["score"] = int(info.get("score", env.score))
                records[idx]["reason"] = info.get("reason")
                records[idx]["steps"] = int(info.get("steps", env.total_steps))

    for idx, record in enumerate(records):
        if not done[idx]:
            env = envs[idx]
            record["score"] = int(env.score)
            record["reason"] = "max_steps"
            record["steps"] = int(env.total_steps)
    return records


@torch.no_grad()
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
        progress_every=5,
        stop_after_failures=int(fail_fast),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Search policy-head patches from harvested action trajectories")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--early-head-max-fill", type=float, default=None)
    parser.add_argument("--train-residual-head", action="store_true")
    parser.add_argument("--residual-min-fill", type=float, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--anchor-seeds", type=_parse_ints, default=None)
    parser.add_argument("--anchor-ranges", type=_parse_ranges, default=None)
    parser.add_argument("--gate-seeds", type=_parse_ints, default=None)
    parser.add_argument("--gate-ranges", type=_parse_ranges, default=None)
    parser.add_argument("--lrs", type=_parse_floats, required=True)
    parser.add_argument("--epochs", type=_parse_ints, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--trajectory-sample-stride", type=int, default=100)
    parser.add_argument("--anchor-stride", type=int, default=100)
    parser.add_argument("--target-weight", type=float, default=20.0)
    parser.add_argument("--trajectory-weight", type=float, default=0.5)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--target-kl-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-kl-weight", type=float, default=0.01)
    parser.add_argument("--anchor-kl-weight", type=float, default=0.2)
    parser.add_argument("--batch-anchor-collect", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--keep-all-records", action="store_true")
    args = parser.parse_args()

    args.anchor_seeds = _unique_ordered((args.anchor_seeds or []) + (args.anchor_ranges or []))
    args.gate_seeds = _unique_ordered((args.gate_seeds or []) + (args.gate_ranges or []))
    if not args.anchor_seeds:
        raise SystemExit("provide --anchor-seeds and/or --anchor-ranges")
    if not args.gate_seeds:
        raise SystemExit("provide --gate-seeds and/or --gate-ranges")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_state = torch.load(args.base, map_location="cpu")
    base_policy = _make_policy(
        board_size=args.board_size,
        hidden_size=args.hidden_size,
        early_head_max_fill=args.early_head_max_fill,
        residual_policy_head=args.train_residual_head,
        residual_policy_min_fill=args.residual_min_fill,
        device=args.device,
        state=base_state,
    )
    base_policy.eval()
    dataset: dict[str, list[Any]] = {
        "features": [],
        "labels": [],
        "weights": [],
        "base_logits": [],
        "kl_weights": [],
    }
    seed_records = []
    for record in _load_improved_records(args.trajectory_json, keep_all=args.keep_all_records):
        collected = _collect_trajectory_record(
            policy=base_policy,
            board_size=args.board_size,
            device=args.device,
            record=record,
            sample_stride=max(1, args.trajectory_sample_stride),
            target_weight=args.target_weight,
            trajectory_weight=args.trajectory_weight,
            target_kl_weight=args.target_kl_weight,
            trajectory_kl_weight=args.trajectory_kl_weight,
            train_residual_head=args.train_residual_head,
        )
        _pop_samples(collected, dataset)
        collected["role"] = "trajectory"
        seed_records.append(collected)
        print({"collect": collected}, flush=True)

    if args.batch_anchor_collect:
        for collected in _collect_anchor_seeds_batch(
            policy=base_policy,
            board_size=args.board_size,
            seeds=args.anchor_seeds,
            device=args.device,
            max_steps=args.max_steps,
            anchor_stride=max(1, args.anchor_stride),
            anchor_weight=args.anchor_weight,
            anchor_kl_weight=args.anchor_kl_weight,
            early_head_max_fill=args.early_head_max_fill,
            train_residual_head=args.train_residual_head,
        ):
            _pop_samples(collected, dataset)
            collected["role"] = "anchor"
            seed_records.append(collected)
            print({"collect": collected}, flush=True)
    else:
        for seed in args.anchor_seeds:
            collected = _collect_anchor_seed(
                policy=base_policy,
                board_size=args.board_size,
                seed=seed,
                device=args.device,
                max_steps=args.max_steps,
                anchor_stride=max(1, args.anchor_stride),
                anchor_weight=args.anchor_weight,
                anchor_kl_weight=args.anchor_kl_weight,
                early_head_max_fill=args.early_head_max_fill,
                train_residual_head=args.train_residual_head,
            )
            _pop_samples(collected, dataset)
            collected["role"] = "anchor"
            seed_records.append(collected)
            print({"collect": collected}, flush=True)

    tensor_dataset = {
        "features": torch.stack(dataset["features"], dim=0),
        "labels": torch.tensor(dataset["labels"], dtype=torch.long),
        "weights": torch.tensor(dataset["weights"], dtype=torch.float32),
        "base_logits": torch.stack(dataset["base_logits"], dim=0),
        "kl_weights": torch.tensor(dataset["kl_weights"], dtype=torch.float32),
    }
    dataset_summary = {
        "samples": int(tensor_dataset["labels"].shape[0]),
        "trajectory_json": str(args.trajectory_json),
        "anchor_seeds": args.anchor_seeds,
        "gate_seeds": args.gate_seeds,
        "seed_records": seed_records,
    }
    print({"dataset": dataset_summary}, flush=True)

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
                    residual_policy_min_fill=args.residual_min_fill,
                    device=args.device,
                    state=base_state,
                )
                train_stats = _train_head(
                    policy=policy,
                    dataset=tensor_dataset,
                    lr=lr,
                    epochs=epoch_count,
                    batch_size=args.batch_size,
                    device=args.device,
                    head_name="residual_policy_head" if args.train_residual_head else "policy_head",
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
