"""Train a low-fill RNN early head from harvested shorter trajectories."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from distill.rnn_model import SnakeRNNPolicy, load_rnn_policy_state
from rnn_eval_seeds_batch import eval_seed_batch
from rnn_targeted_action_patch import _load_improved_records
from snake_env import SnakeEnv


def _parse_ints(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _parse_floats(value: str) -> list[float]:
    result = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one float")
    return result


def _save_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _make_policy(
    *,
    board_size: int,
    hidden_size: int,
    early_head_max_fill: float,
    device: str,
    state: dict[str, Any],
) -> SnakeRNNPolicy:
    policy = SnakeRNNPolicy(
        board_size=board_size,
        n_channels=5,
        hidden_size=hidden_size,
        early_head_max_fill=early_head_max_fill,
    ).to(device)
    load_rnn_policy_state(policy, state)
    return policy


@torch.no_grad()
def _collect_actions(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    device: str,
    seed: int,
    actions: list[int] | None,
    max_steps: int,
    early_head_max_fill: float,
    stride: int,
    sample_steps: set[int] | None,
    weight: float,
    kl_weight: float,
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
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        encoded = policy.encoder(obs_t)
        hidden = policy.gru_cell(encoded, hidden)
        logits = policy.policy_head(hidden)
        if actions is None:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            if step >= len(actions):
                break
            action = int(actions[step])

        fill = env.snake_length / float(board_size * board_size)
        should_sample = step in sample_steps if sample_steps is not None else step % stride == 0
        if fill < early_head_max_fill and should_sample:
            features.append(hidden.squeeze(0).detach().cpu())
            labels.append(action)
            weights.append(float(weight))
            base_logits.append(logits.squeeze(0).detach().cpu())
            kl_weights.append(float(kl_weight))
            samples += 1

        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        if env.snake_length / float(board_size * board_size) >= early_head_max_fill:
            info = {"reason": "fill_cutoff", "score": env.score, "steps": env.total_steps}
            break

    score = int(info.get("score", env.score))
    return {
        "seed": seed,
        "features": features,
        "labels": labels,
        "weights": weights,
        "base_logits": base_logits,
        "kl_weights": kl_weights,
        "samples": samples,
        "score": score,
        "reason": str(info.get("reason", "timeout")),
        "steps": int(info.get("steps", env.total_steps)),
    }


def _pop_samples(record: dict[str, Any], dataset: dict[str, list[Any]]) -> None:
    for key in ("features", "labels", "weights", "base_logits", "kl_weights"):
        dataset[key].extend(record.pop(key))


def _train_early_head(
    *,
    policy: SnakeRNNPolicy,
    dataset: dict[str, torch.Tensor],
    lr: float,
    epochs: int,
    batch_size: int,
    device: str,
    train_final_layer_only: bool,
    l2_weight: float,
) -> dict[str, Any]:
    for name, param in policy.named_parameters():
        if train_final_layer_only:
            param.requires_grad = name.startswith("early_policy_head.2.")
        else:
            param.requires_grad = name.startswith("early_policy_head")
    optimizer = torch.optim.Adam([param for param in policy.parameters() if param.requires_grad], lr=lr)
    l2_refs = [
        param.detach().clone()
        for param in policy.early_policy_head[-1].parameters()
    ] if train_final_layer_only and l2_weight > 0 else []
    features = dataset["features"].to(device)
    labels = dataset["labels"].to(device)
    weights = dataset["weights"].to(device)
    base_probs = F.softmax(dataset["base_logits"].to(device), dim=-1)
    kl_weights = dataset["kl_weights"].to(device)
    n_samples = int(labels.shape[0])
    losses: list[float] = []
    ce_losses: list[float] = []
    kl_losses: list[float] = []

    policy.train()
    for _ in range(epochs):
        order = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, batch_size):
            index = order[start : start + batch_size]
            logits = policy.early_policy_head(features[index])
            ce = F.cross_entropy(logits, labels[index], reduction="none")
            weighted_ce = (ce * weights[index]).sum() / weights[index].sum().clamp_min(1e-6)
            kl_per_sample = F.kl_div(
                F.log_softmax(logits, dim=-1),
                base_probs[index],
                reduction="none",
            ).sum(dim=-1)
            weighted_kl = (kl_per_sample * kl_weights[index]).sum() / kl_weights[index].sum().clamp_min(1e-6)
            loss = weighted_ce + weighted_kl
            if l2_refs:
                l2 = sum(
                    torch.sum((param - ref.to(device)) ** 2)
                    for param, ref in zip(policy.early_policy_head[-1].parameters(), l2_refs)
                )
                loss = loss + l2_weight * l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            ce_losses.append(float(weighted_ce.item()))
            kl_losses.append(float(weighted_kl.item()))

    return {
        "samples": n_samples,
        "epochs": epochs,
        "lr": lr,
        "train_final_layer_only": train_final_layer_only,
        "l2_weight": l2_weight,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "mean_ce": float(np.mean(ce_losses)) if ce_losses else 0.0,
        "mean_kl": float(np.mean(kl_losses)) if kl_losses else 0.0,
    }


@torch.no_grad()
def _eval_gate(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seeds: list[int],
    device: str,
    max_steps: int,
    fail_fast: bool,
) -> list[dict[str, Any]]:
    policy.eval()
    results = eval_seed_batch(
        policy=policy,
        board_size=board_size,
        seeds=seeds,
        device=device,
        max_steps=max_steps,
        use_fill_values=True,
        progress_every=5,
        stop_after_failures=int(fail_fast),
    )
    return results


def _score_results(results: list[dict[str, Any]]) -> tuple[int, float, float]:
    wins = sum(int(result["win"]) for result in results)
    mean_score = float(np.mean([int(result["score"]) for result in results])) if results else 0.0
    win_steps = [int(result["steps"]) for result in results if result["win"] and result.get("steps") is not None]
    mean_win_steps = float(np.mean(win_steps)) if win_steps else float("inf")
    return wins, mean_score, -mean_win_steps


def main() -> int:
    parser = argparse.ArgumentParser(description="Search low-fill early-head RNN patches")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--early-head-max-fill", type=float, default=0.01)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--anchor-seeds", type=_parse_ints, required=True)
    parser.add_argument("--gate-seeds", type=_parse_ints, required=True)
    parser.add_argument("--lrs", type=_parse_floats, required=True)
    parser.add_argument("--epochs", type=_parse_ints, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--trajectory-stride", type=int, default=1)
    parser.add_argument("--trajectory-sample-mode", choices=["stride", "deviation"], default="stride")
    parser.add_argument("--anchor-stride", type=int, default=1)
    parser.add_argument("--trajectory-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-kl-weight", type=float, default=0.0)
    parser.add_argument("--anchor-kl-weight", type=float, default=0.2)
    parser.add_argument("--train-final-layer-only", action="store_true")
    parser.add_argument("--l2-weight", type=float, default=0.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--keep-all-records", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_state = torch.load(args.base, map_location="cpu")
    base_policy = _make_policy(
        board_size=args.board_size,
        hidden_size=args.hidden_size,
        early_head_max_fill=args.early_head_max_fill,
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
        sample_steps = None
        if args.trajectory_sample_mode == "deviation":
            sample_steps = {int(record["deviation_step"])}
        record_weight = args.trajectory_weight * float(record.get("weight", 1.0))
        record_kl_weight = args.trajectory_kl_weight * float(record.get("kl_weight", 1.0))
        collected = _collect_actions(
            policy=base_policy,
            board_size=args.board_size,
            device=args.device,
            seed=int(record["seed"]),
            actions=[int(action) for action in record["actions"]],
            max_steps=args.max_steps,
            early_head_max_fill=args.early_head_max_fill,
            stride=max(1, args.trajectory_stride),
            sample_steps=sample_steps,
            weight=record_weight,
            kl_weight=record_kl_weight,
        )
        _pop_samples(collected, dataset)
        collected["role"] = "trajectory"
        collected["record_weight"] = record_weight
        collected["record_kl_weight"] = record_kl_weight
        seed_records.append(collected)

    for seed in args.anchor_seeds:
        collected = _collect_actions(
            policy=base_policy,
            board_size=args.board_size,
            device=args.device,
            seed=seed,
            actions=None,
            max_steps=args.max_steps,
            early_head_max_fill=args.early_head_max_fill,
            stride=max(1, args.anchor_stride),
            sample_steps=None,
            weight=args.anchor_weight,
            kl_weight=args.anchor_kl_weight,
        )
        _pop_samples(collected, dataset)
        collected["role"] = "anchor"
        seed_records.append(collected)

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
        "trajectory_sample_mode": args.trajectory_sample_mode,
        "anchor_seeds": args.anchor_seeds,
        "gate_seeds": args.gate_seeds,
        "early_head_max_fill": args.early_head_max_fill,
        "seed_records": seed_records,
    }
    print(
        {
            "dataset": {
                key: value
                for key, value in dataset_summary.items()
                if key != "seed_records"
            },
            "seed_record_count": len(seed_records),
        },
        flush=True,
    )

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
                    device=args.device,
                    state=base_state,
                )
                train_stats = _train_early_head(
                    policy=policy,
                    dataset=tensor_dataset,
                    lr=lr,
                    epochs=epoch_count,
                    batch_size=args.batch_size,
                    device=args.device,
                    train_final_layer_only=args.train_final_layer_only,
                    l2_weight=args.l2_weight,
                )
                gate_results = _eval_gate(
                    policy=policy,
                    board_size=args.board_size,
                    seeds=args.gate_seeds,
                    device=args.device,
                    max_steps=args.max_steps,
                    fail_fast=args.fail_fast,
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
