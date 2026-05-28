"""Final-layer-only targeted patch search for recurrent Snake checkpoints.

This keeps inference as the same pure RNN policy, but freezes the encoder, GRU,
and first policy-head layer. Only the final linear classifier is allowed to move.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rnn_cycle_shortcut_patch import (
    _parse_floats,
    _parse_ints,
    _pop_samples,
    _save_atomic,
    _score_results,
)
from rnn_targeted_action_patch import (
    _collect_anchor_seed,
    _collect_anchor_seeds_batch,
    _collect_trajectory_record,
    _eval_gate,
    _load_improved_records,
    _make_policy,
)


def _train_final_layer(
    *,
    policy,
    dataset: dict[str, torch.Tensor],
    lr: float,
    epochs: int,
    batch_size: int,
    device: str,
    l2_weight: float,
) -> dict[str, Any]:
    final_layer = policy.policy_head[-1]
    original_weight = final_layer.weight.detach().clone()
    original_bias = final_layer.bias.detach().clone()

    for param in policy.parameters():
        param.requires_grad = False
    final_layer.weight.requires_grad = True
    final_layer.bias.requires_grad = True

    optimizer = torch.optim.Adam([final_layer.weight, final_layer.bias], lr=lr)
    features = dataset["features"].to(device)
    labels = dataset["labels"].to(device)
    weights = dataset["weights"].to(device)
    base_logits = dataset["base_logits"].to(device)
    base_probs = F.softmax(base_logits, dim=-1)
    kl_weights = dataset["kl_weights"].to(device)
    n_samples = int(labels.shape[0])
    losses: list[float] = []
    ce_losses: list[float] = []
    kl_losses: list[float] = []
    l2_losses: list[float] = []

    policy.train()
    for _ in range(epochs):
        order = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, batch_size):
            index = order[start : start + batch_size]
            logits = policy.policy_head(features[index])
            ce = F.cross_entropy(logits, labels[index], reduction="none")
            weighted_ce = (ce * weights[index]).sum() / weights[index].sum().clamp_min(1e-6)
            kl_per_sample = F.kl_div(
                F.log_softmax(logits, dim=-1),
                base_probs[index],
                reduction="none",
            ).sum(dim=-1)
            weighted_kl = (kl_per_sample * kl_weights[index]).sum() / kl_weights[index].sum().clamp_min(1e-6)
            l2 = (
                F.mse_loss(final_layer.weight, original_weight)
                + F.mse_loss(final_layer.bias, original_bias)
            )
            loss = weighted_ce + weighted_kl + l2_weight * l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            ce_losses.append(float(weighted_ce.item()))
            kl_losses.append(float(weighted_kl.item()))
            l2_losses.append(float(l2.item()))

    return {
        "samples": n_samples,
        "epochs": epochs,
        "lr": lr,
        "l2_weight": l2_weight,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "mean_ce": float(np.mean(ce_losses)) if ce_losses else 0.0,
        "mean_kl": float(np.mean(kl_losses)) if kl_losses else 0.0,
        "mean_l2": float(np.mean(l2_losses)) if l2_losses else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Search final-layer-only RNN policy patches")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--anchor-seeds", type=_parse_ints, required=True)
    parser.add_argument("--gate-seeds", type=_parse_ints, required=True)
    parser.add_argument("--lrs", type=_parse_floats, required=True)
    parser.add_argument("--epochs", type=_parse_ints, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--trajectory-sample-stride", type=int, default=100_000)
    parser.add_argument("--anchor-stride", type=int, default=100)
    parser.add_argument("--target-weight", type=float, default=1000.0)
    parser.add_argument("--trajectory-weight", type=float, default=0.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--target-kl-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-kl-weight", type=float, default=0.0)
    parser.add_argument("--anchor-kl-weight", type=float, default=5.0)
    parser.add_argument("--l2-weight", type=float, default=0.0)
    parser.add_argument("--batch-anchor-collect", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--keep-all-records", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_state = torch.load(args.base, map_location="cpu")
    base_policy = _make_policy(
        board_size=args.board_size,
        hidden_size=args.hidden_size,
        early_head_max_fill=None,
        residual_policy_head=False,
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
            train_residual_head=False,
        )
        _pop_samples(collected, dataset)
        collected["role"] = "trajectory"
        seed_records.append(collected)
        print({"collect": collected}, flush=True)

    if args.batch_anchor_collect:
        anchor_records = _collect_anchor_seeds_batch(
            policy=base_policy,
            board_size=args.board_size,
            seeds=args.anchor_seeds,
            device=args.device,
            max_steps=args.max_steps,
            anchor_stride=max(1, args.anchor_stride),
            anchor_weight=args.anchor_weight,
            anchor_kl_weight=args.anchor_kl_weight,
            early_head_max_fill=None,
            train_residual_head=False,
        )
    else:
        anchor_records = [
            _collect_anchor_seed(
                policy=base_policy,
                board_size=args.board_size,
                seed=seed,
                device=args.device,
                max_steps=args.max_steps,
                anchor_stride=max(1, args.anchor_stride),
                anchor_weight=args.anchor_weight,
                anchor_kl_weight=args.anchor_kl_weight,
                early_head_max_fill=None,
                train_residual_head=False,
            )
            for seed in args.anchor_seeds
        ]

    for collected in anchor_records:
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
                    early_head_max_fill=None,
                    residual_policy_head=False,
                    device=args.device,
                    state=base_state,
                )
                train_stats = _train_final_layer(
                    policy=policy,
                    dataset=tensor_dataset,
                    lr=lr,
                    epochs=epoch_count,
                    batch_size=args.batch_size,
                    device=args.device,
                    l2_weight=args.l2_weight,
                )
                gate_results = _eval_gate(
                    policy=policy,
                    board_size=args.board_size,
                    seeds=args.gate_seeds,
                    device=args.device,
                    max_steps=args.max_steps,
                    fail_fast=args.fail_fast,
                    early_head_max_fill=None,
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
