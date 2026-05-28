"""Constrained final-layer margin patch search for recurrent Snake policies.

This keeps inference as the same pure RNN policy. It freezes every parameter
except the final policy classifier and optimizes hinge constraints: force
harvested corrections to win by a margin while preserving anchor argmaxes.
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
    _save_atomic,
    _score_results,
)
from rnn_targeted_action_patch import (
    _collect_anchor_seeds_batch,
    _collect_trajectory_record,
    _eval_gate,
    _load_improved_records,
    _make_policy,
)


def _parse_optional_ints(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    return _parse_ints(value)


def _append_samples(
    collected: dict[str, Any],
    dataset: dict[str, list[Any]],
    *,
    role: str,
) -> int:
    appended = 0
    for idx, weight in enumerate(collected["weights"]):
        if float(weight) <= 0.0:
            continue
        dataset["features"].append(collected["features"][idx])
        dataset["labels"].append(int(collected["labels"][idx]))
        dataset["base_logits"].append(collected["base_logits"][idx])
        dataset["roles"].append(role)
        appended += 1
    return appended


def _collection_summary(collected: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in collected.items()
        if key not in {"features", "labels", "weights", "base_logits", "kl_weights"}
    }


def _label_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    chosen = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    mask = F.one_hot(labels, num_classes=logits.shape[-1]).bool()
    other = logits.masked_fill(mask, -torch.inf).max(dim=1).values
    return chosen - other


def _as_tensor_dataset(dataset: dict[str, list[Any]]) -> dict[str, torch.Tensor]:
    if not dataset["labels"]:
        raise ValueError("empty dataset")
    return {
        "features": torch.stack(dataset["features"], dim=0),
        "labels": torch.tensor(dataset["labels"], dtype=torch.long),
        "base_logits": torch.stack(dataset["base_logits"], dim=0),
    }


def _dataset_summary(
    *,
    targets: dict[str, torch.Tensor],
    anchors: dict[str, torch.Tensor],
    trajectory_json: Path,
    anchor_seeds: list[int],
    dense_anchor_seeds: list[int],
    gate_seeds: list[int],
    seed_records: list[dict[str, Any]],
) -> dict[str, Any]:
    role_counts: dict[str, int] = {}
    role_samples: dict[str, int] = {}
    failures = []
    target_records = []
    dense_records = []
    for record in seed_records:
        role = str(record.get("role"))
        role_counts[role] = role_counts.get(role, 0) + 1
        role_samples[role] = role_samples.get(role, 0) + int(record.get("appended_samples", 0))
        if int(record.get("score", 0)) < 397:
            failures.append(record)
        if role == "target":
            target_records.append(record)
        if role == "dense_anchor":
            dense_records.append(record)
    return {
        "target_samples": int(targets["labels"].shape[0]),
        "anchor_samples": int(anchors["labels"].shape[0]),
        "trajectory_json": str(trajectory_json),
        "anchor_seed_count": len(anchor_seeds),
        "dense_anchor_seeds": dense_anchor_seeds,
        "gate_seeds": gate_seeds,
        "role_counts": role_counts,
        "role_samples": role_samples,
        "target_records": target_records,
        "dense_anchor_records": dense_records,
        "collection_failures": failures,
    }


def _constraint_stats(
    *,
    policy,
    targets: dict[str, torch.Tensor],
    anchors: dict[str, torch.Tensor],
    target_margin: float,
    anchor_margin: float,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    policy.eval()
    with torch.no_grad():
        target_features = targets["features"].to(device)
        target_labels = targets["labels"].to(device)
        target_logits = policy.policy_head(target_features)
        target_margins = _label_margin(target_logits, target_labels).detach().cpu()

        anchor_flips = 0
        anchor_violations = 0
        anchor_margins: list[torch.Tensor] = []
        for start in range(0, int(anchors["labels"].shape[0]), batch_size):
            stop = min(start + batch_size, int(anchors["labels"].shape[0]))
            features = anchors["features"][start:stop].to(device)
            labels = anchors["labels"][start:stop].to(device)
            base_logits = anchors["base_logits"][start:stop].to(device)
            logits = policy.policy_head(features)
            margins = _label_margin(logits, labels)
            base_margins = _label_margin(base_logits, labels).clamp_min(0.0)
            required = torch.minimum(
                torch.full_like(base_margins, float(anchor_margin)),
                base_margins,
            )
            anchor_flips += int((torch.argmax(logits, dim=-1) != labels).sum().item())
            anchor_violations += int((margins < required).sum().item())
            anchor_margins.append(margins.detach().cpu())

        anchor_margin_t = torch.cat(anchor_margins, dim=0)
    return {
        "target_min_margin": float(target_margins.min().item()),
        "target_mean_margin": float(target_margins.mean().item()),
        "target_violations": int((target_margins < target_margin).sum().item()),
        "target_count": int(target_margins.numel()),
        "anchor_min_margin": float(anchor_margin_t.min().item()),
        "anchor_p01_margin": float(torch.quantile(anchor_margin_t, 0.01).item()),
        "anchor_mean_margin": float(anchor_margin_t.mean().item()),
        "anchor_flips": anchor_flips,
        "anchor_violations": anchor_violations,
        "anchor_count": int(anchor_margin_t.numel()),
    }


def _train_margin_patch(
    *,
    policy,
    targets: dict[str, torch.Tensor],
    anchors: dict[str, torch.Tensor],
    lr: float,
    steps: int,
    batch_size: int,
    target_margin: float,
    anchor_margin: float,
    target_weight: float,
    anchor_weight: float,
    anchor_kl_weight: float,
    l2_weight: float,
    device: str,
) -> dict[str, Any]:
    final_layer = policy.policy_head[-1]
    original_weight = final_layer.weight.detach().clone()
    original_bias = final_layer.bias.detach().clone()

    for param in policy.parameters():
        param.requires_grad = False
    final_layer.weight.requires_grad = True
    final_layer.bias.requires_grad = True

    target_features = targets["features"].to(device)
    target_labels = targets["labels"].to(device)
    anchor_features = anchors["features"]
    anchor_labels = anchors["labels"]
    anchor_base_logits = anchors["base_logits"]
    n_anchors = int(anchor_labels.shape[0])

    optimizer = torch.optim.Adam([final_layer.weight, final_layer.bias], lr=lr)
    losses: list[float] = []
    target_losses: list[float] = []
    anchor_losses: list[float] = []
    kl_losses: list[float] = []
    l2_losses: list[float] = []

    policy.train()
    for _ in range(steps):
        anchor_count = min(batch_size, n_anchors)
        anchor_index = torch.randint(n_anchors, (anchor_count,))
        batch_anchor_features = anchor_features.index_select(0, anchor_index).to(device)
        batch_anchor_labels = anchor_labels.index_select(0, anchor_index).to(device)
        batch_anchor_base_logits = anchor_base_logits.index_select(0, anchor_index).to(device)

        target_logits = policy.policy_head(target_features)
        target_margins = _label_margin(target_logits, target_labels)
        target_loss = F.relu(float(target_margin) - target_margins).mean()

        anchor_logits = policy.policy_head(batch_anchor_features)
        anchor_margins = _label_margin(anchor_logits, batch_anchor_labels)
        anchor_base_margins = _label_margin(
            batch_anchor_base_logits,
            batch_anchor_labels,
        ).clamp_min(0.0)
        required_anchor_margins = torch.minimum(
            torch.full_like(anchor_base_margins, float(anchor_margin)),
            anchor_base_margins,
        )
        anchor_loss = F.relu(required_anchor_margins - anchor_margins).mean()
        anchor_kl = F.kl_div(
            F.log_softmax(anchor_logits, dim=-1),
            F.softmax(batch_anchor_base_logits, dim=-1),
            reduction="batchmean",
        )
        l2 = (
            F.mse_loss(final_layer.weight, original_weight)
            + F.mse_loss(final_layer.bias, original_bias)
        )
        loss = (
            float(target_weight) * target_loss
            + float(anchor_weight) * anchor_loss
            + float(anchor_kl_weight) * anchor_kl
            + float(l2_weight) * l2
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        target_losses.append(float(target_loss.item()))
        anchor_losses.append(float(anchor_loss.item()))
        kl_losses.append(float(anchor_kl.item()))
        l2_losses.append(float(l2.item()))

    delta_weight = final_layer.weight.detach() - original_weight
    delta_bias = final_layer.bias.detach() - original_bias
    return {
        "lr": lr,
        "steps": steps,
        "target_margin": target_margin,
        "anchor_margin": anchor_margin,
        "target_weight": target_weight,
        "anchor_weight": anchor_weight,
        "anchor_kl_weight": anchor_kl_weight,
        "l2_weight": l2_weight,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "mean_target_loss": float(np.mean(target_losses)) if target_losses else 0.0,
        "mean_anchor_loss": float(np.mean(anchor_losses)) if anchor_losses else 0.0,
        "mean_anchor_kl": float(np.mean(kl_losses)) if kl_losses else 0.0,
        "mean_l2": float(np.mean(l2_losses)) if l2_losses else 0.0,
        "delta_weight_norm": float(torch.linalg.vector_norm(delta_weight).item()),
        "delta_bias_norm": float(torch.linalg.vector_norm(delta_bias).item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Search constrained final-layer margin patches")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--anchor-seeds", type=_parse_ints, required=True)
    parser.add_argument("--dense-anchor-seeds", type=_parse_optional_ints, default=[])
    parser.add_argument("--gate-seeds", type=_parse_ints, required=True)
    parser.add_argument("--lrs", type=_parse_floats, required=True)
    parser.add_argument("--steps", type=_parse_ints, required=True)
    parser.add_argument("--target-margins", type=_parse_floats, required=True)
    parser.add_argument("--anchor-margins", type=_parse_floats, required=True)
    parser.add_argument("--l2-weights", type=_parse_floats, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--trajectory-sample-stride", type=int, default=100_000)
    parser.add_argument("--anchor-stride", type=int, default=100)
    parser.add_argument("--dense-anchor-stride", type=int, default=1)
    parser.add_argument("--target-weight", type=float, default=100.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--anchor-kl-weight", type=float, default=1.0)
    parser.add_argument("--cache-path", type=Path, default=None)
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

    if args.cache_path is not None and args.cache_path.exists():
        cached = torch.load(args.cache_path, map_location="cpu")
        targets = cached["targets"]
        anchors = cached["anchors"]
        dataset_summary = cached["dataset_summary"]
        print({"dataset_cache": "loaded", "path": str(args.cache_path)}, flush=True)
    else:
        target_samples: dict[str, list[Any]] = {
            "features": [],
            "labels": [],
            "base_logits": [],
            "roles": [],
        }
        anchor_samples: dict[str, list[Any]] = {
            "features": [],
            "labels": [],
            "base_logits": [],
            "roles": [],
        }
        seed_records = []

        for record in _load_improved_records(args.trajectory_json, keep_all=args.keep_all_records):
            collected = _collect_trajectory_record(
                policy=base_policy,
                board_size=args.board_size,
                device=args.device,
                record=record,
                sample_stride=max(1, args.trajectory_sample_stride),
                target_weight=1.0,
                trajectory_weight=0.0,
                target_kl_weight=0.0,
                trajectory_kl_weight=0.0,
                train_residual_head=False,
            )
            appended = _append_samples(collected, target_samples, role="target")
            collected["role"] = "target"
            collected["appended_samples"] = appended
            summary = _collection_summary(collected)
            seed_records.append(summary)
            print({"collect": summary}, flush=True)

        broad_anchor_records = _collect_anchor_seeds_batch(
            policy=base_policy,
            board_size=args.board_size,
            seeds=args.anchor_seeds,
            device=args.device,
            max_steps=args.max_steps,
            anchor_stride=max(1, args.anchor_stride),
            anchor_weight=1.0,
            anchor_kl_weight=1.0,
            early_head_max_fill=None,
            train_residual_head=False,
        )
        for collected in broad_anchor_records:
            appended = _append_samples(collected, anchor_samples, role="anchor")
            collected["role"] = "anchor"
            collected["appended_samples"] = appended
            summary = _collection_summary(collected)
            seed_records.append(summary)
            print({"collect": summary}, flush=True)

        if args.dense_anchor_seeds:
            dense_anchor_records = _collect_anchor_seeds_batch(
                policy=base_policy,
                board_size=args.board_size,
                seeds=args.dense_anchor_seeds,
                device=args.device,
                max_steps=args.max_steps,
                anchor_stride=max(1, args.dense_anchor_stride),
                anchor_weight=1.0,
                anchor_kl_weight=1.0,
                early_head_max_fill=None,
                train_residual_head=False,
            )
            for collected in dense_anchor_records:
                appended = _append_samples(collected, anchor_samples, role="dense_anchor")
                collected["role"] = "dense_anchor"
                collected["appended_samples"] = appended
                summary = _collection_summary(collected)
                seed_records.append(summary)
                print({"collect": summary}, flush=True)

        targets = _as_tensor_dataset(target_samples)
        anchors = _as_tensor_dataset(anchor_samples)
        dataset_summary = _dataset_summary(
            targets=targets,
            anchors=anchors,
            trajectory_json=args.trajectory_json,
            anchor_seeds=args.anchor_seeds,
            dense_anchor_seeds=args.dense_anchor_seeds,
            gate_seeds=args.gate_seeds,
            seed_records=seed_records,
        )
        if args.cache_path is not None:
            _save_atomic(
                {
                    "targets": targets,
                    "anchors": anchors,
                    "dataset_summary": dataset_summary,
                },
                args.cache_path,
            )
            print({"dataset_cache": "saved", "path": str(args.cache_path)}, flush=True)
    print({"dataset": dataset_summary}, flush=True)

    best_key: tuple[int, float, float] | None = None
    best_record: dict[str, Any] | None = None
    started = time.time()
    with (args.out_dir / "search.jsonl").open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"dataset": dataset_summary}, sort_keys=True) + "\n")
        log_file.flush()
        for lr in args.lrs:
            for step_count in args.steps:
                for target_margin in args.target_margins:
                    for anchor_margin in args.anchor_margins:
                        for l2_weight in args.l2_weights:
                            candidate_started = time.time()
                            policy = _make_policy(
                                board_size=args.board_size,
                                hidden_size=args.hidden_size,
                                early_head_max_fill=None,
                                residual_policy_head=False,
                                device=args.device,
                                state=base_state,
                            )
                            train_stats = _train_margin_patch(
                                policy=policy,
                                targets=targets,
                                anchors=anchors,
                                lr=lr,
                                steps=step_count,
                                batch_size=args.batch_size,
                                target_margin=target_margin,
                                anchor_margin=anchor_margin,
                                target_weight=args.target_weight,
                                anchor_weight=args.anchor_weight,
                                anchor_kl_weight=args.anchor_kl_weight,
                                l2_weight=l2_weight,
                                device=args.device,
                            )
                            constraint_stats = _constraint_stats(
                                policy=policy,
                                targets=targets,
                                anchors=anchors,
                                target_margin=target_margin,
                                anchor_margin=anchor_margin,
                                device=args.device,
                                batch_size=args.batch_size,
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
                            lr_label = f"{lr:.1e}".replace("+", "").replace(".", "p")
                            tm_label = f"{target_margin:.2f}".replace(".", "p")
                            am_label = f"{anchor_margin:.2f}".replace(".", "p")
                            l2_label = f"{l2_weight:.1e}".replace("+", "").replace(".", "p")
                            candidate_path = (
                                args.out_dir
                                / f"lr{lr_label}_st{step_count}_tm{tm_label}_am{am_label}_l2{l2_label}.pt"
                            )
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
                                    "constraint_stats": constraint_stats,
                                    "train_stats": train_stats,
                                }
                            win_steps = [
                                int(result["steps"])
                                for result in gate_results
                                if result["win"] and result.get("steps") is not None
                            ]
                            record = {
                                "lr": lr,
                                "steps": step_count,
                                "target_margin": target_margin,
                                "anchor_margin": anchor_margin,
                                "l2_weight": l2_weight,
                                "train_stats": train_stats,
                                "constraint_stats": constraint_stats,
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
