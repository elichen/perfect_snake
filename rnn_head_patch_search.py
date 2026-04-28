"""Simultaneous policy-head patch search for recurrent Snake checkpoints.

This is a non-cheating train-time repair tool. Inference remains the same pure
RNN policy over the standard observation stream. The script freezes the encoder
and GRU, collects high-fill hidden states from the base policy, and fits the
policy head on a small set of correction labels plus preservation anchors.
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
from distill.rnn_model import SnakeRNNPolicy, load_rnn_policy_state
from rnn_eval_seeds import eval_seed
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


def _parse_correction_source(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SEED=CHECKPOINT")
    seed_text, path_text = value.split("=", 1)
    try:
        seed = int(seed_text.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid seed in correction source: {seed_text!r}") from exc
    path = Path(path_text.strip())
    if not path:
        raise argparse.ArgumentTypeError("empty checkpoint path in correction source")
    return seed, path


def _parse_seed_int(value: str) -> tuple[int, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SEED=VALUE")
    seed_text, value_text = value.split("=", 1)
    try:
        seed = int(seed_text.strip())
        parsed_value = int(value_text.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid seed/value pair: {value!r}") from exc
    return seed, parsed_value


def _save_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _make_policy(*, board_size: int, hidden_size: int, device: str, state: dict[str, Any]) -> SnakeRNNPolicy:
    policy = SnakeRNNPolicy(board_size=board_size, n_channels=5, hidden_size=hidden_size).to(device)
    load_rnn_policy_state(policy, state)
    return policy


@torch.no_grad()
def _collect_seed_samples(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    seed: int,
    device: str,
    max_steps: int,
    min_score: int,
    min_fill: float,
    anchor_min_score: int,
    anchor_min_fill: float,
    anchor_stride: int,
    is_target_seed: bool,
    correction_weight: float,
    anchor_weight: float,
) -> dict[str, Any]:
    env = SnakeEnv(n=board_size, gamma=0.999, alpha=0.2, seed=seed)
    obs, _ = env.reset(seed=seed)
    hidden = policy.initial_state(1, device)
    features: list[torch.Tensor] = []
    labels: list[int] = []
    weights: list[float] = []
    base_logits: list[torch.Tensor] = []
    corrections = 0
    anchors = 0
    sampled_events: list[dict[str, Any]] = []
    info: dict[str, Any] = {}

    for step in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        encoded = policy.encoder(obs_t)
        hidden = policy.gru_cell(encoded, hidden)
        logits = policy.policy_head(hidden)
        action = int(torch.argmax(logits, dim=-1).item())

        fill = float(env.snake_length) / float(board_size * board_size)
        correction_focus = int(env.score) >= min_score and fill >= min_fill
        anchor_focus = int(env.score) >= anchor_min_score and fill >= anchor_min_fill
        should_anchor = anchor_focus and (anchor_stride <= 1 or step % anchor_stride == 0)
        teacher: int | None = None
        if is_target_seed and correction_focus:
            try:
                cycle, head_idx = find_aligned_cycle(env)
                teacher, _ = expert_action(env, cycle, head_idx)
                teacher = int(teacher)
            except Exception:
                break
        is_correction = teacher is not None and action != teacher
        if is_correction or should_anchor:
            label = int(teacher) if is_correction else action
            weight = correction_weight if is_correction else anchor_weight
            if weight > 0:
                features.append(hidden.squeeze(0).detach().cpu())
                labels.append(label)
                weights.append(float(weight))
                base_logits.append(logits.squeeze(0).detach().cpu())
                corrections += int(is_correction)
                anchors += int(not is_correction)
                if len(sampled_events) < 6:
                    sampled_events.append(
                        {
                            "step": step,
                            "score": int(env.score),
                            "length": int(env.snake_length),
                            "fill": round(fill, 4),
                            "role": "correction" if is_correction else "anchor",
                            "action": action,
                            "teacher": teacher,
                            "label": label,
                            "weight": float(weight),
                        }
                    )

        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "features": features,
        "labels": labels,
        "weights": weights,
        "base_logits": base_logits,
        "corrections": corrections,
        "anchors": anchors,
        "events": sampled_events,
        "score": int(info.get("score", env.score)),
        "reason": info.get("reason"),
        "steps": info.get("steps"),
    }


def _collect_dataset(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    target_seeds: list[int],
    anchor_seeds: list[int],
    device: str,
    max_steps: int,
    min_score: int,
    min_fill: float,
    anchor_min_score: int,
    anchor_min_fill: float,
    anchor_stride: int,
    correction_weight: float,
    anchor_weight: float,
    correction_sources: list[tuple[int, Path]],
    correction_source_min_scores: dict[int, int],
    hidden_size: int,
) -> dict[str, Any]:
    target_set = set(target_seeds)
    all_seeds = list(dict.fromkeys([*target_seeds, *anchor_seeds]))
    seed_records = []
    features: list[torch.Tensor] = []
    labels: list[int] = []
    weights: list[float] = []
    base_logits: list[torch.Tensor] = []
    for seed in all_seeds:
        record = _collect_seed_samples(
            policy=policy,
            board_size=board_size,
            seed=seed,
            device=device,
            max_steps=max_steps,
            min_score=min_score,
            min_fill=min_fill,
            anchor_min_score=anchor_min_score,
            anchor_min_fill=anchor_min_fill,
            anchor_stride=anchor_stride,
            is_target_seed=seed in target_set,
            correction_weight=correction_weight,
            anchor_weight=anchor_weight,
        )
        features.extend(record.pop("features"))
        labels.extend(record.pop("labels"))
        weights.extend(record.pop("weights"))
        base_logits.extend(record.pop("base_logits"))
        print(
            {
                "collect": "base",
                "seed": seed,
                "corrections": record["corrections"],
                "anchors": record["anchors"],
                "score": record["score"],
                "reason": record["reason"],
            },
            flush=True,
        )
        seed_records.append(record)

    for seed, checkpoint_path in correction_sources:
        correction_policy = _make_policy(
            board_size=board_size,
            hidden_size=hidden_size,
            device=device,
            state=torch.load(checkpoint_path, map_location="cpu"),
        )
        correction_policy.eval()
        record = _collect_seed_samples(
            policy=correction_policy,
            board_size=board_size,
            seed=seed,
            device=device,
            max_steps=max_steps,
            min_score=correction_source_min_scores.get(seed, min_score),
            min_fill=min_fill,
            anchor_min_score=anchor_min_score,
            anchor_min_fill=anchor_min_fill,
            anchor_stride=anchor_stride,
            is_target_seed=True,
            correction_weight=correction_weight,
            anchor_weight=0.0,
        )
        features.extend(record.pop("features"))
        labels.extend(record.pop("labels"))
        weights.extend(record.pop("weights"))
        base_logits.extend(record.pop("base_logits"))
        record["correction_source"] = str(checkpoint_path)
        print(
            {
                "collect": "source",
                "seed": seed,
                "corrections": record["corrections"],
                "anchors": record["anchors"],
                "score": record["score"],
                "reason": record["reason"],
                "source": str(checkpoint_path),
            },
            flush=True,
        )
        seed_records.append(record)

    if not features:
        raise RuntimeError("no head-patch samples collected")
    return {
        "features": torch.stack(features, dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
        "weights": torch.tensor(weights, dtype=torch.float32),
        "base_logits": torch.stack(base_logits, dim=0),
        "seed_records": seed_records,
    }


def _train_head(
    *,
    policy: SnakeRNNPolicy,
    dataset: dict[str, Any],
    lr: float,
    epochs: int,
    batch_size: int,
    kl_weight: float,
    device: str,
) -> dict[str, Any]:
    for name, param in policy.named_parameters():
        param.requires_grad = name.startswith("policy_head")
    optimizer = torch.optim.Adam([param for param in policy.parameters() if param.requires_grad], lr=lr)
    features = dataset["features"].to(device)
    labels = dataset["labels"].to(device)
    weights = dataset["weights"].to(device)
    base_logits = dataset["base_logits"].to(device)
    base_probs = F.softmax(base_logits, dim=-1)
    n = int(labels.shape[0])
    losses = []
    ce_losses = []
    kl_losses = []

    policy.train()
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            index = order[start : start + batch_size]
            logits = policy.policy_head(features[index])
            ce = F.cross_entropy(logits, labels[index], reduction="none")
            weighted_ce = (ce * weights[index]).sum() / weights[index].sum().clamp_min(1e-6)
            kl = F.kl_div(
                F.log_softmax(logits, dim=-1),
                base_probs[index],
                reduction="none",
            ).sum(dim=-1).mean()
            loss = weighted_ce + kl_weight * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            ce_losses.append(float(weighted_ce.item()))
            kl_losses.append(float(kl.item()))

    return {
        "samples": n,
        "epochs": epochs,
        "lr": lr,
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
) -> list[dict[str, Any]]:
    policy.eval()
    results = []
    for seed in seeds:
        result = eval_seed(
            policy=policy,
            board_size=board_size,
            seed=seed,
            device=device,
            max_steps=max_steps,
        )
        results.append(result)
        if fail_fast and not result["win"]:
            break
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Simultaneous RNN policy-head patch search")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--target-seeds", type=_parse_ints, required=True)
    parser.add_argument("--anchor-seeds", type=_parse_ints, required=True)
    parser.add_argument("--gate-seeds", type=_parse_ints, required=True)
    parser.add_argument("--lrs", type=_parse_floats, required=True)
    parser.add_argument("--epochs", type=_parse_ints, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--min-score", type=int, default=300)
    parser.add_argument("--min-fill", type=float, default=0.0)
    parser.add_argument("--anchor-min-score", type=int, default=None)
    parser.add_argument("--anchor-min-fill", type=float, default=0.0)
    parser.add_argument("--anchor-stride", type=int, default=1)
    parser.add_argument("--correction-weight", type=float, default=10.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument(
        "--correction-source",
        action="append",
        type=_parse_correction_source,
        default=[],
        help="Additional on-policy correction source as SEED=CHECKPOINT. Useful for patched trajectories that fail new seeds.",
    )
    parser.add_argument(
        "--correction-source-min-score",
        action="append",
        type=_parse_seed_int,
        default=[],
        help="Override correction min score for a source seed as SEED=MIN_SCORE.",
    )
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--save-all", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    anchor_min_score = args.min_score if args.anchor_min_score is None else args.anchor_min_score
    base_state = torch.load(args.base, map_location="cpu")
    correction_source_min_scores = dict(args.correction_source_min_score)
    base_policy = _make_policy(
        board_size=args.board_size,
        hidden_size=args.hidden_size,
        device=args.device,
        state=base_state,
    )
    base_policy.eval()
    dataset = _collect_dataset(
        policy=base_policy,
        board_size=args.board_size,
        target_seeds=args.target_seeds,
        anchor_seeds=args.anchor_seeds,
        device=args.device,
        max_steps=args.max_steps,
        min_score=args.min_score,
        min_fill=args.min_fill,
        anchor_min_score=anchor_min_score,
        anchor_min_fill=args.anchor_min_fill,
        anchor_stride=max(1, args.anchor_stride),
        correction_weight=args.correction_weight,
        anchor_weight=args.anchor_weight,
        correction_sources=args.correction_source,
        correction_source_min_scores=correction_source_min_scores,
        hidden_size=args.hidden_size,
    )
    dataset_summary = {
        "samples": int(dataset["labels"].shape[0]),
        "corrections": int(sum(record["corrections"] for record in dataset["seed_records"])),
        "anchors": int(sum(record["anchors"] for record in dataset["seed_records"])),
        "min_score": args.min_score,
        "min_fill": args.min_fill,
        "anchor_min_score": anchor_min_score,
        "anchor_min_fill": args.anchor_min_fill,
        "anchor_stride": max(1, args.anchor_stride),
        "correction_sources": [(seed, str(path)) for seed, path in args.correction_source],
        "correction_source_min_scores": correction_source_min_scores,
        "seed_records": dataset["seed_records"],
    }
    print({"dataset": dataset_summary}, flush=True)

    started = time.time()
    best_key: tuple[int, float, int] | None = None
    with (args.out_dir / "search.jsonl").open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"dataset": dataset_summary}, sort_keys=True) + "\n")
        log_file.flush()
        for lr in args.lrs:
            for epoch_count in args.epochs:
                candidate_started = time.time()
                policy = _make_policy(
                    board_size=args.board_size,
                    hidden_size=args.hidden_size,
                    device=args.device,
                    state=base_state,
                )
                train_stats = _train_head(
                    policy=policy,
                    dataset=dataset,
                    lr=lr,
                    epochs=epoch_count,
                    batch_size=args.batch_size,
                    kl_weight=args.kl_weight,
                    device=args.device,
                )
                gate_results = _eval_gate(
                    policy=policy,
                    board_size=args.board_size,
                    seeds=args.gate_seeds,
                    device=args.device,
                    max_steps=args.max_steps,
                    fail_fast=args.fail_fast,
                )
                wins = sum(int(result["win"]) for result in gate_results)
                mean_score = float(np.mean([result["score"] for result in gate_results]))
                key = (wins, mean_score, len(gate_results))
                lr_label = f"{lr:.2e}".replace("+", "").replace(".", "p")
                candidate_path = args.out_dir / f"lr{lr_label}_ep{epoch_count}.pt"
                saved_path = None
                if args.save_all or best_key is None or key > best_key or wins == len(args.gate_seeds):
                    _save_atomic(policy.state_dict(), candidate_path)
                    saved_path = str(candidate_path)
                if best_key is None or key > best_key:
                    best_key = key
                record = {
                    "lr": lr,
                    "epochs": epoch_count,
                    "target_seeds": args.target_seeds,
                    "anchor_seeds": args.anchor_seeds,
                    "gate_seeds": args.gate_seeds,
                    "min_score": args.min_score,
                    "anchor_min_score": anchor_min_score,
                    "anchor_min_fill": args.anchor_min_fill,
                    "anchor_stride": max(1, args.anchor_stride),
                    "correction_weight": args.correction_weight,
                    "anchor_weight": args.anchor_weight,
                    "correction_sources": [(seed, str(path)) for seed, path in args.correction_source],
                    "correction_source_min_scores": correction_source_min_scores,
                    "kl_weight": args.kl_weight,
                    "train_stats": train_stats,
                    "gate_results": gate_results,
                    "wins": wins,
                    "gate_count": len(gate_results),
                    "mean_score": mean_score,
                    "saved_path": saved_path,
                    "elapsed_sec": round(time.time() - candidate_started, 1),
                    "total_elapsed_sec": round(time.time() - started, 1),
                }
                print(record, flush=True)
                log_file.write(json.dumps(record, sort_keys=True) + "\n")
                log_file.flush()
                if wins == len(args.gate_seeds):
                    print({"gate_passed": True, "checkpoint": saved_path}, flush=True)
                    return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
