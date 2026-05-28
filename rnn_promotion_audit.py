"""Promotion audit for pure RNN Snake checkpoints.

The audit is deterministic greedy inference only. It does not use planners,
search, rule fallbacks, or privileged inference-time features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from distill.rnn_model import SnakeRNNPolicy, load_rnn_policy_state
from rnn_eval_seeds import summarize
from rnn_eval_seeds_batch import eval_seed_batch


def _parse_range(value: str) -> tuple[str, list[int]]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("ranges must use start:count or name=start:count")
    name: str | None = None
    spec = value
    if "=" in value:
        name, spec = value.split("=", 1)
    start_s, count_s = spec.split(":", 1)
    start = int(start_s)
    count = int(count_s)
    if count < 1:
        raise argparse.ArgumentTypeError("range count must be >= 1")
    label = name or f"{start}-{start + count - 1}"
    return label, list(range(start, start + count))


def _parse_ranges(value: str) -> list[tuple[str, list[int]]]:
    ranges = [_parse_range(part.strip()) for part in value.split(",") if part.strip()]
    if not ranges:
        raise argparse.ArgumentTypeError("expected at least one range")
    return ranges


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("expected at least one seed")
    return seeds


def _load_policy(args: argparse.Namespace) -> SnakeRNNPolicy:
    policy = SnakeRNNPolicy(
        board_size=args.board_size,
        n_channels=5,
        hidden_size=args.hidden_size,
        early_head_max_fill=args.early_head_max_fill,
        residual_policy_head=args.residual_policy_head,
        residual_policy_min_fill=args.residual_min_fill,
    ).to(args.device)
    load_rnn_policy_state(policy, torch.load(args.checkpoint, map_location="cpu"))
    policy.eval()
    return policy


def _audit_suite(
    *,
    policy: SnakeRNNPolicy,
    board_size: int,
    label: str,
    seeds: list[int],
    device: str,
    max_steps: int,
    early_head_max_fill: float | None,
    residual_min_fill: float | None,
    stop_after_failures: int,
) -> dict[str, Any]:
    print({"suite": label, "episodes": len(seeds), "first_seed": seeds[0], "last_seed": seeds[-1]}, flush=True)
    results = eval_seed_batch(
        policy=policy,
        board_size=board_size,
        seeds=seeds,
        device=device,
        max_steps=max_steps,
        use_fill_values=early_head_max_fill is not None or residual_min_fill is not None,
        progress_every=max(1, min(25, len(seeds))),
        stop_after_failures=stop_after_failures,
    )
    suite_summary = summarize(results, board_size=board_size, checkpoint=label)
    return {"label": label, "seeds": seeds, "results": results, "summary": suite_summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit RNN checkpoint against deterministic promotion gates")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--early-head-max-fill", type=float, default=None)
    parser.add_argument("--residual-policy-head", action="store_true")
    parser.add_argument("--residual-min-fill", type=float, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument(
        "--ranges",
        type=_parse_ranges,
        default=_parse_ranges("20001:100,30001:100,40001:100,50001:100,60001:200"),
        help="Comma-separated start:count or name=start:count suites",
    )
    parser.add_argument("--hard-seeds", type=_parse_seeds, default=None)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--stop-after-failures", type=int, default=1)
    parser.add_argument("--max-mean-win-steps", type=float, default=None)
    parser.add_argument("--max-p95-win-steps", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    policy = _load_policy(args)
    suites: list[dict[str, Any]] = []
    combined_results: list[dict[str, Any]] = []
    for label, seeds in args.ranges:
        suite = _audit_suite(
            policy=policy,
            board_size=args.board_size,
            label=label,
            seeds=seeds,
            device=args.device,
            max_steps=args.max_steps,
            early_head_max_fill=args.early_head_max_fill,
            residual_min_fill=args.residual_min_fill,
            stop_after_failures=args.stop_after_failures,
        )
        suites.append(suite)
        combined_results.extend(suite["results"])
        if args.stop_after_failures and suite["summary"]["failures"]:
            break

    if args.hard_seeds and not any(suite["summary"]["failures"] for suite in suites):
        hard_suite = _audit_suite(
            policy=policy,
            board_size=args.board_size,
            label="hard",
            seeds=args.hard_seeds,
            device=args.device,
            max_steps=args.max_steps,
            early_head_max_fill=args.early_head_max_fill,
            residual_min_fill=args.residual_min_fill,
            stop_after_failures=args.stop_after_failures,
        )
        suites.append(hard_suite)
        combined_results.extend(hard_suite["results"])

    combined_summary = summarize(combined_results, board_size=args.board_size, checkpoint=str(args.checkpoint))
    required_episodes = sum(len(seeds) for _, seeds in args.ranges) + (len(args.hard_seeds) if args.hard_seeds else 0)
    reliability_passed = (
        int(combined_summary["episodes"]) == required_episodes
        and not bool(combined_summary["failures"])
    )
    mean_win_steps = combined_summary.get("mean_win_steps")
    p95_win_steps = combined_summary.get("p95_win_steps")
    path_gate_configured = args.max_mean_win_steps is not None or args.max_p95_win_steps is not None
    mean_steps_passed = (
        None
        if args.max_mean_win_steps is None or mean_win_steps is None
        else float(mean_win_steps) <= float(args.max_mean_win_steps)
    )
    p95_steps_passed = (
        None
        if args.max_p95_win_steps is None or p95_win_steps is None
        else float(p95_win_steps) <= float(args.max_p95_win_steps)
    )
    path_efficiency_passed = None
    if path_gate_configured:
        configured_results = [
            result for result in (mean_steps_passed, p95_steps_passed) if result is not None
        ]
        path_efficiency_passed = bool(configured_results) and all(configured_results)
    checklist = {
        "pure_neural_policy": True,
        "deterministic_greedy_inference": True,
        "no_planner_search_or_rule_fallback": True,
        "perfect_score_required": args.board_size * args.board_size - 3,
        "required_episodes": required_episodes,
        "evaluated_episodes": int(combined_summary["episodes"]),
        "all_evaluated_won": not bool(combined_summary["failures"]),
        "gate_fully_evaluated": int(combined_summary["episodes"]) == required_episodes,
        "reliability_passed": reliability_passed,
        "mean_win_steps": mean_win_steps,
        "p95_win_steps": p95_win_steps,
        "max_mean_win_steps": args.max_mean_win_steps,
        "max_p95_win_steps": args.max_p95_win_steps,
        "path_efficiency_gate_configured": path_gate_configured,
        "mean_steps_passed": mean_steps_passed,
        "p95_steps_passed": p95_steps_passed,
        "path_efficiency_passed": path_efficiency_passed,
        "promotion_passed": reliability_passed and path_efficiency_passed is not False,
    }
    payload = {
        "checkpoint": str(args.checkpoint),
        "args": {
            "board_size": args.board_size,
            "hidden_size": args.hidden_size,
            "early_head_max_fill": args.early_head_max_fill,
            "residual_policy_head": args.residual_policy_head,
            "residual_min_fill": args.residual_min_fill,
            "device": args.device,
            "max_steps": args.max_steps,
            "stop_after_failures": args.stop_after_failures,
        },
        "checklist": checklist,
        "suites": suites,
        "combined_summary": combined_summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print({"out": str(args.out), "checklist": checklist, "summary": combined_summary}, flush=True)
    return 0 if checklist["promotion_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
