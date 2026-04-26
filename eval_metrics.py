"""Shared evaluation metrics for Snake checkpoints and train-time eval."""

from __future__ import annotations

from typing import Dict, Iterable


PHASE_BUCKETS = (
    ("phase_lt20", 0.0, 0.20),
    ("phase_20_80", 0.20, 0.80),
    ("phase_80_95", 0.80, 0.95),
    ("phase_gte95", 0.95, 1.01),
)

DEATH_REASONS = ("self", "wall", "stall")


def classify_phase_bucket(snake_len: int, board_area: int) -> str:
    fill_ratio = snake_len / float(max(1, board_area))
    for name, lo, hi in PHASE_BUCKETS:
        if lo <= fill_ratio < hi:
            return name
    return PHASE_BUCKETS[-1][0]


def summarize_phase_metrics(
    *,
    scores: Iterable[int],
    terminal_lengths: Iterable[int],
    reasons: Iterable[str],
    perfect_score: int,
    episodes: int,
) -> Dict[str, float | int]:
    score_list = list(scores)
    length_list = list(terminal_lengths)
    reason_list = list(reasons)
    if not (len(score_list) == len(length_list) == len(reason_list)):
        raise ValueError("scores, terminal_lengths, and reasons must have the same length")

    board_area = perfect_score + 3
    counts: Dict[str, int] = {f"{name}_count": 0 for name, _, _ in PHASE_BUCKETS}
    counts["win_count"] = 0
    for reason in DEATH_REASONS:
        counts[f"death_{reason}_count"] = 0
    counts["death_other_count"] = 0

    for score, snake_len, reason in zip(score_list, length_list, reason_list):
        if score >= perfect_score:
            counts["win_count"] += 1
            continue

        phase_bucket = classify_phase_bucket(snake_len, board_area)
        counts[f"{phase_bucket}_count"] += 1

        reason_key = f"death_{reason}_count"
        if reason_key in counts:
            counts[reason_key] += 1
        else:
            counts["death_other_count"] += 1

    result: Dict[str, float | int] = {}
    denom = float(max(1, episodes))
    for key, count in counts.items():
        result[key] = int(count)
        rate_key = f"{key[:-6]}_rate" if key.endswith("_count") else f"{key}_rate"
        result[rate_key] = float(count / denom)

    return result
