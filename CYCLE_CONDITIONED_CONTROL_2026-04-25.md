# Cycle-Conditioned 20x20 Snake Control Branch

Date: 2026-04-25

## Mission Context

The original mission remains a pure neural-network policy that reaches a deterministic 100% win rate on 20x20 Snake, where a win is score `397/397`, without privileged inference-time feature engineering. The PPO frontier did not satisfy this: the prior best PPO checkpoint had `0/1000` deterministic wins under deep eval.

The branch described here changes the strategy from PPO endgame repair to a neural Hamiltonian-cycle follower with explicit cycle-conditioning inputs. This is valuable as a control and a feasibility result, but it does **not** satisfy the original mission under the stricter no-privileged-features criterion.

## Current Best Checkpoint

Checkpoint:

`experiments/ham_cycle_lowfill_fix_s44_20260425T_resume/model.best_eval.pt`

This checkpoint is a pure neural policy at inference. It does not use search, planning, or a fallback controller at inference. However, it does use fixed cycle-conditioning observation channels produced by the evaluator. Under the project mission, this should be treated as privileged inference-time feature engineering, not as a solved original-observation policy.

## Code Added

`cycle_bc_exhaustive.py` trains a cycle-conditioned policy on synthetic Hamiltonian-cycle states.

`distill_eval_batched.py` evaluates cycle-conditioned distillation checkpoints with many Snake environments batched through one model forward pass.

`cycle_bc_scan.py` random-scans synthetic on-cycle states for policy/expert disagreements.

`cycle_bc_exact_length_scan.py` exhaustively scans a fixed snake length across all cycles, head indices, and food positions.

## Training Path

The first exhaustive cycle-BC model was trained in:

`experiments/ham_cycle_exhaustive_s44_20260423T2055`

It reached `100%` synthetic held-out action accuracy at steps `1000-1400`, and produced real wins, but its first 5-seed eval was only `4/5`. The failure was seed `10002`, score `0`, wall death at step `239`.

Diagnostic trace showed the first disagreement at score `0`, length `3`, head `(19, 19)`, direction right. The expert action was left; the model chose straight into the wall.

The current checkpoint was produced by low-fill fine-tuning from that model:

`experiments/ham_cycle_lowfill_fix_s44_20260425T_resume`

Fine-tune settings: resume from `ham_cycle_exhaustive_s44_20260423T2055/model.best_eval.pt`, train only fill range `0.0-0.03`, LR `1e-5`, stopped manually at step `500` to avoid drift.

## Evidence So Far

Real deterministic episode gates:

- `5/5` wins, all score `397/397`, seed start `10001`.
- `20/20` wins, all score `397/397`, seed start `20001`.
- `100/100` wins, all score `397/397`, seed start `30001`.
- `1000/1000` wins, all score `397/397`, seed start `40001`.

Synthetic disagreement scans:

- `500,000/500,000` random all-fill on-cycle states matched the Hamiltonian expert.
- `500,000/500,000` random low-fill states matched the Hamiltonian expert.
- `2,540,800/2,540,800` exact length-3 states matched the Hamiltonian expert across every cycle, every head index, and every legal food position.

## Interpretation

This is the first branch with sustained nonzero and then perfect deterministic win-rate evidence on 20x20. It strongly suggests the network has enough capacity to implement a perfect 20x20 Snake policy when given explicit cycle-conditioning channels.

It should not be reported as solving the original mission. The correct interpretation is: the remaining barrier is not neural capacity alone; it is learning or representing the global cycle/space-filling strategy from non-privileged observations.

## Next Gates

The `1000`-episode batched deterministic eval passed in `4864` seconds. Under a cycle-conditioned pure-NN criterion, this branch is solved. Under the original no-privileged-features mission criterion, it is a control branch only.

The next research branch should distill the cycle-conditioned behavior into a student that sees only the original non-privileged observation contract, or use the cycle-conditioned policy as a train-time teacher without exposing cycle identity or cycle target channels at inference.
