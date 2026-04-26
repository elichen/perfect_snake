# 20x20 Snake Research Handoff

## Mission

Train a **pure neural network** policy that achieves **100% deterministic win rate** on **20x20 Snake**.

Success criterion:
- saved checkpoint
- greedy / deterministic inference
- repeated `397 / 397` wins
- no planner, search, fallback controller, or expert at inference time

This handoff is meant to replace stale high-level docs. The older plateau notes in [AGENTS.md](/Users/elichen/code/perfect_snake/AGENTS.md) are no longer the current state of the project.

## Current Status

The best known saved checkpoint is:
- [ppo_research_145_micro_s183](/Users/elichen/code/perfect_snake/experiments/ppo_research_145_micro_s183_177559752735/summary.json)

Its best **20-episode** deterministic eval slice is:
- `392.35 / 397`
- `median = 393.0`
- `win_rate = 0.0`
- `phase_lt20_rate = 0.0`
- `phase_gte95_rate = 1.0`

This number is useful as a historical waypoint, but it is **not** a trustworthy benchmark by itself.

The harder benchmark we actually trust is the deeper deterministic probe recorded in [mission_loop_state.json](/Users/elichen/code/perfect_snake/experiments/mission_loop_state.json):
- `0 / 200` wins
- `mean_score = 363.87`

Interpretation:
- the policy is still strong
- the policy is no longer failing early
- but the `392.35` result is a lucky 20-episode slice, not a stable near-mastery frontier
- any loop logic that promotes candidates on 20 deterministic episodes alone is selecting substantial noise

Operational implication:
- retire the 20-episode gate as the primary research benchmark
- future candidate promotion should use a deeper deterministic benchmark

## Best Chain So Far

The strongest known path was:
- [ppo_research_143_resume_s183](/Users/elichen/code/perfect_snake/experiments/ppo_research_143_resume_s183_177559574261/summary.json): `390.10`
- [ppo_research_144_micro_s183](/Users/elichen/code/perfect_snake/experiments/ppo_research_144_micro_s183_177559662451/summary.json): `391.45`
- [ppo_research_145_micro_s183](/Users/elichen/code/perfect_snake/experiments/ppo_research_145_micro_s183_177559752735/summary.json): `392.35`

What this means:
- a strong near-frontier basin can sometimes be improved by a very short, very conservative continuation chain
- but the improvement window is narrow
- additional continuation usually overshoots or regresses

Earlier major step:
- [ppo_research_018_resume_s55](/Users/elichen/code/perfect_snake/experiments/ppo_research_018_resume_s55_177549857655/summary.json): `383.95`

Repeatable fallback family:
- [ppo_research_189_resume_s183](/Users/elichen/code/perfect_snake/experiments/ppo_research_189_resume_s183_177566727057/summary.json): `390.55`

Interpretation:
- the `189` family is real and repeatable enough to matter
- but it has not surpassed the `143 -> 144 -> 145` chain

## Failure Manifold of the Frontier

The frontier checkpoint was explicitly harvested for near-perfect failures:
- [ppo_research_156_failharv_s183](/Users/elichen/code/perfect_snake/experiments/ppo_research_156_failharv_s183_1775602512697274/harvest_summary.json)

Key facts from the harvest:
- `72` harvested failures
- dominated by `self` collisions
- failures are concentrated at scores `393` and `394`
- whenever the safe-action label was available on the terminal decision, the policy chose the wrong action

The most important signal is **when** the policy diverges from a shallow survival heuristic.

Re-analysis of [failures.jsonl](/Users/elichen/code/perfect_snake/experiments/ppo_research_156_failharv_s183_1775602512697274/failures.jsonl):

| Steps Before Death | Policy Matches 1-Ply Safe | Misses | Unknown | Avg Miss Margin |
|---|---:|---:|---:|---:|
| 0 | 0 | 9 | 63 | 0.00 |
| 1 | 60 | 12 | 0 | 2.17 |
| 2 | 67 | 5 | 0 | 1.75 |
| 3 | 26 | 46 | 0 | 1.37 |
| 4 | 70 | 2 | 0 | 1.75 |
| 5 | 58 | 14 | 0 | 1.11 |

Additional structure:
- `57 / 72` failures terminate on edges or corners
- `45 / 72` fail exactly at score `393`
- the terminal move is usually already forced (`safe=na` in `63 / 72`)

This is the cleanest current signal in the project:
- the trap is typically set **3 moves before death**
- not on the terminal move
- and not as a broad policy collapse

Top-level interpretation:
- this is not a broad policy weakness anymore
- this is a late-game action-selection / calibration problem
- the remaining gap appears to be the last `1-5` moves
- the sharpest training signal is specifically the `3`-ply pre-death disagreement regime

## What Has Worked

The only line that is currently competitive is PPO.

Useful ingredients in the strong PPO family:
- head-centered observation
- flood-fill observation
- auxiliary flood-fill decoder
- symmetric augmentation
- full resume-state checkpointing
- extremely conservative resume / micro-resume continuation

Practical lesson:
- fresh controls can still open strong basins
- but most branch value has come from preserving and cautiously extending a good basin, not from architecture churn

## What Has Not Worked

### Distillation

Distillation is not currently competitive.

The best distillation / recurrent distillation line plateaued far below PPO and never entered the near-frontier regime. It should not be the default research branch.

### Broad Aux Churn

Most auxiliary ideas did not beat the PPO frontier:
- cycle-target variants
- tail-target variants
- safe-action aux variants
- elite BC replay
- body-age / temporal occupancy aux

Only flood-fill aux clearly earned its place.

### Temporal Context Probes

These were tested and are currently negative:
- `obs_history=2`
- `action_history_obs=2`

They were bad enough to retire from automatic routing.

### Historical Endgame Basin Repros

These were real enough to run, but not frontier-winning:
- [ppo_research_170_endbody_s220](/Users/elichen/code/perfect_snake/experiments/ppo_research_170_endbody_s220_177561093202/summary.json): `383.80`
- [ppo_research_172_endmix_s222](/Users/elichen/code/perfect_snake/experiments/ppo_research_172_endmix_s222_177561194712/summary.json): `381.65`

Later clean repros of those families also underperformed the frontier.

### Dense Harvest / Heuristic Endgame Retries

Repeatedly weak or non-frontier:
- dense checkpoint harvest
- endgame safe-action heuristic retries
- body-age-as-observation
- additional same-source micro/nano retries once a source had already failed

## Why We Still Think 100% Is Plausible

The current best model already:
- avoids early-game collapse entirely
- reaches `95%+` fill in every eval episode
- gets to `392.35 / 397` mean deterministically

That does **not** look like a representation incapable of solving the game.

It looks like:
- a late-game stability problem
- or an ultra-local objective / continuation problem
- with very high sensitivity to tiny PPO updates

So the correct conclusion is:
- pure NN mastery is still plausible
- the current feedforward PPO path is unstable near mastery
- sampled train wins do not imply deterministic eval wins

## Loop / Automation State

The autonomous controller lives in:
- [mission_loop.py](/Users/elichen/code/perfect_snake/mission_loop.py)
- [LOOP_POLICY.md](/Users/elichen/code/perfect_snake/LOOP_POLICY.md)

State and decisions live in:
- [mission_loop_state.json](/Users/elichen/code/perfect_snake/experiments/mission_loop_state.json)
- [mission_loop.log](/Users/elichen/code/perfect_snake/experiments/mission_loop.log)

At the time of this handoff:
- the controller is stopped
- [mission_loop.stop](/Users/elichen/code/perfect_snake/experiments/mission_loop.stop) exists
- `active_run` in [mission_loop_state.json](/Users/elichen/code/perfect_snake/experiments/mission_loop_state.json) is empty

Last loop behavior before stop:
- mostly routine control sweeps
- many controls in the `280-370` range
- no new promotable research branch
- no nonzero deterministic eval win

The state file records important negative evidence, including:
- suppressed fallback families
- harvest results
- win probe results
- prior micro / nano / parent-seed attempts

## Current PI Read

The project is no longer bottlenecked by:
- basic competence
- early-game shaping
- large architecture changes

It is bottlenecked by:
- preserving a near-perfect basin under continuation
- or learning the final few endgame decisions without destabilizing the rest of the policy

The clearest scientific facts are:
- the apparent frontier fails late, not early
- `0 / 200` deterministic wins means the current best basin is not already an eval winner
- many targeted heuristic fixes underperform the clean PPO frontier
- continuation quality depends heavily on exact source basin and step size
- the strongest actionable structure in the data is a sharp `3`-step pre-death mismatch with a shallow survival heuristic

## Recommended Next Steps For A New Researcher

If I were handing this off cold, I would ask the next researcher to do these in order.

### 1. Replace the coarse eval gate with a real benchmark

Start from:
- [ppo_research_145_micro_s183](/Users/elichen/code/perfect_snake/experiments/ppo_research_145_micro_s183_177559752735)
- [ppo_research_156_failharv_s183](/Users/elichen/code/perfect_snake/experiments/ppo_research_156_failharv_s183_1775602512697274)

Verify:
- the optimistic `392.35` 20-episode slice
- the much more meaningful `0 / 200` win-probe result
- the `393-394` self-collision concentration
- the `3`-step mismatch table above

Concrete recommendation:
- write a `deep_eval.py`
- run `1000` deterministic episodes against a list of candidate checkpoints
- emit `win_rate`, `mean`, `std`, and score CDFs
- benchmark at least:
  - `143`
  - `144`
  - `145`
  - strongest recent controls
  - any checkpoint a new branch wants to promote

If these cannot be reproduced, stop and debug evaluation invariants first.

### 2. Treat this as a local endgame research problem

Do not start with:
- generic seed farming
- broad new aux heads
- distillation
- generic larger networks

Do start with:
- hypotheses specifically about the last `1-5` moves
- interventions that preserve the frontier basin as much as possible

### 3. Planner-conditioned endgame correction is the most defensible next branch

The repo already has a shallow planner probe in [planner_probe.py](/Users/elichen/code/perfect_snake/planner_probe.py).

The next branch should use it as a **training-only teacher**, not an inference-time crutch:
- roll out the incumbent for many episodes
- harvest states at `fill >= 0.85` where a depth-3 planner disagrees with the policy
- train with behavior-cloning style corrections on those states
- keep a KL anchor to the incumbent policy so the basin does not drift
- aggregate more corrected endgame states across rounds

Why this is the best next branch:
- it directly targets the observed `3`-ply failure signature
- it preserves the “pure NN at inference” constraint
- it gives the model a dense endgame correction signal that sparse PPO returns do not provide

Clear falsification:
- if the corrected policy still has `0` real wins under deep eval, this branch is probably dead too

### 4. Prefer small, answerable experiments

The right experiment style now is:
- fixed short continuation budget
- dense evals
- hard kill criteria
- compare against the `145` frontier directly

This problem does not currently reward long blind runs.

### 5. Candidate next research questions

The most defensible unresolved questions are:

1. Is the remaining failure due to a tiny observation alias in the final corridor regime that is not solved by shallow history?
2. Is PPO locally too unstable near mastery, such that a different continuation objective is needed?
3. Can we train specifically on harvested `390-396` failures without destroying the rest of the basin?

### 6. Concrete high-value directions

If continuing research from here, I would prioritize:

- deep deterministic evaluation as the new primary benchmark
- failure-conditioned endgame DAgger from a depth-3 planner at high fill
- a stricter local continuation objective than the current PPO micro/nano setup
- targeted analysis of whether the final mistakes are actually action-logit calibration errors versus missing state

One specific continuation change is especially justified:
- add an explicit KL anchor to the incumbent policy during micro / short continuation
- the current “micro” logic mostly reduces step size, but does not explicitly constrain basin drift
- that directly targets the most documented failure mode in the current PPO line

I would de-prioritize:
- more plain controls
- more historical basin repros that already lost to the frontier
- more lightweight history toggles
- more safe-action heuristic branches without a better local objective

## Minimal Artifact Set To Read First

If another researcher only reads a few files, read these:

1. [ppo_research_145_micro_s183/summary.json](/Users/elichen/code/perfect_snake/experiments/ppo_research_145_micro_s183_177559752735/summary.json)
2. [ppo_research_156_failharv_s183/harvest_summary.json](/Users/elichen/code/perfect_snake/experiments/ppo_research_156_failharv_s183_1775602512697274/harvest_summary.json)
3. [mission_loop_state.json](/Users/elichen/code/perfect_snake/experiments/mission_loop_state.json)
4. [LOOP_POLICY.md](/Users/elichen/code/perfect_snake/LOOP_POLICY.md)
5. [mission_loop.py](/Users/elichen/code/perfect_snake/mission_loop.py)

## Bottom Line

We are not trying to teach the agent how to play Snake anymore.

We are trying to turn:
- a policy that deterministically reaches the last few moves every time

into:
- a policy that makes the last few moves correctly every time

Current best deterministic checkpoint:
- [ppo_research_145_micro_s183](/Users/elichen/code/perfect_snake/experiments/ppo_research_145_micro_s183_177559752735/summary.json)
- optimistic 20-episode slice: `392.35 / 397`
- `0 / 20` wins in its saved eval
- `0 / 200` wins in the deeper deterministic probe

That is close enough that the problem still looks solvable, but the correct next move is no longer “more loop churn.”

The next researcher should:
1. replace the noisy gate with deep deterministic evaluation
2. use the `3`-ply pre-death mismatch signal as the next training target
3. only then decide whether the current PPO line is still alive
