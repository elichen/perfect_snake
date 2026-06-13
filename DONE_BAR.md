# Mission Done-Bar (20x20 Snake)

Date: 2026-05-29

This file fixes the acceptance criteria for the refined mission so we stop chasing a
receding target. Every fresh holdout we have run keeps surfacing a new tail failure
(40004 → 40055 → 60131/60146 → ...). Without a defined bar, "done" never arrives,
and a repair that passes the suites it was tuned on reads as success when it isn't.

Companion file: `experiments/seed_registry.json` (burned ranges, scar seeds, reserved
fresh blocks). Engine: `rnn_promotion_audit.py` (deterministic greedy inference only).

## 1. What "done" means

- **Pure neural network**, deterministic greedy inference, no planner / search / rule
  fallback, no privileged inference-time features.
- Scores **397/397**.
- **Win rate is a hard constraint.** Path efficiency (mean / p95 / steps-per-food)
  ranks candidates *among win-rate passers only*; it never buys back a single failure.

## 2. The statistical reality (why a single number isn't "100%")

The seed space is effectively unbounded and the policy is deterministic per seed, so
we can never *prove* the true failure rate `p = 0` empirically. We can only bound it.
With `N` fresh seeds and **zero** failures, the rule of three gives a 95% upper bound:

| Fresh clean seeds | 95% upper bound on failure rate `p` | Plain reading |
|---|---|---|
| 200 | < 1.5% | weak — current 60001–60200 was here |
| 600 | < 0.5% | regression-grade only |
| 5,000 | < 0.06% | certification-grade |
| 50,000 | < 0.006% | near-claim of mastery |

The current frontier fails 2/200, so it is nowhere near even the 5,000-seed bar.
A literal "100%" claim needs either ~50k clean **or** a structural argument
(§6), not just passing the suites we happened to test.

## 3. Seed allocation and quarantine discipline

The one rule that makes the whole gate meaningful: **a seed used for training,
anchoring, scanning, or any prior validation can never count as a fresh holdout pass.**
The repair chain has repeatedly harvested failures from a suite and then "validated"
on the overlapping suite. That is contamination and it invalidates the certification.

Tracked in `experiments/seed_registry.json`:

- **Burned ranges** — `20001–50100`, `60001–60200`. Stay as *regression* suites
  (must keep passing); never valid as fresh holdout.
- **Scar set** — append-only registry of every seed ever observed to fail a
  promoted-track checkpoint (currently the 30 hard seeds + 60131, 60146 + a few
  historical). Always in every gate. Add, never remove.
- **Iteration holdout blocks** — fresh 200-blocks (`90001–90200`, ...), one consumed
  per promotion attempt, then burned.
- **Certification blocks** — large fresh blocks (`70001–75000`, ...), each used at
  most once. A failure burns the block and moves the seeds to the scar set.

## 4. The two-stage gate

### Iteration gate (fast — run on every candidate)

Pass requires **100%** on all of:
- Regression suite: `20001:100,30001:100,40001:100,50001:100,60001:200` (600).
- Full scar set.
- One fresh rotating iteration-holdout block (200), drawn from the registry.

`--stop-after-failures 1` is fine here (fail fast). Path efficiency is *reported* on
the fixed 600-seed suite (`mean_win_steps`, `p95_win_steps`, `steps_per_food`) but is
not blocking at this stage.

```bash
.venv/bin/python -u rnn_promotion_audit.py <candidate>.pt \
  --board-size 20 --hidden-size 512 --device mps \
  --ranges 20001:100,30001:100,40001:100,50001:100,60001:200,iter=90001:200 \
  --hard-seeds 40055,50085,50090,40043,40004,40099,50052,20008,20031,20038,20039,20060,20064,20099,30014,30034,30058,30074,30082,30083,30086,30097,40015,40067,40089,40090,50013,50029,50045,50076,60131,60146,40014,20100,20250 \
  --max-steps 100000 --stop-after-failures 1 \
  --out experiments/<candidate>_iteration_audit.json
```

### Certification gate (run ONCE, only after the iteration gate is clean)

Pass requires **0 failures** on a fresh quarantined block of **≥ 5,000 seeds**, run
with `--stop-after-failures 0` so every failure is counted (not just the first).

```bash
.venv/bin/python -u rnn_promotion_audit.py <candidate>.pt \
  --board-size 20 --hidden-size 512 --device mps \
  --ranges cert=70001:5000 \
  --max-steps 100000 --stop-after-failures 0 \
  --out experiments/<candidate>_certification_audit.json
```

Any failure: append the failing seeds to the scar set, mark the block burned in the
registry, reserve the next block, and return to repair.

## 5. Promotion criterion (mission complete — empirical)

A checkpoint is **DONE (empirical)** iff *all* hold:
1. Audit asserts pure-NN greedy, no fallback.
2. Regression suite 600/600.
3. Full scar set 100%.
4. Fresh iteration holdout 200/200.
5. Certification holdout ≥ 5,000 / ≥ 5,000 (→ 95% bound `p < 0.06%`).

Record the certification count and the implied bound in the promotion note.

## 6. Gold standard (literal 100% claim)

Empirical bars certify; they do not prove. The literal claim needs one of:
- **≥ 50,000** fresh clean seeds (`p < 0.006%` @95%), or
- a **structural argument** that self-collision is impossible — e.g. show the RNN
  provably tracks a Hamiltonian cycle to within shortcuts that are proven
  non-trapping. The cycle-conditioned teacher has this property by construction;
  the open question is whether the pure RNN can be shown to inherit it.

Until then, report the empirical bound honestly ("certified `p < 0.06%` at 95%
confidence over 5,000 fresh seeds"), not "100%".

## 7. Reviewer checklist (for codex-peer runs)

When reviewing a promotion claim, reject it unless:
- [ ] Training / anchor / scan seeds do **not** overlap the holdout being credited.
- [ ] Certification used `--stop-after-failures 0` (full count, real rate).
- [ ] Every newly found failure was appended to the scar set.
- [ ] Path efficiency is reported on the fixed 600-seed suite (comparable across candidates).
- [ ] No promotion is claimed on in-slice suites alone.
- [ ] The certification block was previously unused (check the registry).

## 8. Tooling gaps to close in `rnn_promotion_audit.py`

The engine is sound but the done-bar needs three small additions (proposed, not yet built):
1. `--scar-file experiments/seed_registry.json` — load the append-only scar seeds
   instead of re-typing 35 seeds per run.
2. Emit a **confidence bound** in the checklist (rule of three over total fresh
   episodes) so the report states `p < x% @95%`.
3. A **contamination check** — warn if any audited range intersects the registry's
   burned ranges while being credited as a fresh holdout.
