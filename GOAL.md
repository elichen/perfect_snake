# Research Goal (locked 2026-05-29)

## North star

A **pure neural network** that wins 20x20 Snake **100% deterministically** by playing
**intelligently** — seeking food and navigating efficiently — **not** by tracing a
fixed Hamiltonian cycle.

The cycle follower is a reward hack: it solves the board but ignores the food and runs
a screensaver. The PPO navigator has the right spirit (it hunts food and reads space)
but cannot close reliability. The target wants **PPO's brain with the cycle's
discipline**: take food-seeking shortcuts whenever a shortcut is provably safe, and
respect the space-filling structure only when the board gets tight.

## Why intelligence and 100% pull against each other

Greedy food-seeking is provably **not** 100% safe in Snake — a purely greedy snake can
always trap itself. So "fully solve" and "play intelligently" are in genuine tension.
The resolution is the classic near-optimal Snake strategy (space-filling discipline +
provably-safe shortcuts), which is both efficient and 100%. The open research question:
**can a pure NN learn that shortcut-safety judgment?**

## Operational definition (so "reward hack" is a number, not a vibe)

1. **Win rate — hard constraint.** 397/397, deterministic greedy, pure NN at inference,
   no planner/search/rule fallback. Certified per `DONE_BAR.md`.
2. **Path efficiency — co-primary objective (not a tiebreaker).** `steps_per_food` must
   sit far below cycle-grade. The current RNN cycles at ~100 steps/food; an intelligent
   player should be far lower. **The exact numeric bar is TBD — calibrate against real
   intelligent play before fixing it.**

A solution that hits 100% but at ~100 steps/food **fails this goal** — it's the wrong
kind of win.

**Efficiency must be fill-stratified** (codex review, 2026-05-29): a single
`steps_per_food` hides where the intelligence lives — a policy can be smart until 80%
fill, then cycle, and still post a mediocre-but-undiagnosed average. Report and gate on:
- `steps_per_food` overall, plus per fill bucket: `0-50`, `50-80`, `80-95`, `95-100`.
- `actual_steps / shortest_path_to_food` when the food is reachable.

Geometry anchor: expected Manhattan distance between two random cells on an empty 20x20
board is ~13.3, so a perfect early-game seeker averages a bit above that; body
constraints raise the late-game cost. Candidate bars (lock after measuring a real
intelligent teacher): weak anti-cycle `<70`, credible `<50`, strong `<35-40` overall;
plus `<30-35` before 80% fill so the policy can't be cycle-like from the start.

## What this demotes

The RNN seed-by-seed repair line is now a **baseline to beat on efficiency-at-
reliability**, not the main effort. It is the narrow deep hole; we keep it as a
reference, not a thrust.

## Yardstick

`DONE_BAR.md` (win-rate certification) **plus** the efficiency co-objective above. Every
research line is measured on the same yardstick. Promote nothing that wins the cycle way.

## Portfolio decision (2026-05-29, after codex review)

Claude (PI) decision. Codex's ranking was A > C > D > B and converged with my lead.

- **Lead: A — search/safety teacher → DAgger distillation into a pure NN.** It's the
  only direction that can explicitly *generate* the missing joint behavior (efficient
  food-seeking + space-filling safety). The repo's distillation priors don't kill it:
  those were naive BC copying brittle trajectories into a net already in a cycle
  attractor (compounding error), not teacher amplification with DAgger + value/safety
  targets.
- **Folded into A, not run standalone:**
  - **C (efficiency-first reward)** as a *training objective inside A*, never as
    standalone reward tinkering — reward design alone historically caused oscillation
    and self-trapping. Safety must be enforced (curriculum / shielding / constrained opt).
  - **Viability/safety critic** (codex's 5th direction) as a stronger distillation
    target than action imitation: teacher emits action + safety/value labels; student
    learns policy + value + viability; inference uses policy logits only.
  - **D (hierarchical hunt/fill)** only if teacher labels later expose clean modes —
    learned mode gates are failure multipliers otherwise.
- **B (board-size curriculum/transfer)** as support infrastructure (data/regime tool
  inside A), not a main bet. Generic conv transfer is low-confidence; any size-agnostic
  arch must have explicit global reasoning (GNN / set-transformer over occupied cells /
  conv + flood-fill aux).

**First move — teacher feasibility probe (the cheapest falsifier for A, and it also
calibrates the efficiency bar).** Before building any distillation, prove the teacher
itself is both safe and efficient: a food-seeking player with a safety shield
(shortest-path-to-food, take the shortcut only if it preserves a space-filling
completion; else fall back to the Hamiltonian backbone). Measure win rate + fill-
stratified steps_per_food on fresh suites (10x10 first, then 20x20). Decision rule: if
the teacher can't hit ~100% at `<50-60` steps/food on 20x20, A has no payload and I
pivot. If it can, that number calibrates the goal's efficiency bar and becomes the
distillation target. Note: the repo's existing `grid_path` teacher wins but at ~96
steps/food (cycle-grade) — it is NOT this teacher; the safe-shortcut teacher is new.

## Teacher feasibility result (2026-05-29) — A is GO

Built `safe_teacher.py` (Hamiltonian cycle + Tapsell-style safe shortcuts; shortcut only
below ~50% fill, follow the cycle after). Measured on FRESH seeds, greedy planning:

| Player (same harness, fresh seeds) | 20x20 win | steps/food | mean win steps |
|---|---|---|---|
| Pure cycle (= the RNN's strategy) | 100% | 101.2 | 40177 |
| Repo's existing `cycle` shortcut teacher | 100% | 95.9 | 38086 |
| BFS greedy + tail-safety | 0% (stalls late) | — | — |
| **My safe-shortcut teacher (disable@0.5)** | **300/300 (100%)** | **52.5** | **20836** |

Zero failures across 320 fresh 20x20 seeds + 50 fresh 10x10 seeds. The teacher is
provably safe (always falls back to the full cycle) and ~2x more efficient than the
cycle, clearing the `<50-60` "credible intelligent" bar.

Two verified findings that shape distillation:
- **The late-game lever.** Shortcuts help only in the first ~50% of fill. Both the
  repo's existing shortcut teacher AND a never-disable ablation bloat the late game
  (steps/food climbs to 120-160, worse than the plain cycle). Disabling shortcuts past
  ~50% fill is the single change that takes the teacher from ~96 to ~52. This also
  explains why prior distillation plateaued at cycle-grade: its teacher was effectively
  cycle-grade.
- **Greedy alone can't finish.** A BFS shortest-path teacher with tail-reachability
  safety is near-optimal early (0-50% fill: 33 steps/food) but wins 0% — it stalls in
  the endgame with no space-filling discipline. Confirms the GOAL.md thesis: 100%
  requires cycle discipline; efficiency requires shortcuts; the win is the blend.

**Calibrated efficiency bar (locked):** the distilled pure NN must win 100% AND reach
roughly the teacher's efficiency — target `steps_per_food <= ~55` overall (vs the cycle's
~101), with the `0-50% fill` bucket the main lever (teacher cuts it 152 -> 54). A
`<35-40` "strong" bar would require a smarter teacher (true shortest-path-to-food with
flood-fill safety) and is backlog, not the current gate.

**Distillation probe result (2026-05-29): pure BC/DAgger FAILS.** Distilling the safe
teacher into a pure RNN on 10x10 reached 99.x% teacher-action accuracy but 0% greedy
win (score plateaus ~16-26/97, dies mid-game). DAgger rounds didn't rescue it. This is
the third independent confirmation of the compounding-error wall (the repo's grid_replay
94%->0/5 and grid_replay_dagger 96.6%->0.2 in the 2026-05-05 log). Root cause: ~1%
per-step error over a ~1300-step game leaves the teacher's manifold, and the safe
teacher's labels are only valid on its own cycle-aligned states. Offline imitation is
the wrong tool no matter the accuracy. Artifacts: `experiments/distill_safe_10x10_s44*`.

**Pivot: RL-guided efficiency (A + C combined).** The historically-successful path was
online (the winning RNN came from online RL/BC, not offline BC). Train with PPO so the
agent learns from its OWN experience (no compounding error), guided toward the safe
teacher's efficient-safe behavior via efficiency reward shaping and/or teacher
action-advice / KL anchor. Win stays the hard constraint (death penalty dominates).
Target: push steps/food from cycle-grade toward the teacher's ~15 (10x10) / ~52 (20x20)
while holding 100% win. Validate on 10x10 first, then scale.

## KEY FINDING (2026-05-29): imitating the efficient teacher is OBSERVATION-bottlenecked

Two distillation methods both fail to learn the safe teacher, for the same reason:
- Offline DAgger (train_bc_rnn): 99% teacher-action accuracy, 0% greedy win (compounding error).
- Online hidden-carry BC (rnn_online_cycle_bc, the method that produced the winning
  cycle-RNN): **0% win, accuracy plateaus at ~77% — it cannot even fit the teacher.**

The ~77% accuracy ceiling is the tell. The teacher's *shortcut* decisions depend on
GLOBAL cycle geometry (head/food/tail positions along the Hamiltonian cycle) that is
not recoverable from the egocentric local observation. The pure cycle IS imitable
(local obs + recurrent phase -> "continue the serpentine"); the shortcuts are not.
**The bottleneck is the observation, not the training method.** This is why every prior
efficiency effort plateaued at cycle-grade: the policy literally lacks the information
to decide a safe shortcut.

Routes considered:
1. **Richer observation** — TESTED flood-fill: online safe-teacher BC WITH `--flood-fill`
   left accuracy at ~0.77 and win rate at 0%, identical to plain obs. So reachability is
   NOT the missing signal; the shortcut needs cycle-position geometry. A cycle-position
   channel would carry it, but that is the "privileged feature" the cycle-conditioned
   branch already used and the mission rules out at inference. Route 1 via standard
   channels looks closed; only a non-privileged signal that happens to encode the safe
   shortcut would revive it.
2. **RL discovers obs-expressible efficiency (now the lead).** Don't imitate the
   teacher's global-info shortcuts — let PPO find efficient behaviors that ARE computable
   from the obs, with the teacher/safety as a CONSTRAINT, not an imitation target
   (codex's "behavior prior / safe-action set"). The teacher becomes a safety reference
   and an efficiency upper bound (~52 steps/food), not an action oracle.

Hard constraint for either route: the efficiency target must be expressible from the
observation the final (pure-NN) policy uses. The RL-guided design below is the lead.

## VALIDATED (2026-06-01): the efficient winner is real, and the failure is the endgame

Strong 200-fresh-seed deterministic audit of `route2_ppo_20x20_eff/best_eval.pt`
(seeds 200001-200200), answering Andrej's "is 12% real / survivor-biased?" challenge:
- **21/200 wins = 10.5%, Wilson 95% CI 7.0-15.5%** — above the "20+/200 -> real" bar.
  The win signal is real, not eval noise.
- **steps_per_food = 47.9 counting ALL episodes (incl. losses)** — not just winners.
  Efficiency is genuine (~half the cycle's ~100); survivor bias ruled out.
- mean 324, median 365, max 397, min 14. Deaths: 174 self, 5 stall, 0 wall.
- **Failure concentrates at the endgame:** death-fill buckets (10% each) =
  [1,8,10,4,7,1,5,14,34,116]. 150/200 die at >=80% fill, 116/200 in the final 90-100%
  bucket. Exactly where topology discipline is needed.

Andrej's review corrections folded in: (1) "continuations don't help" is too strong —
only naive constant-LR continuation was shown to regress; a low-LR/KL-stabilized/
safety-shaped continuation is untested. (2) The obs-bottleneck is "these teacher action
labels aren't learnable from this obs/arch", not a hard impossibility. (3) The 12%
policy might solve an easy seed-subset with no scalable path to the hard subset — the
topology experiment is the test.

## AWS ATTEMPT + LOCAL SWEEP (2026-06-02)

Tried AWS (account2, us-east-1). Key finding from a calibration instance: **CPU is ~20x
too slow** — c7i.4xlarge got 342 SPS (`--device cpu` runs the 4.4M model's backprop on
the same cores as the env, vs the Mac's MPS offloading the model). 150M steps = ~120h on
CPU. So the workload needs a **GPU** instance (g5, `--device cuda`); the Mac's ~8k SPS is
GPU-assisted + env-bound. Realistic GPU sweep cost: ~$3/run, ~$15-20 for 6 runs. Since
the Mac runs these for FREE (serially, ~5h each) and AWS only buys parallelism/wall-clock,
chose **free serial on the Mac**. All AWS resources torn down (S3 + IAM + instances);
total spend ~$0.30. Calibration also nailed the remote setup recipe (AL2023 +
python3.11-devel + CPU torch + pinned freeze) if GPU is ever wanted — note the `gym==0.23`
+ pufferlib-from-source + `Python.h` (python3.11-devel) gotchas.

Local serial sweep RUNNING (`run_local_sweep.sh`, ~16h, MPS, free): tail-safety -0.15
(s101) -> control (s202) -> tail-safety -0.15 (s303), each with a 200-fresh-seed audit
(200001-200200) for apples-to-apples vs the 10.5% baseline. Watching: does tail-safety
win >0% (the topology penalty got 0% + wall deaths), and does the death-fill cluster
shift down? Plus control s202 adds a 2nd baseline draw for the variance question.

## READY-TO-FIRE AWS SWEEP PLAN (written 2026-06-02)

Gate met: substrate validated (10.5% win, 47.9 steps/food all-episode), and one local
5-hour run per mechanism is variance-limited (can't separate effect from basin luck).
GPU/parallel exploitation is now justified. Cleaner mechanism built:
`--tail-safety-penalty` (flat penalty when a surviving high-fill move leaves the tail
unreachable — binary, survival-relevant, no boundary-avoidance pathology) + existing
`--topology-penalty` (stranded-cells, known-bad at -0.15) + existing
`--aux-safe-action-soft-target`.

**Hardware — vCPU-bound, NOT GPU-bound.** The model is a 4.4M MLP; the bottleneck is the
Python env (scipy flood-fill + head-centered 39x39 obs). MPS gave ~8k SPS, almost all
env time. So prioritize **vCPU count and parallel jobs**, not a big GPU:
- Pick a vCPU-rich instance (e.g. c7i.24xlarge ~96 vCPU; or g5.12xlarge if a GPU is
  wanted: 48 vCPU + 4xA10G). Run N independent training jobs in parallel (each
  num-envs=256 uses many cores); one box runs ~4-10 jobs.
- The model on GPU is a minor win; the leverage is many concurrent runs.

**Highest-leverage prep (do before/with the sweep): speed up the env.** 2-5x per-run.
1. Replace `scipy.ndimage.label` flood-fill with a hand BFS / numba kernel.
2. Cache the per-step flood-fill: obs, topology_penalty, tail_safety all recompute it —
   compute once per step and reuse.
3. Profile head-centered obs construction (39x39 rebuild each step).

**The sweep grid** (each run = exact winning recipe: head-centered + flood-fill obs, aux
flood-fill, MLP 2x, gamma 0.999, horizon 256, gae-lambda 0.9, vf-clip 1.0,
curriculum-prob 0.3, 150M steps, DEFAULT LR anneal-to-zero — that schedule produced the
wins). Vary:
- **Controls:** plain recipe x {6 seeds}. CRITICAL — establishes the true baseline win-
  rate DISTRIBUTION (is 10.5% typical or a lucky draw? this is the missing denominator).
- **tail-safety:** `--tail-safety-penalty {-0.05,-0.15,-0.3} --tail-safety-min-fill 0.80`
  x {4 seeds each}.
- **tail-safety + aux:** best tail-safety coef + `--aux-safe-action-soft-target
  --aux-safe-action-soft-target-min-fill 0.80` x {4 seeds}.
- **(diagnostic) topology at low coef:** `--topology-penalty -0.05` x {3 seeds}
  (Andrej: separate signal from magnitude on the stranded variant).
~30-40 runs total.

**Eval gate (per candidate best_eval):** 200 fresh deterministic seeds (e.g. 200001+,
quarantined), report win_rate + Wilson CI, all-episode steps_per_food, death-fill
histogram. PROMOTE only if the win-rate CI clears the control distribution AND
steps_per_food <= ~55. Then certify survivors per DONE_BAR (5000-seed). Decision rule:
if NO mechanism arm's CI beats the control distribution, viability-shaping is falsified —
pivot (e.g. recurrent policy, or search-in-the-loop endgame).

**Rough cost:** ~30-40 runs x ~150M steps. With an optimized env at ~30-50k SPS on
many cores, ~1-2 hr/run, ~8-10 parallel -> a few instance-days. Tens-to-low-hundreds of
USD, not thousands.

## RESULT (2026-06-10): tail-safety -0.15 (s101) ALSO FAILED — penalties are the wrong form

`sweep_tailsafe015_s101` 200-seed audit (200001-200200, vs baseline 21/200 = 10.5%):
- **0/200 wins** (Wilson CI 0-1.9%); steps/food 47.0 (efficiency held); deaths 198 self / 2 wall.
- Death-fill histogram [1,8,17,17,16,15,18,22,71,15]: the 90-100% bucket dropped from
  the baseline's 116 to 15 — the penalty DID reduce endgame trap deaths — but deaths
  spread across every mid-fill bucket instead and mean score fell 324 -> 256.
- **Interpretation: with gamma=0.999, a fill-gated penalty is not actually gated.**
  Values propagate ~700 steps, so extra negative reward at >=80% fill reshapes the whole
  value landscape and degrades mid-game play. Both failed mechanisms (topology, tail-
  safety) injected late-fill penalties on top of the existing death penalty; both
  collapsed wins the same way. If shaping returns, it should be potential-based
  (policy-invariant PBRS) or positive-only (safe-action bonus), not penalties.

**Incident:** the Jun 2 sweep driver was killed Jun 3 ~10:14 (terminal closure; no
traceback) — control s202 died at 40M/150M and tail-safety s303 never ran. Relaunched
2026-06-10 nohup'd (`run_local_sweep2.sh`): s202 resumed from best_eval_resume.pt
(35.3M, optimizer+scheduler restored) to 150M, then a fresh control s404. **Plan change:
s404 replaces s303** — with two mechanisms at 0/200, the control DISTRIBUTION (is 10.5%
reproducible or a lucky draw?) is the decision hinge, not a second tail-safety seed.
Decision rule unchanged: controls reproduce ~8-15% => penalty-shaping falsified, pivot
to PBRS/positive shaping or recurrent/search-endgame lines; controls collapse to ~0-2%
=> basin variance dominates, local serial A/B is underpowered => env-speedup + parallel
seed sweeps.

## PAUSED 2026-06-13 (battery). Resume state + next action

All training/drivers/env-workers stopped cleanly; no orphans. Checkpoints intact:
- **THE LEVER (resume here):** `cont_ctrl_s202r_178127544146/best_eval_resume.pt` =
  the 83.5%-win checkpoint. Push continuation leg 2 from it.
- PBRS arm was interrupted at ~161M (only ~15M into its +120M leg, eval win still ~40%
  = uninformative — same slow-start shape cont_ctrl had). Partial resume ckpt saved at
  `cont_pbrs_s202r_178130394869/best_eval_resume.pt` if we want to finish the paired arm.

**To resume:** `nohup ./run_local_sweep4.sh > experiments/local_sweep4_driver.log 2>&1 &`
(continuation leg 2, +120M at constant 1e-5 from the 83.5% ckpt; auto-audits 200 seeds).
Decision deferred: PBRS gave no verdict, so leg 2 runs plain control by default; only
carry PBRS (`run_local_sweep4.sh pbrs`) if we first finish the PBRS arm and it beats
83.5%. Recommendation: skip PBRS, just pull the continuation lever — it's the clear win.

## BREAKTHROUGH 2 (2026-06-12): low-LR continuation = 83.5% win at 44.2 steps/food

`cont_ctrl_s202r` (resume 42.5% ckpt @146.3M, constant lr 1e-5, +120M -> 266M):
**167/200 wins (83.5%, Wilson 77.7-88.0%)**, steps_per_food 44.2, **median score 397**
(the median episode is a win), mean_win_steps 17.6k (cycle RNN: ~39.7k). Stratified
steps/food 39.8 / 61.9 / 35.1 / 7.1 — every late bucket improved sharply.

Two structural findings:
1. **The endgame wall fell.** Exactly 1 of 33 deaths in the 90-100% bucket (controls:
   ~95-125). Residual losses scatter across early/mid fill (14 of 33 below 30% fill) —
   rare-state blunders, not systematic trapping. The remaining 16.5% is a different,
   smaller problem than the one the project has fought for months.
2. **The melt fear was wrong at the right LR.** Constant 1e-5 didn't just hold the
   basin; eval win climbed 0.4 -> 0.9 across the leg and was STILL RISING at 266M.
   The lever is not exhausted. (May's regression was the LR, 1.5e-4, not continuation
   per se — Andrej's correction stands confirmed.)

Plan: keep pulling the lever. Leg 2 (+120M at 1e-5) launches from cont_ctrl's best
checkpoint as soon as the paired PBRS arm audits; if cont_pbrs materially beats 83.5%,
leg 2 carries PBRS instead. As win rate approaches ~95%, switch promotion claims to
DONE_BAR discipline (fresh holdout blocks, then 5000-seed certification; the 200001-
200200 suite stays as the comparable iteration yardstick only).

## UPDATE (2026-06-12): fresh control s404 = 41.0% — ~40% is the recipe's TYPICAL outcome

`sweep_control_s404` (fresh scratch draw, seed 404, 150M): **82/200 (41.0%, Wilson
34.4-47.9%)**, steps_per_food 49.7, mean 337.8, deaths 107 self / 9 wall / 2 stall.
Control distribution now {10.5%, 42.5%, 41.0%}: two independent ~41-42% draws say the
May 10.5% baseline was the unlucky tail, not the typical case. Pooled controls 188/600
(31.3%) vs penalty mechanisms 0/400 — verdict unchanged, stronger. Both June controls
show ~5% wall deaths (May baseline had 0) — minor, watch it.
Paired continuation (sweep3) launched 2026-06-12 07:37 from the s202r 42.5% checkpoint:
verified constant lr 1e-5 (PufferLib only steps the scheduler when anneal_lr is on, so
--no-anneal-lr + --override-resume-lr holds 1e-5 exactly), arms cont_ctrl_s202r then
cont_pbrs_s202r (-0.3 @ 0.80), each +120M to 266M, same seed 777, 200-seed audits.

## BREAKTHROUGH (2026-06-11): control redraw s202r = 42.5% win at 50.3 steps/food — NEW BEST

`sweep_control_s202r` (exact winning recipe, seed 202, 150M steps; the resumed control):
**85/200 wins (42.5%, Wilson 35.9-49.4%)**, steps_per_food 50.3 all-episode, mean 325.6,
deaths 102 self / 10 wall / 3 stall. 4x the prior best (21/200 = 10.5%).

Supplemental 50-ep stratified eval of its best checkpoint (new eval.py metrics):
win 23/50, median score **388.5/397** (half of episodes reach the final ~2% of cells),
steps_per_food_by_fill = **39.8 / 70.7 / 56.1 / 22.0** (0-50 / 50-80 / 80-95 / 95-100).
The early game is MORE efficient than the safe teacher's 54-in-bucket; the endgame
finishes at a stunning 22 steps/food when it finishes. The whole mission deficit is now
"survive the last cells": deaths cluster at 90-100% fill.

What this settles and reopens:
1. **Mechanism verdict is final:** pooled controls 106/400 vs pooled penalty mechanisms
   0/400 (topology, tail-safety flat) — Fisher p ~ 1e-31. The flat late-fill penalties
   were genuinely harmful. Closed.
2. **Basin variance is enormous:** same recipe drew 10.5% and 42.5%. From-scratch
   single-run A/Bs are uninterpretable; mechanism tests must be paired or multi-seed.
3. **LR-freeze insight:** s202r's wins emerged at 126-146M, where the cosine-to-zero LR
   had already fallen to ~2e-5 -> ~5e-7. Wins crystallize AS the LR freezes. This
   explains the May 29 continuation regression at constant 1.5e-4 (re-melted the basin).
   Continuations must run at the LR where gains began (~1e-5) or below.
4. Env speedup did NOT raise training SPS (still ~8.2k) — training is not env-bound;
   the speedup pays off in audits/evals/tools instead.

**Locked next experiment (paired continuation, fires after s404's audit):**
resume `sweep_control_s202r_178111665825/best_eval_resume.pt` (146.3M, the 42.5%
checkpoint, Adam moments intact) with `--resume-add-steps --override-resume-lr
--lr 1e-5 --no-anneal-lr --timesteps 120000000`, two arms sharing init and env seed:
- A `cont_ctrl_s202r`: continuation alone (does the basin hold/improve at 1e-5?).
- B `cont_pbrs_s202r`: + `--tail-safety-pbrs -0.3 --tail-safety-pbrs-min-fill 0.80`.
  The paired warm start makes this the first interpretable shaping test; PBRS is
  policy-invariant so arm B cannot lose to A by value-landscape distortion alone.
Gate: 200-seed audits (200001-200200), report win CI + steps_per_food (must stay <=~55)
+ stratified buckets. KILL any arm that regresses win-rate CI below s202r's 35.9% floor.

## PREP (2026-06-10): env 2-4x faster, PBRS tail-safety built, fill-stratified eval

Built while sweep2 trains (all gated on bitwise equivalence, `test_env_equiv.py`):
1. **snake_env speedup (GOAL prep item done):** incremental occupancy grid (collision,
   body channel, food placement, flood-fill input), precomputed scipy label structure +
   buffer, per-state flood-fill cache (obs/penalties/tail-safety share one compute),
   walls-as-rectangle, history-copy guard. Single-process SPS: long-snake 10.9k -> 44.9k
   (4.1x), short 17.1k -> 35.4k (2.1x). Bitwise-identical obs/reward/term vs the old env
   across 7 configs incl. wins/stalls/penalties. `bench_env.py` is the profiling harness.
2. **`--tail-safety-pbrs` (the corrected mechanism):** potential phi = coef (negative)
   while tail-unreachable at fill >= min-fill, else 0; reward += gamma*phi(s')-phi(s).
   Entry charge refunded on recovery — telescopes to ~0 on surviving paths, so it cannot
   distort the mid-game value landscape the way the flat penalties did (the diagnosis of
   the topo/tail-safety 0/200 failures). Dying while unreachable keeps the entry charge
   (no terminal correction, matching the distance-shaping convention) = net signal
   exactly on the failure mode. Tail-reachability now derived from the cached flood-fill
   (fuzz-verified == BFS on 4k states); verified shaped term == gamma*phi'-phi exactly vs
   a shadow env, no obs/termination leakage. Cost when active: ~free (reuses the obs ff).
3. **Fill-stratified efficiency in eval.py** (`steps_per_food_by_fill`, buckets
   0-50/50-80/80-95/95-100 per the yardstick). Sanity-run on route2 10x10: 13.3 / 25.1 /
   66.0 / — : efficient early, breaks down exactly where discipline is missing.

## RESULT (2026-06-01): topology penalty -0.15 (stranded-cells) FAILED

`route2_ppo_20x20_topo` 200-seed audit (seeds 200001-200200, vs the 10.5% baseline):
- **0/200 wins** (CI 0-1.9%) — killed the wins (baseline 21/200).
- **52 wall deaths (26%)** vs baseline's 0 — a new pathology. This is mechanism-
  attributable (a plain policy never hits walls), so the penalty DID change behavior,
  and harmfully: the stranded-cells signal is too blunt and distorts boundary play.
- Deaths still endgame-heavy (149 at >=80% fill); steps/food 44.7 (moot, no wins).

Two takeaways: (1) the blunt stranded-cells penalty is the wrong signal — replace with
Andrej's explicit action-conditional tail-reachability. (2) **We've hit the local-
experiment ceiling:** one 5-hour from-scratch run per mechanism can't separate a real
effect from the project's large basin variance (baseline 10.5% was itself one lucky
draw). Cleanly testing a mechanism now REQUIRES a seed x mechanism sweep — i.e. this is
the point where AWS parallelism is justified. Don't keep burning serial single runs.

## EXPERIMENT (done): topology penalty (Andrej's lead mechanism)

New `--topology-penalty` in snake_env/train.py: at fill >= 0.80, a surviving move that
strands free cells (head can't reach part of the free space = the self-trap) gets a
penalty scaled by the stranded fraction (reuses flood-fill; only on surviving moves so
no suicide incentive). `route2_ppo_20x20_topo` = exact winning recipe (default LR
anneal-to-zero preserved) + `--topology-penalty -0.15 --topology-penalty-min-fill 0.80`,
150M steps. Clean A/B vs the 10.5%-win baseline. Hypothesis: RL already seeks food well;
it needs topology discipline near the end. Gate candidates on 200 fresh seeds
(failure-inclusive steps_per_food must stay <=~55). If it shows signal, scale on AWS
(seed x coefficient sweep); if flat, build the stronger action-conditional viability
version (tail-reachability labels + aux head).

## BREAKTHROUGH SUBSTRATE (2026-05-29): an efficient pure-NN policy that wins (sometimes)

Strong-config 20x20 PPO (head-centered, flood-fill, aux, gamma 0.999, horizon 256,
curriculum; the exp072 recipe) trained 150M steps. Clean deterministic eval of its
best checkpoint (`experiments/route2_ppo_20x20_eff_178007982988/best_eval.pt`, 50 eps):
- **win_rate 12%**, mean_score 322, median 371
- **steps_per_food 48.3** (vs the cycle's ~100), **mean_win_steps 19,065** (vs the cycle
  RNN's ~39,700) — its WINS are less than HALF the cost of the cycle frontier.

This is the mission's two halves both demonstrated as pure NNs on 20x20:
- This PPO policy: efficient (48 steps/food) but only 12% reliable (self-traps at ~91% fill).
- The RNN frontier: 98.6% reliable but cycle-grade (~100 steps/food).

The intelligent-AND-winning policy EXISTS; it is just undertrained/unreliable. This run
did only 150M steps (exp072 reached 94% mean at 260-500M) and was still improving (hit
20% eval win at the end). So the immediate lever is MORE training + (if it plateaus at
the endgame wall) explicit late-game discipline. Continuation running:
`route2_ppo_20x20_eff_cont` (resume best_eval, +120M steps, constant lr 1.5e-4).

## Route-2 baseline result (2026-05-29): RL = efficient brain, endgame = missing discipline

Plain PPO on 10x10 (flood-fill, 25M steps): mean score ~80/97, **0% win — plateaus at
~80% fill and self-traps**, while playing efficiently (~21 steps/food live, vs cycle 25 /
teacher 15). So RL naturally finds efficient food-seeking but lacks endgame discipline to
finish — the mirror image of the cycle (finishes, inefficient) and the BFS teacher
(efficient, stalls). The 0% on 10x10 is partly known 10x10 variance (PPO has won 10x10
before), but the efficient-but-traps split is the signal.

Synthesis hypothesis: efficiency comes free from RL; the mission's hard part is keeping
the WIN once efficiency is allowed. On 20x20 the endgame pressure is exactly what pushes
RL toward cycle-like packing (losing efficiency) — that is why 20x20 is hard and 10x10
isn't. So the real question: does a strong 20x20 PPO policy play efficiently or
cycle-like, and can late-game discipline (safe-action at very high fill only, NOT
cycle-follow which would re-impose the cycle) close the endgame while preserving early
efficiency? Next run: measure strong-config 20x20 PPO efficiency (ep_len/ep_score) as it
trains; artifacts `experiments/route2_ppo_10x10_178007858053` (10x10, ~21 steps/food, 0% win).

## Locked next-experiment design (RL-guided, after codex review 2026-05-29)

Codex reward ranking (lowest risk first): (b) teacher action-advice bonus, early-fill
only >> (c) KL anchor to the **incumbent** (not the teacher — KL-to-teacher recreates the
BC wall) >> (d) terminal inverse-step bonus (too sparse; later tie-breaker) >> (a)
per-step time penalty (worst: suicide/stall/oscillation, do NOT start here).

Cleaner formulation (adopt): treat the teacher as a **behavior prior / safe-action set**,
not a trajectory to copy. PPO learns from its own experience; reward/aux prefer
teacher-sanctioned safe actions; among safe actions RL optimizes efficiency. Optionally
train a safety/viability aux head from teacher labels (NOT used at inference — pure NN
logits only at eval).

Recommended first run (codex): **fine-tune a WINNING RNN** (not scratch, not the BC
student) with:
- food +1, death -1, existing stall termination, NO per-step penalty;
- early `safe_teacher` action-match bonus for `fill < 0.5` (small, positive-only);
- incumbent KL/sequence-KL anchor, strongest for `fill >= 0.5`;
- late cycle/follow/safe-action preservation for `fill >= 0.5`;
- eval by fill buckets: win rate + steps/food overall and per bucket.
- KILL: any win-rate regression or late-fill degradation. SUCCESS: still 100% win AND a
  clear `0-50%` steps/food reduction vs the incumbent.

Open dependency: codex recommends 10x10 first (cheapest), but no winning 10x10 RNN exists
on disk. Options: (i) train a winning 10x10 RNN incumbent first via the online RNN
trainer, then fine-tune; or (ii) apply the recipe directly to the 20x20 incumbent
(`broad_anchor`, ~101 steps/food, wins ~98.6%) — the real target, more expensive.
Vehicle: `rnn_online_cycle_bc.py` / `ppo_lstm_online_cycle_bc.py` (online PPO+BC for the
RNN), to be adapted for safe-teacher advice + incumbent KL.

## Role

Claude is the research driver/PI. Codex is the peer (reviews + runs; see `AGENTS.md`).
