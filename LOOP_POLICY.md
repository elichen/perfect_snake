# Loop Policy

## Mission

- Train a pure neural network policy that achieves 100% deterministic win rate on 20x20 Snake.
- Success means a saved checkpoint that scores 397/397 reliably in greedy play.
- Peak lucky-seed scores do not count unless they can be reproduced or safely extended.

## PI View

This file is the high-level research policy for the autonomous loop.
The loop should act like a small lab:

1. Keep one control arm alive so we know the current PPO family variance.
2. Spend most budget on concrete hypotheses, not generic seed farming.
3. Preserve the strongest known basin.
4. Reject weak branches quickly.
5. Keep disk churn low and decisions explicit.
6. Treat any active run with no file activity for 30 minutes as stale; terminate it and continue the loop.

## Current Frontier

- Best saved PPO basin: `ppo_research_145_micro_s183`
- Best deterministic eval mean: `392.35 / 397`
- Current frontier profile: `lt20 = 0.0%`, `95%+ = 100.0%`, `win_rate = 0.0%`
- Best current family: strong micro basin -> one conservative short continuation -> two micro continuations
- Updated PI read: once a basin is already in the `392+` regime, full micro steps appear too coarse and tend to overshoot. The next local search step should be smaller than `resume_micro`, not another generic control sweep.
- Updated PI read: the minimal temporal-context probes are now negative. Both `obs_history=2` and `action_history_obs=2` underperformed badly, so they are retired from automatic cold-streak routing.
- Updated PI read: a cold-streak `resume_short` revisit from a strong micro basin must open in the `380+` range to justify a second hop. Low-`370s` first hops are informative but not chain-worthy.
- Updated PI read: once a specific micro basin has already failed the cold-streak `micro -> short` revisit test, suppress repeated retries from that exact source and move on to a different fallback family.
- Updated PI read: every new `390+` / `lt20=0` near-frontier incumbent should get one deeper deterministic win-probe. A `0/200` result means the basin does not already hide rare greedy wins and should not be treated as a latent eval-winner.
- Updated PI read: cold-streak `micro -> short` revisits should now use a fresh continuation seed, not the source run's seed. The same-seed revisit path has already looked too brittle to justify more repeats.
- Updated PI read: PI-killed alt-seed short resumes count as cold evidence. If a fallback family keeps getting cut before normal completion, that still means the source is spent and should be suppressed.
- Distillation is not currently competitive with PPO and should not be the default loop branch.

## Research Questions

The loop is currently testing these questions:

1. **Ultra-late failure manifold hypothesis**
   - The frontier already reaches `95%+` fill in every eval episode, so the remaining gap is the final few moves.
   - Before injecting more targeted training, harvest deterministic failures from scores `390-396` and summarize recurring endgame motifs.
   - Purpose: stop guessing about the last failure mode and make the next training branch answerable.
   - Current evidence from the `392.35` incumbent: harvested failures are dominated by `393-394` self-collisions, and whenever the late safe-action heuristic is defined on the terminal decision, the policy is choosing the wrong action. This points to ultra-late action calibration, not broad policy weakness.
   - Updated PI read: the first dedicated endgame heuristic retries from this incumbent still underperform the frontier, so cold-streak fallback should now prefer local-variance branches from the proven upstream resume/micro chain before repeating more near-perfect heuristic probes.

2. **Transient peak hypothesis**
   - Are we missing the best checkpoint because eval/checkpoint cadence is too coarse?
   - This is now de-emphasized. Dense harvest has underperformed repeatedly and should only be used on truly near-incumbent controls.

3. **Continuation stability hypothesis**
   - Can a near-mastered basin be extended with a very small, conservative PPO continuation from a full resume state?
   - Test with short low-LR resume probes from `best_eval_resume.pt`.
   - For exceptional fresh controls, prefer a **direct micro-resume** before a full `resume_short`.

4. **Control distribution**
   - Is the baseline seed family still producing new strong basins?
   - Test with occasional fresh exact seed screens.
   - Treat sub-`360` controls as cold for loop-routing purposes; they are useful variance samples, but they should not reset the research fallback machinery when the frontier is already `392+`.

5. **Cold-streak exploitation**
   - If the control arm goes cold for multiple consecutive seeds, the loop should stop spending every cycle on fresh controls and revisit the best historical near-frontier resume source.
   - Purpose: bias the search back toward the only research arm that has stayed close to the frontier.

6. **Continuation-seed variance**
   - The current frontier came from a specific resume parent (`003 -> 018`), but the loop has only probed that parent with the original continuation seed.
   - If the control arm stays cold and no near-frontier resume candidate qualifies, test a few fresh continuation seeds from the incumbent's historical parent resume state.
   - Purpose: distinguish "bad source state" from "good source state, bad continuation seed".

7. **Incumbent micro-seed variance**
   - If the parent-seed sweep saturates below the frontier, test a few fresh-seeded **micro-resume** probes directly from the incumbent's `best_eval_resume.pt`.
   - Purpose: check whether the incumbent basin itself can be nudged upward with different continuation randomness, without committing to a full short resume.

8. **Short temporal context**
   - Status: negative.
   - `obs_history=2` was tested twice and stayed in the high-`200s` / low-`300s` on deterministic eval.
   - `action_history_obs=2` was worse.
   - Conclusion: small visible-history additions are not the missing ingredient for the final `393-396` failures.

9. **Late temporal occupancy supervision**
   - If the continuation families saturate below the frontier, inject a new auxiliary target that teaches how long occupied cells will remain blocked.
   - Run this only as a conservative late-game micro-resume from a strong saved basin.
   - Purpose: the frontier already has `lt20=0`, so the remaining gap is likely temporal corridor timing rather than early-game competence.

10. **Hybrid ultra-late calibration**
   - If a body-age endgame probe lands in the high `380s` without improving the frontier, combine the temporal-occupancy target with the soft safe-action target and late confidence in a single ultra-late micro-resume.
   - Purpose: body-age covers `safe=na` motifs, while the soft safe-action target can still shape terminal choices when the heuristic is defined.

11. **Self-imitation from sampled late wins**
   - If an ultra-late targeted branch still fails greedily but shows a high fraction of train-time wins or near-perfect trajectories, feed those trajectories back with a small elite BC loss.
   - Purpose: the current evidence says the policy can sample much better last-move behavior than it executes greedily. This branch tests whether targeted self-imitation can consolidate that sampled behavior.

12. **Direct greedy sharpening**
   - If the targeted aux and self-imitation branches remain below the frontier, run a clean ultra-late micro-resume from the frontier with no extra replay and no extra teacher heads.
   - Only change:
     - lower entropy
     - explicit late-confidence sharpening
   - Purpose: test whether the remaining gap is just logit calibration in the last few moves.

## Allowed Experiment Families

### Control Arm

- Fresh exact `exp078`-style seed screens from the `exp074` resume base.
- This is the baseline, not the main budget sink.

### Research Arm A: Endgame Failure Harvest

- For a near-perfect incumbent:
  - `mean_score >= 390`
  - `phase_lt20_rate == 0.0`
  - `phase_gte95_rate == 1.0`
  - `win_rate == 0.0`
- Run a standalone deterministic evaluation harvest on the incumbent's `best_eval.pt`:
  - up to `100` episodes
  - stop early once `40` harvested failures are collected
  - collect only failures with terminal score `390-396`
  - write per-failure traces plus a motif summary
- Goal: measure the actual last-move failure manifold before committing to any more endgame-specific training.
- This is now the required precursor for any new targeted endgame micro branch on a new incumbent.

### Research Arm B: Safe-Action Endgame Micro

- If a completed failure harvest for a near-perfect incumbent shows:
  - dominant failure reason `self`
  - at least `20` harvested failures
  - and either `last_action_safe_mismatch_rate >= 0.5` or top signatures containing explicit safe-action misses
- Run a conservative micro-resume from the incumbent's `best_eval_resume.pt` with:
  - the near-perfect endgame curriculum window
  - ultra-late `aux_safe_action_target`
  - small `safe_action_bonus`
  - small `late_confidence_coef`
- Goal: push the final `393-396` action choices toward the safe branch without perturbing the earlier game.
- This now has priority over the older generic `resume_endgame_micro` probe.

### Research Arm B2: Hybrid Endgame Combo Micro

- If an ultra-late body-age micro branch reaches the healthy high-frontier regime:
  - `mean_score >= 380`
  - `phase_lt20_rate == 0.0`
- Immediately follow it with a second ultra-late micro-resume that combines:
  - `aux_body_age_target`
  - `aux_safe_action_soft_target`
  - `safe_action_bonus`
  - `late_confidence`
- Goal: unify the two partially useful late-game signals instead of treating them as separate fixes.
- This should be preferred over another plain control seed once the triggering body-age branch exists.

### Research Arm B3: Endgame Elite-BC Micro

- If a targeted endgame branch produces many train-time wins or near-perfect trajectories but still underperforms greedily:
  - keep the same ultra-late curriculum and auxiliary setup
  - add a small `elite_bc` replay loss
  - save only trajectories with terminal score `>= 390` and fill `>= 0.98`
- Goal: convert sampled last-move success into the default greedy policy instead of relying on heuristics alone.

### Research Arm B4: Endgame Sharpen Micro

- If the elite-BC branch is weak or clearly underpowered, fall back to the cleanest possible endgame calibration probe:
  - ultra-late curriculum
  - no extra replay
  - no extra teacher heads
  - lower entropy
  - stronger late-confidence penalty
- Goal: isolate whether the frontier already contains the right behavior and just needs sharper greedy preferences.

### Research Arm C: Dense Harvest

- Rerun only an exceptional control seed with:
  - `eval_every_steps=65536`
  - `checkpoint_interval=1`
- Goal: catch transient improvements that the coarse first-gate screen may miss.
- Current status: low priority after multiple negatives. Only use on controls that are effectively near-incumbent.

### Research Arm D: Conservative Resume

- Resume from `best_eval_resume.pt` with:
  - `resume_add_steps=true`
  - `override_resume_lr=true`
  - `lr=1e-6`
  - `timesteps=131072`
  - `eval_every_steps=32768`
  - `checkpoint_interval=1`
- Goal: test whether a good basin can be safely extended at all.
- If a first resume lands in the healthy mid-frontier regime, allow one more full conservative resume before switching to a micro-resume.
- Specifically:
  - `365 <= mean_score < 380` with `lt20 == 0`: allow one second `resume_short`
  - `mean_score >= 380` with `lt20 == 0`: switch to **micro-resume** only

### Research Arm E: Micro Resume

- Resume from `best_eval_resume.pt` with the same conservative settings, but a shorter added horizon:
  - `timesteps=65536`
  - `eval_every_steps=16384`
  - `checkpoint_interval=1`
- Goal: test whether the frontier can be nudged upward without overshooting the stable basin.
- For exceptional fresh controls, this is now the preferred first continuation probe.

### Research Arm E2: Nano Resume

- For a near-frontier micro basin:
  - `mean_score >= 392`
  - `phase_lt20_rate == 0.0`
- Run a smaller continuation from `best_eval_resume.pt`:
  - `timesteps=32768`
  - `eval_every_steps=8192`
  - `lr=5e-7`
  - `checkpoint_interval=1`
- Goal: test whether the remaining `393-396` mistakes can be corrected with a smaller PPO step than `resume_micro`, avoiding the overshoot seen on the third micro continuation.
- This has priority over additional fresh controls when the frontier is already in the near-perfect regime.

### Research Arm E3: History-2 Short Resume

- Retired.
- Two direct probes from the frontier were clearly negative and this family should not be auto-enqueued again unless a new hypothesis specifically reopens it.

### Research Arm F: Cold-Streak Resume Revisit

- If the controller sees a sustained cold streak of weak controls, it should enqueue a conservative resume from the best historical `resume_short` result that:
  - has `lt20 == 0`
  - scores at least `365`
  - is within `5` mean-score points of the current incumbent
  - is not already the incumbent
- Goal: keep exploiting the only research arm that has shown near-frontier behavior instead of burning the full budget on cold fresh seeds.
- For near-perfect incumbents (`390+` mean, `lt20=0`), do not replay the exact same continuation seed from that historical resume. Use a fresh continuation seed against the same `best_eval_resume.pt` so the branch measures local variance instead of mere replayability.
- Updated PI refinement: if that strong historical resume is itself a child of an earlier resume state, prefer varying the **upstream parent resume state** first. In practice, test fresh first-hop variance before chaining deeper from a weak child branch.

### Research Arm G: Parent-Seed Resume Probe

- If the control arm stays cold and there is no qualifying near-frontier resume candidate within the normal threshold, fall back to the historical parent resume state that led to the current incumbent.
- Run a conservative resume from that parent's `best_eval_resume.pt`, but use a **fresh continuation seed**.
- Cap this probe family to a small number of attempts per parent source.
- Goal: test whether the fertile parent state is still useful when continuation randomness changes, without repeatedly re-running the exact same branch.
- If one of these fresh first-hop parent probes is strong but still below the frontier, prefer another **fresh first-hop seed from the same parent** over an immediate deeper same-seed chain.
- Rationale: the first fresh-seed parent probe stayed healthy, while the immediate same-seed second hop regressed.
- Once a run is already in the `parent_seed_sweep_followup` family, do **not** keep chaining deeper on the same seed. Treat these as first-hop probes only.

### Research Arm H: Incumbent Micro Seed Probe

- If the control arm is cold, the near-frontier resume revisit is unavailable, and the parent-seed sweep is exhausted or unconvincing, enqueue a fresh-seeded `resume_micro` directly from the incumbent's `best_eval_resume.pt`.
- Cap this family to a small number of attempts per incumbent.
- Goal: test whether the incumbent can be improved with tiny low-LR continuation steps once seed variance is allowed to change.
- Current status: still more promising than the first body-age probe and should be retried before spending many more cycles on cold controls.

### Research Arm I: Body-Age Micro Resume

- If the control arm is still cold after the incumbent micro family has been sampled, enqueue a late-game `resume_micro` from the incumbent with a new auxiliary temporal-occupancy target:
  - `--aux-body-age-target`
  - `--aux-body-age-target-coef 0.25`
  - `--aux-body-age-target-min-fill 0.80`
- Cap this family to a small number of attempts per incumbent.
- Goal: teach the encoder which blocked cells are about to open so late-game continuation can stabilize without changing inference-time purity.
- Current status: first probe underperformed the plain incumbent micro branch, so this family is de-emphasized after one miss.

## Control Recipe

- `board_size=20`
- `timesteps=300000`
- `num_envs=64`
- `horizon=256`
- `minibatch_size=4096`
- `device=cpu`
- `gamma=0.999`
- `gae_lambda=0.9`
- `vf_clip_coef=1.0`
- `network_scale=2`
- `flood_fill=true`
- `aux_flood_fill=true`
- `head_centered=true`
- `curriculum_prob=0.1`
- `curriculum_min_fill=0.9`
- `curriculum_max_fill=0.98`
- `curriculum_follow_bonus=0.005`
- `curriculum_follow_min_fill=0.95`
- `lr=5e-6`
- `no_anneal_lr=true`
- `eval_every_steps=250000`
- `eval_deterministic=true`
- `eval_episodes=20`
- `resume=experiments/exp074_multi_path_curriculum_ft_177317600936/best_eval.pt`

## Promotion Rules

### Incumbent update

- Higher `win_rate` wins
- then higher `mean_score`
- then higher `phase_gte95_rate`
- then lower `phase_lt20_rate`

### Control-triggered follow-ups

- If a control run gets `mean_score >= 372` and `phase_lt20_rate == 0.0`
  - enqueue a direct **micro-resume** from its `best_eval_resume.pt`

- If a control run gets `mean_score >= 379` and `phase_lt20_rate == 0.0`
  - enqueue a dense harvest rerun of the same seed

- If a control run gets `360 <= mean_score < 372`
  - do **not** enqueue a continuation probe
  - treat it as informative control evidence only
  - rationale: the latest high-360s control continuation (`212 -> 213`) regressed badly, so this band is not strong enough to justify spending continuation budget

- If a harvest run gets `mean_score >= 370`
  - enqueue a conservative resume probe from its `best_eval_resume.pt`

- If a first `resume_short` gets `365 <= mean_score < 380` and `phase_lt20_rate == 0.0`
  - enqueue one second `resume_short` from that new `best_eval_resume.pt`

- If a `resume_short` gets `mean_score >= 380` and `phase_lt20_rate == 0.0`
  - enqueue one **micro-resume** from that new `best_eval_resume.pt`

- If a second-hop `resume_short` stays below `380`
  - do **not** auto-spawn a `resume_micro`
  - rationale: the strong micro-to-short branch improved once, but the deeper sub-380 short basin did not support a useful micro follow-up

- If a `resume_micro` gets `mean_score >= 380` and `phase_lt20_rate == 0.0`
  - enqueue one second `resume_micro` from that new `best_eval_resume.pt`
  - rationale: first-hop micro resumes in the high-370s are still not reliably improving the frontier, so only near-incumbent first hops justify a second chained nudge

- Do not auto-extend `resume_micro` beyond the second micro hop
  - rationale: the forced third micro continuation from the `392.35` basin regressed sharply, so the useful chain currently appears to stop at two micro hops

- If a new incumbent is near-perfect (`mean >= 390`, `lt20 == 0`, `95%+ == 100%`, `win_rate == 0`)
  - enqueue one `failure_harvest` analysis run on that incumbent before any new targeted endgame probe
  - record motif counts and safe-action mismatch rates in the loop state/journal
  - rationale: the remaining gap is small enough that the controller should measure the exact failure manifold first, not guess

- If a completed failure harvest on the incumbent shows dominant `self` failures and strong safe-action mismatch signal
  - enqueue one `resume_endgame_safe_micro` immediately from that incumbent
  - if the soft-safe branch lands in the high `370s` or better without reintroducing early collapse, allow one direct `resume_endgame_combo_micro` follow-up
  - rationale: the current frontier fails in the final few moves, and the harvest already tells us those moves are often action-choice errors, not missing early-game competence

- If `5` consecutive control runs score below `350`
  - enqueue a conservative resume from the best historical qualifying `resume_short` source
  - if a historical alt-seed resume source has gone cold, suppress it and fall through to the next-best family
  - define "gone cold" as at least `2` of the last `3` alt-seed revisits scoring below `370` or reintroducing clearly nontrivial early collapse (`lt20 > 0.05`)
  - rationale: once a near-frontier resume source starts failing repeatedly under fresh continuation seeds, the loop should stop treating it as the default fallback

- If the cold streak reaches `10` and there is still no qualifying near-frontier resume candidate
  - first, if the incumbent itself came from a `resume_micro` parent with `mean_score >= 390` and `lt20 == 0`, enqueue one fresh-seeded `resume_micro` directly from that parent
  - cap these parent-micro probes to `1` attempt per parent source
  - rationale: the `143 -> 144 -> 145` chain that produced the frontier left one worthwhile unexplored variance branch, but the first clean revisit from `144` underperformed badly enough that it should not be repeated
  - prefer a conservative resume from the strongest fresh `resume_micro` basin using a fresh continuation seed
  - only fall back to the incumbent's historical parent `best_eval_resume.pt` if that parent source is not already suppressed by the failed micro-to-short revisit test
  - cap these parent-seed probes to `3` attempts per parent source
  - if the parent-seed probes are exhausted, enqueue a fresh-seeded `resume_micro` from the incumbent itself
  - if the incumbent micro probes are exhausted, enqueue a fresh-seeded `resume_micro_body_age` from the incumbent
  - if those local-variance probes are exhausted and there is still no qualifying historical resume candidate, enqueue one fresh-seeded `resume_micro` from the strongest recent control basin with `mean >= 372` and `lt20 == 0`
  - cap these control-alt-micro probes to `1` attempt per control source
  - rationale: the loop had gotten stuck doing endless cold controls because the historical resume families were exhausted; a strong control source deserves one fresh-seeded local-variance probe before the controller gives up to more controls
  - updated PI read: this fallback has now failed on three `373-375` class historical control sources, so it should only trigger for truly exceptional controls (`mean >= 380`, `lt20 == 0`) rather than generic strong controls
  - rationale: the family is not dead in principle, but the current evidence says mid-`370s` control basins do not survive a fresh-seeded micro continuation well enough to justify budget
  - if there is still no qualifying resume candidate and no `380+` control source for a fresh-seeded micro, allow one fresh-seeded `resume_nano` from the strongest historical control basin with `mean >= 380` and `lt20 == 0`
  - cap these control-alt-nano probes to `1` attempt per control source
  - updated PI read: the nano family also failed on the tested `373-375` historical control basins, so it should now be reserved for genuinely exceptional control sources rather than generic strong controls
  - rationale: `micro` and `nano` both look too destructive on the mid-`370s` historical controls; the family is only still justified if a future control opens much closer to the frontier
  - if the local-variance probes and strong-control variance probes are exhausted, enqueue one fresh-seeded continuation from the strongest historical endgame-targeted basin (`endbody`, `endmix`, or `endsafe`) only if it is truly near-frontier (`mean >= 384`, `lt20 == 0`, `95%+ >= 80%`), preserving the basin's matching family rather than downgrading it to a plain micro
  - cap these historical-endgame-basin micro probes to `1` attempt per source
  - updated PI read: the two strongest historical endgame-basin repros (`170_endbody`, `172_endmix`) both came back materially below the frontier on fresh seeds, so lower-scoring variants like `endsafe` are no longer worth automatic fallback budget
  - rationale: the family is not producing frontier-threatening replays, and stronger members already failed the test
- If the control streak reaches `5` and there is no qualifying near-frontier resume candidate
  - prefer the remaining incumbent micro probe before continuing a long run of cold controls
  - if the incumbent micro probes are exhausted, prefer one conservative `resume_short` from the strongest historical `resume_micro` basin with `mean_score >= 375` and `lt20 == 0`
  - rationale: a good micro basin may want a slightly longer conservative continuation, not another micro hop
  - if the generic resume fallback is suppressed as cold and there is a historical `endbody` basin with `mean_score >= 383` and `lt20 == 0`, enqueue one plain conservative `resume_short` from that basin's `best_eval_resume.pt`
  - cap this historical `endbody` fallback to one attempt per source basin
  - rationale: the best distinct late-game basin is worth one clean plain continuation before giving up on continuation research entirely
  - if a post-suppression alt-seed `resume_short` still comes back weak (`mean < 340` or `lt20 > 0`), escalate immediately to that historical `endbody` fallback instead of returning to controls first
  - rationale: once both the old cold source and the next-best self-source alt-seed fail, the controller should switch families immediately rather than burn another control cycle
  - if the incumbent is already near-perfect (`mean >= 391`, `lt20 == 0`, `95%+ == 100%`, `win_rate == 0`), prefer one targeted `resume_endgame_micro`
  - but only after a completed `failure_harvest` on that incumbent
  - use tighter late-game curriculum pressure:
    - `curriculum_prob=0.2`
    - `curriculum_min_fill=0.96`
    - `curriculum_max_fill=0.995`
    - `curriculum_follow_bonus=0.01`
    - `curriculum_follow_min_fill=0.97`
  - rationale: once every eval episode reaches the final 5%, the remaining gap is likely ultra-late corridor timing, not general policy quality
  - current evidence: one forced probe from the `392.35` basin produced many sampled train wins but only `373.55` greedy eval, so do not repeat this exact probe on the same incumbent

- If a fresh parent-seed resume probe gets `mean_score >= 372` with `lt20 == 0` but remains below the rare-exceptional harvest threshold
  - enqueue another fresh-seeded first-hop parent probe from the same parent source
  - do not prioritize a deeper same-seed chain first

### Baseline interpretation

- `mean_score >= 370`: control arm is still alive
- `360 <= mean_score < 372` with `lt20 == 0`: interesting, worth a conservative resume
- `mean_score >= 372` with `lt20 == 0`: exceptional, worth a direct micro-resume
- `mean_score >= 379` with `lt20 == 0`: rare exceptional, worth harvest plus micro-resume
- `mean_score >= 390` with `lt20 == 0` and `95%+ == 100%`: pause blind endgame tuning and harvest failures first
- sustained controls below `350`: control arm is cold, revisit the best historical resume source
- sustained controls below `350` with no qualifying resume source: probe fresh continuation seeds from the fertile incumbent parent
- sustained controls below `350` after incumbent micro exhaustion: prefer the late-game `safe -> combo -> body-age` fallback order
- otherwise: discard as non-frontier

## Stop Conditions

The loop should keep running unless:

- `experiments/mission_loop.stop` exists
- disk free space falls below the configured floor
- the interpreter or training script becomes unavailable

## Disk Guardrails

After each completed run:

- keep `run.json`, `metrics.jsonl`, `summary.json`
- keep `best_eval.pt`, `best_eval_resume.pt`, `first_train_win.pt`, `latest_train_win.pt`
- keep the root final checkpoint `<run_id>.pt`
- delete intermediate `model_*_NNNNNN.pt` snapshots after the run is summarized

## Journal Expectations

The loop must write:

- `experiments/mission_loop_state.json`
- `experiments/mission_loop_journal.jsonl`
- `experiments/mission_loop.log`

Every run should log:

- launch
- family/spec
- completion
- selector stats
- follow-up scheduling decisions
- incumbent updates
- cleanup actions
