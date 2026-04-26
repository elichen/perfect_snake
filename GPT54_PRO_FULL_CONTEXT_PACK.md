# GPT-5.4 Pro Full Context Pack

Use this as the prompt/context pack for `gpt-5.4-pro`.

Suggested usage:
- Paste this entire file into the model.
- If you can attach files, also attach `EXPERIMENTS.md`, `train.py`, `snake_env.py`, `distill/train_bc.py`, `distill/train_bc_rnn.py`, `distill/conditioning.py`, and `distill/expert.py`.
- Ask the model to give you a concrete experiment plan, not code.

---

## Prompt

You are my ML research lead. I want your best experiment recommendations to push a Snake agent to **100% deterministic win rate** on `20x20`.

Use deep reasoning. Take your time. Do not optimize for brevity. I do **not** want generic RL advice. I want concrete, high-value, research-level experiment proposals grounded in the exact evidence below.

### Hard constraints

- Final inference must remain a **pure neural network policy**.
- No inference-time search, planner, fallback controller, or expert handoff.
- Training-time expert signals, curricula, auxiliary targets, imitation/distillation, staged training, and phase-specific heads are allowed.
- Optimize for **information gain per wall-clock time** on a MacBook Pro.
- Be decisive. If a line is saturated, say so explicitly.
- If I should stop doing something, tell me clearly.

### Goal

Achieve **100% deterministic win rate in actual play** on `20x20` Snake.

Success means:
- greedy/deterministic evaluation or play
- not stochastic training rollouts
- not "the policy can occasionally sample a perfect game"

### What I want from you

Please give me:

1. A ranked list of the next `5-8` experiments to run.
2. Your view of the **highest-probability path to mastery**.
3. A **decision tree** for experiment sequencing.
4. Direct answers to the open questions listed below.
5. Strong opinions on what is saturated, what is underrated, and what to stop doing.

For each proposed experiment, include:

- short name
- branch: PPO, distillation, recurrent distillation, or staged hybrid
- exact hypothesis
- exact change relative to the current best baseline
- why it is high-value now
- expected result if the hypothesis is right
- what would falsify it
- kill criteria / early-stop criteria
- what to do next if it succeeds
- what to do next if it fails

I prefer a small number of strong recommendations over a giant menu.

---

## Environment and Eval Contract

- Board size: `20x20`
- Perfect score: `397`
- Deterministic eval is the metric that counts
- Training-time wins do not count unless the saved checkpoint wins deterministically

Helpful practical constraint:
- PPO CPU runs around `~4k SPS` in the smaller 64-env regime
- Distillation runs can be slow because deterministic eval episodes become long once policies get decent

Optimize for **wall-clock efficiency**, not just theoretical appeal.

---

## Repo / Code Map

Main PPO path:
- `train.py`
- `snake_env.py`
- `eval.py`
- `experiment_tracker.py`

Standalone expert-distillation path:
- `distill/train_bc.py`
- `distill/evaluate.py`
- `distill/train_bc_rnn.py`
- `distill/evaluate_rnn.py`
- `distill/model.py`
- `distill/rnn_model.py`
- `distill/conditioning.py`
- `distill/expert.py`

Useful logs / notes:
- `EXPERIMENTS.md`
- `experiments/index.jsonl`

Expert:
- There is a perfect Hamiltonian-cycle expert available for training-time supervision only
- It achieves 100% wins in the real env

---

## PPO Branch: What Happened

### Main result

The strongest deterministic PPO eval on `20x20` was:

- `exp072_head_centered`
- best deterministic eval: `374.7 / 397`
- deterministic `win_rate = 0.0`

This is the strongest PPO eval result we ever got on `20x20`.

### Best PPO recipe

The main PPO recipe that worked best was:

- 2x MLP
- head-centered observation
- flood-fill observation
- auxiliary flood-fill decoder
- symmetric augmentation
- gamma `0.999`
- horizon `256`
- curriculum spawning

### Important PPO details

- `exp072` was the big breakthrough run
- `exp078_bonus_0p005` later got back to `370.6 / 397` deterministic eval, still `win_rate = 0.0`
- later PPO branches produced actual perfect **training** games, but not deterministic eval wins

### Critical PPO fact

On `20x20`, PPO **never** recorded a nonzero deterministic eval win rate.

What PPO did manage:

- it produced exact `397` games during training/stochastic sampling
- it repeatedly produced `train_win_checkpoint`s in later curriculum/cycle branches

What PPO did **not** manage:

- turn those sampled wins into repeated deterministic wins from a saved checkpoint

### PPO failure pattern

Earlier PPO frontier:

- early blunders before `~20%` fill
- late self-traps / corridor errors at high fill

In other words, PPO could get very close to perfect, but never made perfect play the default greedy policy.

### Key PPO artifacts

- `exp072_head_centered`
- `exp078_bonus_0p005`
- `exp076_cycle_bonus_endgame_cpu`
- `exp079_bonus_0p005_capture`

Interpretation:

- PPO reached a regime where the policy could **sample** winning trajectories
- PPO never consolidated those into reliable deterministic play

---

## Distillation Branch: What Happened

### Why this branch exists

The PPO line showed that:

- a pure NN can get very close
- the policy can sometimes sample winning behavior
- but PPO struggles to make that winning behavior the default deterministic mode

So the distillation branch uses a perfect expert **during training only**, while keeping the final model pure NN.

### Expert

The Hamiltonian-cycle expert is perfect:

- 100% wins in the real env
- training-only
- no inference-time expert allowed in the final solution

### Major distillation discovery

Unconditioned imitation failed because of real action aliasing:

- the same visible snake state can match multiple Hamiltonian cycles
- those matching cycles can require different next actions

This means naïve BC on the raw observation is ill-posed.

### What fixed that partially

I added **cycle conditioning** to the standalone distillation path.

That made the distillation line viable.

### Current best distillation result

Best deterministic distillation checkpoint so far:

- run: `manual_distill_205_cyclecond_rollin0p1_s60_1773429536`
- deterministic eval: `67.85 / 397`
- deterministic `win_rate = 0.0`

This is the current best pure-NN checkpoint from the expert-distillation track.

### Distillation failure pattern right now

The current best distillation line is still mostly an **early-game robustness** failure:

- about `70%` of deterministic episodes die below `20%` fill
- about `30%` reach `20-80%` fill
- almost none reach true late game
- deaths are mostly `stall`, then `wall`

That means this branch is still mostly failing before the hard late-game Hamiltonian structure even matters.

### Distillation experiments already tried

Tried and measured:

- plain cycle-conditioned BC from scratch
- low-LR continuation from the best BC checkpoint
- fixed policy roll-in
- annealed roll-in
- fill-gated roll-in
- DAgger-style replay mixing policy-visited states into training

### Distillation quantitative frontier

Representative deterministic `20`-episode results:

- `manual_distill_202_cyclecond_full_long_s60_1773425190`: `45.30`
- `manual_distill_203_cyclecond_resume_lowlr_s60_1773426396`: `65.30`
- `manual_distill_205_cyclecond_rollin0p1_s60_1773429536`: `67.85`
- `manual_distill_206_cyclecond_rollin0p05_s60_1773431296`: `65.30`
- `manual_distill_210_dagger0p25_buf1024_pr0p5_rollin0p1_maxfill0p2_s60_1773438156`: `65.20`

Interpretation:

- cycle-conditioned MLP distillation is real
- small roll-in helps a bit
- DAgger-style replay in this MLP basin did **not** beat the incumbent

### Current reading of the distillation basin

The MLP distillation basin appears close to saturated:

- it can clearly learn something useful
- it improves from near-zero to mid-60s deterministic mean
- but the various roll-in / replay variants have not broken past that plateau in real evaluation

---

## Recurrent Distillation Status

There is a standalone recurrent distillation path:

- `distill/train_bc_rnn.py`
- `distill/evaluate_rnn.py`
- `distill/rnn_model.py`

Important caveat:

- earlier RNN distillation experiments failed **before** cycle conditioning existed
- the current recurrent path does **not** yet contain the newer cycle-conditioning / roll-in / replay ideas

That means the old negative result on recurrent distillation may no longer be fully trustworthy.

---

## Constraints About Code Organization

- I prefer keeping distillation code separate from PPO code
- Shared environment/model-definition utilities are fine
- I do **not** want the distillation path tangled into the PPO trainer unless there is a very strong reason

---

## Key Negative Results To Factor In

Please treat these as real evidence, unless you explicitly argue why the context changed enough that they should be revisited:

- unconditioned feedforward BC failed badly
- unconditioned recurrent BC failed badly
- PPO on `20x20` never got deterministic eval wins
- more PPO scalar sweeps alone did not convert sampled wins into deterministic wins
- the current MLP distillation basin seems to plateau around the high-60s

Possible context-changed negatives:

- recurrent distillation may be worth revisiting **with cycle conditioning**
- some older distillation negatives may have been caused by aliasing rather than lack of model capacity

---

## Working Hypotheses

These are my current beliefs. Please challenge them if you think they are wrong.

1. The current MLP distillation basin is near saturation.
2. The next justified pivot is likely **cycle-conditioned recurrent distillation**.
3. PPO probably only becomes worth reopening after obtaining a much stronger distilled policy.
4. The remaining problem is mostly **recovery from small deviations**, not endgame planning.
5. The project may need stronger sequence modeling or stronger supervised targets, not more scalar tuning.

---

## Open Questions I Want You To Answer Directly

1. Is cycle-conditioned recurrent distillation the next real move, or likely another dead end?
2. Is the current MLP distillation basin saturated?
3. Should I continue PPO now, or only after getting a stronger distilled policy?
4. What is the best way to expose the NN to recovery states without inference-time search?
5. Should I add multi-step targets, phase-specific heads, sequence modeling, or something else next?
6. Which past negative results should still be trusted, and which should be revisited because the context changed?

---

## What I Want You To Optimize For

Optimize for:

- probability of reaching true mastery
- wall-clock efficiency
- clarity of experiment sequencing
- avoiding dead-end sweeps

Do **not** optimize for:

- elegance
- novelty for its own sake
- giant architecture churn
- advice like “just use more compute”

---

## Deliverable Format

Please answer in this structure:

### 1. Executive View
- your single best path forward
- your fallback path
- what to stop doing

### 2. Ranked Experiment List
For each experiment:
- name
- branch
- exact hypothesis
- exact changes
- expected upside
- likely failure mode
- kill criteria
- next action if it works
- next action if it fails

### 3. Decision Tree
- if experiment A does X, run B
- if experiment A fails with Y, skip C and run D

### 4. Opinionated Takes
- what I’m underrating
- what I’m overrating
- what is most likely to convert “sampled good play” into “greedy perfect play”

### 5. Recommended Next Experiment
End with:
- one exact next experiment
- exact config or pseudoconfig
- why it should be first
- what metric threshold makes you continue vs kill it

---

## Helpful Identifiers

You can refer to these runs by name:

PPO:
- `exp072_head_centered`
- `exp078_bonus_0p005`
- `exp076_cycle_bonus_endgame_cpu`
- `exp079_bonus_0p005_capture`

Distillation:
- `manual_distill_202_cyclecond_full_long_s60_1773425190`
- `manual_distill_203_cyclecond_resume_lowlr_s60_1773426396`
- `manual_distill_205_cyclecond_rollin0p1_s60_1773429536`
- `manual_distill_206_cyclecond_rollin0p05_s60_1773431296`
- `manual_distill_210_dagger0p25_buf1024_pr0p5_rollin0p1_maxfill0p2_s60_1773438156`

Current incumbent:
- `manual_distill_205_cyclecond_rollin0p1_s60_1773429536`

---

## Final Reminder

I do not want generic advice.

I want the best experiment plan you can produce given:

- PPO got very close but never deterministic-perfect
- a perfect expert exists
- raw BC was ill-posed due to aliasing
- cycle conditioning partially fixed that
- the MLP distillation line now appears to plateau in the high-60s
- the next move needs to be high-value, not just “another sweep”

