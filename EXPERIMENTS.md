# Experiment Log

**Goal:** 100% win rate on 20x20 Snake (perfect score = 397)

---

## Phase 1: World-Centric Observation (Failed)

Initial experiments used 9-channel world-centric observation with direction encoded as separate channels.

### exp001 - Baseline (World-Centric)
**Config:** board=10, obs=world (9ch), lr=3e-4, seed=42, 50M steps
**Result:** 0% win rate, score ~70-76

### exp002 - 2x Network (World-Centric)
**Config:** Same as exp001, doubled network widths
**Result:** 0% win rate, score ~50 (worse)

### exp003 - Lower LR (World-Centric)
**Config:** 2x network, lr=1e-4
**Result:** 0% win rate, score ~24 (worse)

### exp004 - Symmetric Augmentation (World-Centric)
**Config:** baseline + 50% horizontal flip per episode
**Result:** 0% win rate, score ~74-82 (no improvement)

**Conclusion:** World-centric observation failed to achieve wins.

---

## Phase 2: Egocentric Observation (Success)

Switched to 5-channel egocentric observation where grid is rotated so snake always faces "up".

### exp005 - Egocentric Only
**Config:** board=10, obs=egocentric (5ch), 10M steps
**Result:** 0% win rate, score ~74.8 (eval deterministic)

### exp006 - Egocentric + Symmetric
**Config:** board=10, egocentric + 50% horizontal flip, 10M steps
**Result:** 67% win rate (eval deterministic), score ~86.2

### exp007 - Egocentric + Symmetric (Long Run)
**Config:** board=10, egocentric + symmetric, 26M steps
**Result:** 100% win rate (training), score 97 (perfect)

**Conclusion:** Egocentric + symmetric augmentation achieves perfect play on 10x10.

---

### exp010 - 10x10 Rerun (Post-Cleanup)
**Config:** board=10, egocentric + symmetric, network-scale=1, 50M steps, seed=42
**Result:** 0% win rate (eval deterministic), score ~81.3

**Note:** Another 0% run despite same config as exp006/007. Confirms high variance.

---

## Phase 3: Scaling to 20x20 (In Progress)

### exp011 - 20x20 with 1x Network
**Config:** board=20, egocentric + symmetric, network-scale=1, 100M steps
**Status:** TODO

### exp012 - 20x20 with 2x Network
**Config:** board=20, egocentric + symmetric, network-scale=2, horizon=128, 100M steps, seed=42
**Result:** 0% win rate, eval score 155.6/397 (39%), ~20k SPS

**Progression:**
- 50M steps: score ~51 (13%)
- 75M steps: score ~63 (16%)
- 100M steps: score ~156 (39%)

**Note:** Learning is happening but slow. 39% of perfect at 100M steps suggests need for longer training or architectural changes.

### exp013 - 20x20 with Horizon=512, LR=1e-4
**Config:** board=20, egocentric + symmetric, network-scale=2, horizon=512, lr=1e-4, 82M steps (stopped early)
**Result:** 0% win rate, eval score 63.2/397 (16%), ~18k SPS

**Comparison with exp012 at similar steps:**
- exp012 @ 75M: score 63 (16%)
- exp013 @ 82M: score 63 (16%)

**Conclusion:** Longer horizon + lower LR did NOT help. Same performance with more steps. The horizon=128 with lr=3e-4 was actually more efficient.

---

## Phase 4: GRPO Algorithm Comparison (Failed)

Tested GRPO (Group Relative Policy Optimization) from DeepSeekMath as an alternative to PPO. GRPO eliminates the value network by computing advantages from group statistics.

### exp_grpo_episode - GRPO with Episode-Level Credit
**Config:** board=10, GRPO, credit=episode, num-envs=256, min-episodes=64, symmetric
**Result @ 26M steps:** 0% win rate, eval score 3.68/97 (4%)

### exp_grpo_step - GRPO with Step-Level Credit
**Config:** board=10, GRPO, credit=step (return-to-go), num-envs=256, min-episodes=64, symmetric
**Result @ 9M steps:** 0% win rate, eval score 0.22/97 (<1%)

### PPO Baseline (concurrent run)
**Config:** board=10, PPO, num-envs=256, horizon=128, symmetric
**Result @ 5M steps:** 0% win rate, eval score 44.42/97 (46%)

**Sample Efficiency Comparison:**

| Algorithm | Steps | Eval Score | Relative Efficiency |
|-----------|-------|------------|---------------------|
| PPO | 5M | 44.42/97 | 1x (baseline) |
| GRPO episode | 26M | 3.68/97 | ~60x worse |
| GRPO step | 9M | 0.22/97 | ~400x worse |

**Why GRPO Failed:**

1. **No value baseline** - GRPO uses batch mean as baseline (state-independent), while PPO's V(s) learns state-dependent expected returns
2. **Poor credit assignment** - Episode-level: all actions get same advantage. Step-level: return-to-go correlates with episode length, not action quality
3. **Reward density mismatch** - GRPO designed for sparse outcome rewards (LLMs). Snake has dense per-step rewards that PPO exploits via GAE
4. **Long episodes** - Snake episodes are 50-2000+ steps. One bad move kills you but GRPO blames all actions equally

**Conclusion:** GRPO is architecturally mismatched for dense-reward sequential MDPs. The value function in PPO isn't optional for temporal credit assignment - it's essential. GRPO's "critic-free" design, beneficial for LLMs, becomes a major liability for game-playing agents.

---

## Phase 5: Breaking the 40% Plateau (In Progress)

Both exp012 and exp013 plateau at ~39% (score ~155/397). Investigating why.

### exp022 - Stall Handling Fix
**Hypothesis:** Stall penalty (-0.5) and truncation (not termination) underpenalizes stalling. Agent learns to wander instead of pursuing food aggressively.

**Changes:**
- `stall_penalty`: -0.5 → -1.0 (same as death)
- `stall_terminates`: True (terminated, not truncated - PPO won't bootstrap)

**Config:** board=20, network-scale=2, symmetric, 100M steps
**Result:** 0% win rate, eval score **154.5/397 (39%)**

**Eval progression:**
| Steps | Score | % |
|-------|-------|---|
| 10M | 42.8 | 11% |
| 20M | 40.7 | 10% |
| 30M | 63.0 | 16% |
| 40M | 81.7 | 21% |
| 50M | 56.1 | 14% |
| 60M | 119.8 | 30% |
| 70M | 132.9 | 33% |
| 80M | 69.0 | 17% |
| 90M | 154.5 | 39% |

**Conclusion:** Stall fix did NOT break the plateau. Same ~39% result as exp012. High eval variance (10 episodes too few).

### exp023 - Alpha Decay
**Hypothesis:** Distance shaping (alpha=0.2) hurts late-game. Optimal paths often require moving AWAY from food first (to navigate around body). Constant shaping fights this.

**Changes:**
- Alpha now decays with snake length: `alpha_eff = alpha * (1 - length/board_area)`
- At length 3: α=0.199 (full shaping)
- At length 160 (40%): α=0.12 (reduced)
- At length 300 (75%): α=0.05 (minimal)

**Config:** board=20, network-scale=2, symmetric, 100M steps
**Result:** 0% win rate, eval score **~139/397 (35%)** - WORSE than baseline

**Conclusion:** Alpha decay made it worse (35% vs 39%). Reverted.

### exp024/exp028 - Tail Channel
**Hypothesis:** Adding 6th observation channel showing tail position helps agent see where space opens up.

**Result:** Hurt performance on both 20x20 (149 vs 154) and 10x10 (68% vs 100% win rate at 20M steps).

**Conclusion:** Extra channel adds noise without useful signal. Reverted.

---

## Phase 6: Architecture Exploration

### exp030 - CNN on 10x10
**Hypothesis:** CNN's spatial inductive bias (translation invariance, local patterns) might learn faster than MLP.

**Config:** board=10, CNN (32→64→64 channels), 50M steps
**Result:** **100% win rate at 6M steps** (vs MLP at 40M) - 6x faster!

| Steps | CNN | MLP |
|-------|-----|-----|
| 6M | 100% win | 0% win |
| 20M | 80-100% | first win |
| 40M | oscillating | 100% |

**Conclusion:** CNN learns dramatically faster on 10x10. But oscillates between 70-100% instead of stable 100%.

### exp031 - CNN on 20x20 (2x scale)
**Config:** board=20, CNN 2x (64→128→128), 55M steps
**Result:** Peaked at 75.5 score @ 30M, then oscillated 50-75. Never approached MLP's 154.

**Conclusion:** CNN's advantage doesn't scale to 20x20. Lower ceiling than MLP.

### exp033-044 - Directional Architectures
Tested architectures that align observation structure with action space (3 directions → 3 actions).

| Exp | Architecture | Params | Peak Score | Notes |
|-----|--------------|--------|------------|-------|
| exp033 | Ray-casting | 5K | 0.0 | Dead - too simple, lost spatial context |
| exp034 | 3-branch CNN | 62K | 6.6 @ 7M | Slow learner |
| exp035 | Attention LR 3e-4 | 61K | 38.9 @ 7M | Collapsed at 8M |
| exp039 | Attention LR 1e-4 | 61K | 0.0 | Too slow to learn |
| exp040 | Attention LR 2e-4 | 61K | ~35 | Unstable oscillation |
| exp043 | Attention strided | 61K | 35.8 @ 14M | Oscillated 10-35 |

**Conclusion:** Directional architectures all failed. Either dead, slow, or unstable. MLP remains best.

### exp045/exp046 - MLP Scale Comparison
**Hypothesis:** Maybe MLP just needs more capacity (4x) or less (1x)?

| Scale | Params | Peak Score | Notes |
|-------|--------|------------|-------|
| 1x | 3.2M | 112.6 @ 40M | Fastest learner per step, then collapsed |
| 2x | 7.9M | 154.5 @ 90M | Best overall (baseline) |
| 4x | 21.5M | 83.2 @ 60M | Slower, oscillating 50-83 |

**Conclusion:** 2x is sweet spot. 1x learns fast but collapses. 4x might be overfitting.

---

## Phase 7: Training Stability & Finetuning

### exp047 - LR + Entropy Decay (seed 42)
**Hypothesis:** Decaying LR (min_lr_ratio=0.1) and entropy (0.02→0.002) reduces policy oscillation.

**Config:** board=20, network-scale=2, symmetric, 100M steps
**Result:** 0% win rate, eval score **127.4/397 (32%)** peak, collapsed to 56.7

**Conclusion:** Peak was decent but severe late collapse. LR decay destabilizes.

### exp048 - LR + Entropy Decay (seed 1)
**Config:** board=20, network-scale=2, min_lr_ratio=0.05, entropy 0.02→0.001, 100M steps
**Result:** 0% win rate, eval score **49.3/397 (12%)** peak

**Conclusion:** Much worse with different seed. Approach unreliable.

### exp049 - Entropy Annealing Only (seed 7)
**Hypothesis:** Maybe just entropy decay (no LR decay) helps.

**Config:** board=20, network-scale=2, no_anneal_lr=true, entropy 0.02→0.001, 100M steps
**Result:** 0% win rate, eval score **103.7/397 (26%)** peak, last 58.1

**Conclusion:** Better than exp048 but still oscillates. Entropy-only insufficient.

### exp050 - Finetune from Checkpoint (seed 11)
**Hypothesis:** Start from exp016 checkpoint, finetune with low LR (5e-5), reduced epochs.

**Config:** board=20, network-scale=2, LR=5e-5, 60M steps
**Result:** 0% win rate, eval score **116.6/397 (29%)** peak, last 108.7

**Conclusion:** Most stable run — small variance. Finetuning with low LR works.

### exp051-054 - CNN with Coordinate Channels
Various CNN architectures with centered head, coordinate channels, different strides/pooling.

| Exp | Architecture | Peak Score | Notes |
|-----|-------------|------------|-------|
| exp051 | CNN no-stride | 92.0 | Underperforms MLP |
| exp052 | CNN pool=11 | failed | Broke immediately |
| exp053 | CNN stride=2, pool=3 | 80.7 | Stable but low ceiling |
| exp054 | CNN stride=2, LR=1e-4 | 75.0 | Lower LR hurt |

**Conclusion:** CNN variants all underperform MLP on 20x20. Not worth pursuing.

### exp055 - MLP Finetune + Lower Entropy (seed 11)
**Config:** Finetune from checkpoint, lower entropy (0.01→0.001), 16384 minibatch, 2 epochs
**Result:** 0% win rate, eval score **136.6/397 (34%)** peak, last 90.9

**Conclusion:** Better peak than exp050 but more variance.

### exp056 - MLP Finetune Ultra-Conservative (seed 11)
**Config:** Finetune from checkpoint, LR=2.5e-5, constant entropy=0.003, 1 epoch
**Result:** 0% win rate, eval score **162.2/397 (41%)** peak, last 124.2

**Conclusion:** **NEW BEST PEAK: 162.2** — ultra-conservative finetuning slightly beats baseline 154.5. Still high variance.

---

## Phase 8: Flood-Fill Observation (Breakthrough)

Adding a 6th observation channel with flood-fill reachability broke the 40% ceiling. The channel marks which empty cells are reachable from the snake's head via connected-component analysis (scipy.ndimage.label).

### exp057/058/059 - Gamma/Alpha Experiments (No Flood-Fill)
Tested gamma=0.999 and alpha=0 without flood-fill.

| Exp | Config | Peak Score | Notes |
|-----|--------|------------|-------|
| exp057 | 10x10, alpha=0, gamma=0.999 | 79.2 | Agent can't find food without shaping |
| exp058 | 20x20, alpha=0, gamma=0.999 | 43.1 | Same — alpha=0 kills food-seeking |
| exp059 | 20x20, alpha=0.05, gamma=0.995 | 36.3 | Low alpha insufficient |
| exp060 | 20x20, gamma=0.995 | 139.9 | Slight improvement, high variance |

**Conclusion:** Gamma alone doesn't break plateau. Alpha=0 is fatal on 20x20.

### exp061 - Flood-Fill Observation Channel
**Hypothesis:** Agent needs to see which cells are reachable to avoid self-trapping.

**Changes:** Added 6th observation channel — flood-fill reachability from head using scipy connected components. ~3.6x env slowdown, ~10% training SPS impact.

**Config:** board=20, network-scale=2, 6-channel obs, gamma=0.995, 200M steps
**Result:** **216.4/397 (54.5%)** — up from 162.2 (41%)

**Bug found:** evaluate_policy() wasn't passing flood_fill_obs to eval env, causing MPS to hang on shape mismatch. Fixed by passing all env kwargs through.

**Resumed run:** Reached **249.7/397 (62.9%)** at extended training.

**Conclusion:** Flood-fill observation is a major breakthrough — biggest single improvement.

### exp063/064 - Weight-Tied Iterative CNN
**Hypothesis:** Iterative CNN with shared weights might learn flood-fill-like reasoning.

| Exp | Config | Peak Score | Notes |
|-----|--------|------------|-------|
| exp063 | Iterative CNN 1x, 12 iterations | ~60 | 3-4x worse than MLP |
| exp064 | Iterative CNN 2x, 12 iterations | ~80 | Still far below MLP |

**Conclusion:** Iterative CNN failed. MLP remains best architecture.

### exp065 - MLP + Aux Flood-Fill Decoder
**Hypothesis:** Adding auxiliary loss to predict flood-fill map improves feature learning.

**Changes:** Added flood-fill decoder head off shared features, trained with separate BCE loss after PPO step.

**Config:** board=20, network-scale=2, flood-fill obs, aux flood-fill decoder, 100M steps
**Result:** **239.8/397 (60.4%)**

**Conclusion:** Aux decoder helps slightly vs flood-fill obs alone.

### exp066 - MLP + Aux, 200M Steps with Cosine LR
**Config:** Same as exp065, 200M steps, cosine LR decay (3e-4 → 3e-5)
**Result:** **255.2/397 (64.3%)** at 190M steps

### exp067 - 1B Steps, Full Cosine
**Config:** 1B steps, cosine LR over full 1B
**Result:** Peaked at **172/397** — LR too high for too long, never recovered.

**Conclusion:** Cosine over 200M then constant min_lr is better than cosine over full run.

### exp068 - 1B Steps, Fast Cosine Decay
**Config:** 1B steps, cosine over 200M then constant min_lr=3e-5
**Result:** **264.2/397 (66.5%)** at 301M steps

---

## Phase 9: Extended Credit Assignment (Breakthrough)

### exp069 - Curriculum Spawning
**Hypothesis:** Training from late-game positions helps agent learn endgame patterns.

**Changes:** 30% of episode resets start at 50-85% fill using zigzag Hamiltonian path placement.

**Config:** board=20, network-scale=2, flood-fill, aux, curriculum_prob=0.3, 200M steps
**Result:** **258.4/397 (65.1%)** — same as without curriculum

**Conclusion:** Curriculum alone doesn't help much.

### exp070 - Curriculum + gamma=0.999 + horizon=256
**Hypothesis:** Agent needs longer credit assignment horizon. gamma=0.999 failed alone (exp057) but might work with curriculum + horizon=256.

**Changes:** gamma=0.999 (from 0.995), horizon=256 (from 128), gae_lambda=0.9 (from 0.95), vf_clip_coef=1.0

**Config:** board=20, network-scale=2, flood-fill, aux, curriculum, 500M steps
**Result:** **313.2/397 (78.9%)** at 271M steps — **massive new best!**

**Resumed run:** Reached **326.5/397 (82.2%)**

**Error analysis (50 eps, epoch 4200):**
- mean=295.8, median=322, max=336 (84.8%)
- 100% self-collision deaths (0% wall deaths)
- 29/50 episodes reach 80%+ fill
- Reachability at death: 16% (down from 50% in exp068) — agent now dies genuinely trapped
- 3/50 early blunders (<15% fill)

**Key insight:** gamma=0.999 previously failed alone, but works with curriculum + higher horizon. Credit assignment was the bottleneck — agent needs to propagate rewards ~700 steps to learn from decisions that create traps.

---

## Phase 10: Head-Centered Observation

### exp072 - Head-Centered + Bug Fixes
**Hypothesis:** Head-centered observation (39x39 grid centered on head) provides translation invariance — agent sees the same local pattern regardless of board position.

**Changes:**
- Head-centered observation: 5 x 39 x 39 (vs 5 x 22 x 22 board-centered)
- Head always at grid center (19, 19)
- Walls computed dynamically based on head position
- **Bug fix:** flood-fill now marks ALL reachable connected components (was only marking first neighbor's component due to premature `break`)
- **Bug fix:** aux flood-fill decoder target corrected for head-centered mode

**Config:** board=20, network-scale=2, head-centered, flood-fill, aux, curriculum, gamma=0.999, horizon=256, 500M steps

**Progression:**
| Steps | Eval Score | % |
|-------|-----------|---|
| 10M | 42.0 | 10.6% |
| 40M | 157.6 | 39.7% |
| 100M | 236.8 | 59.6% |
| 130M | 281.6 | 70.9% |
| 170M | 317.6 | 80.0% |
| 240M | 322.1 | 81.1% |
| 260M | 371.8 | 93.6% |

**Best eval: 371.8/397 (93.6%)**

**Error analysis (50 eps, epoch 6200 ~411M steps):**
- mean=278.7, median=360, max=**394 (99.2% fill — 3 food from perfect!)**
- 5 episodes scored 394/397
- Bimodal failure: early blunders (<20% fill, ~18%) or brilliant play to 90%+
- 100% self-collision deaths
- Aux loss dropped from ~0.2 (pre-fix) to ~0.077 (post-fix)

---

## Results Summary

| Board | Best Result | Experiment | Key Changes |
|-------|-------------|------------|-------------|
| 10x10 | **100% win rate** | exp007 | Egocentric + symmetric, 26M steps |
| 20x20 | 162.2/397 (41%) | exp056 | Ultra-conservative finetune |
| 20x20 | 216.4/397 (54%) | exp061 | + Flood-fill observation |
| 20x20 | 255.2/397 (64%) | exp066 | + Aux flood-fill decoder, cosine LR |
| 20x20 | 264.2/397 (67%) | exp068 | + 1B steps, fast cosine decay |
| 20x20 | 313.2/397 (79%) | exp070 | + gamma=0.999, horizon=256, curriculum |
| 20x20 | **371.8/397 (94%)** | exp072 | + Head-centered obs, flood-fill bugfix |
| 20x20 | **394/397 (99.2%)** | exp072 | Best single episode (3 food from perfect) |

---

## Key Findings

1. **Egocentric observation is critical** - Rotating grid so snake faces "up" reduces 4 direction cases to 1
2. **Symmetric augmentation helps** - Horizontal flip provides effective data augmentation
3. **MLP 2x is optimal scale** - 1x collapses, 4x overfits, 2x is the sweet spot (4.4M params)
4. **Flood-fill observation channel is the biggest single improvement** - Broke the 40% ceiling to 54%
5. **gamma=0.999 + horizon=256 is the second biggest improvement** - 66% → 79%, but only works with curriculum spawning
6. **Head-centered observation + flood-fill bugfix** - 79% → 94% eval, 99.2% best single episode
7. **Curriculum spawning enables gamma=0.999** - Spawning at 50-85% fill gives agent late-game experience; gamma=0.999 alone causes instability
8. **Aux flood-fill decoder helps slightly** - Separate BCE loss on flood-fill prediction
9. **Cosine LR decay over 200M then constant min_lr** - Better than cosine over full run or no decay
10. **Distance shaping alpha=0.2 is essential** - Agent can't find food on 20x20 without it
11. **GRPO is 60-400x worse than PPO** - Designed for sparse rewards, fails with dense per-step rewards
12. **CNN doesn't scale to 20x20** - Learns faster on 10x10 but lower ceiling on 20x20
13. **Directional architectures all failed** - Ray-casting, 3-branch, attention all worse than MLP
14. **Weight-tied iterative CNN failed** - 3-4x worse than MLP
15. **Alpha decay, tail channel both hurt** - Removed features
16. **Policy oscillation is real but manageable** - Long training + cosine LR + curriculum overcomes it

## Network Architectures

| Scale | Backbone | Policy Head | Value Head | Params |
|-------|----------|-------------|------------|--------|
| 1x | 1024→512→256→128 | 128→64→3 | 128→128→64→1 | 1.5M |
| 2x | 2048→1024→512→256 | 256→128→3 | 256→256→128→1 | 4.4M |
| 4x | 4096→2048→1024→512 | 512→256→3 | 512→512→256→1 | 14.5M |

## Commands

```bash
# Current best config (exp072)
caffeinate -i python -u train.py --board-size 20 --timesteps 500000000 --num-envs 256 --horizon 256 --minibatch-size 8192 --symmetric --network-scale 2 --device mps --eval-every-steps 5000000 --eval-deterministic --eval-episodes 10 --flood-fill --aux-flood-fill --gamma 0.999 --gae-lambda 0.9 --vf-clip-coef 1.0 --curriculum-prob 0.3 --head-centered --cosine-lr-steps 200000000 --min-lr 3e-5 --exp-name exp072_head_centered

# Evaluate checkpoint
python eval.py experiments/checkpoint.pt --board-size 20 --episodes 100 --deterministic --device mps --flood-fill --aux-flood-fill --network-scale 2 --head-centered

# Error analysis
python error_analysis.py experiments/checkpoint.pt --board-size 20 --episodes 50 --device mps --network-scale 2 --head-centered

# Watch agent play
python play.py experiments/checkpoint.pt --board-size 20 --device mps --network-scale 2 --flood-fill --aux-flood-fill --head-centered --delay 0.05

# Export for web
python export_web.py experiments/checkpoint.pt --board-size 20 --network-scale 2 --head-centered --output weights.json
```

## Remaining Failure Modes

At 94% eval (exp072), deaths are:
- **Early blunders (~18%):** Random-looking deaths at <20% fill, likely policy noise
- **Self-trapping at 80-90% fill:** Agent boxes itself into corners with no escape
- **At death:** 52% are true traps (<=3 reachable cells), 48% have room but make bad local moves
- **Negative correlation** between score and reachability at death: good games die trapped, bad games die with room still available

## Next Steps

1. **Reach perfect play (397/397)** - Agent already scores 394 in best episodes
2. **Reduce early blunders** - 18% of episodes fail before 20% fill
3. **Better endgame planning** - Search-augmented play (MCTS) at 80%+ fill
4. **Longer training** - exp072 still improving at 500M steps

---

## 2026-05-05 Path-Efficient RNN Follow-up

The win-rate objective is now solved by a pure RNN policy, but its strategy is cycle-like and path-inefficient. Current path-efficiency baseline is:

- `experiments/shortcut_base_label_strong_s114c_20260505T120413/lr3p00e-07_ep10.pt`
- 15-seed gate: 15/15 wins, mean_win_steps=39648.73, p95_win_steps=40974.4, steps_per_food=99.87

Sequence-level shortcut BC was tested to see whether recurrent/full-model updates can absorb train-time path-planner labels better than frozen-hidden policy-head patches. All inference remained pure neural network, with no planner/search/rule fallback at eval.

Results:

- `rnn_online_grid_path_seq_s117_20260505T160559`: full grid-path teacher rollout, lr=1e-7, KL=0.1. Failed the hard gate at 14/15 wins; seed 20100 died at score 395 in an exact rerun.
- `rnn_online_tail_base_seq_s118_20260505T162619`: base rollout, tail-path labels through score 200, lr=3e-8, KL=0.5. Preserved 15/15 but worsened mean_win_steps to 39697.0.
- `rnn_online_grid_base_seq_s119_20260505T163626`: base rollout, grid-path labels, lr=3e-8, KL=0.5. Failed hard gate at 12/15 wins.
- `rnn_online_tail_base_corrections_s120_20260505T164700`: base rollout, tail-path correction-only CE, lr=1e-8, KL=1.0. Preserved 15/15 but was an exact no-op on the gate: mean_win_steps=39648.73.
- `rnn_online_tail_base_corrections_s121_20260505T165623`: same correction-only setup with lr=3e-8. Preserved 15/15 but worsened mean_win_steps to 39697.0.
- `rnn_teacher_eval.py` was added to evaluate train-time shortcut teachers directly. On seeds 30034,20008,30014,20100,30023, `grid_path` won 5/5 with mean_win_steps=38001.2, while `tail_path --shortcut-score-max 200` failed 0/5 by stalling near score 201.
- `rnn_online_grid_full_low_s122_20260505T171103` and `rnn_online_grid_full_low_s123_20260505T171936`: full grid-path teacher rollout with KL=1.0 at lr=1e-8 and 3e-8. Both preserved 15/15 but were exact no-ops on the gate: mean_win_steps=39648.73.
- `rnn_online_grid_full_low_s124_20260505T172804`: full grid-path teacher rollout with lr=1e-7, KL=1.0. Broke the gate at 14/15 and worsened mean_win_steps among wins to 39736.5.
- `rnn_online_grid_scratch_s125_20260505T173632`: scratch online RNN distillation from grid-path teacher for 12 full teacher episodes. Teacher trajectories all won in 35.6k-39.3k steps and training action accuracy reached ~90%, but greedy eval stayed 0/5; seed 30034 died by wall at score 0 after 10 steps. Simple online BC from the shorter teacher is therefore insufficient.
- `distill.train_bc_rnn` was extended with `--teacher-mode`, `--max-episode-steps`, and shortcut planner arguments so replay-based RNN BC can train on full grid-path trajectories.
- `rnn_grid_replay_s126_20260505T174601`: replay BC from two full grid-path trajectories, 500 train steps. Sampled-window accuracy reached 94.3%, but greedy eval stayed 0/5; seed 30034 stalled at score 0 after 801 steps.
- `rnn_grid_replay_dagger_s127_20260505T174754`: same replay setup plus one DAgger round on student stall states. Accuracy reached 96.6%, but short eval stayed mean_score=0.2, win_rate=0.0 and hit the hard-kill criterion.
- `rnn_stochastic_harvest.py` was added to sample stochastic pure-policy rollouts near the winning RNN. On seed 30034, epsilon perturbations of 0.0001-0.0003 through score 200 produced 0/10 wins, usually dying early; random exploration is too destructive.
- `rnn_single_deviation_scan.py` found a stronger action-space signal. For seed 30034, forcing one alternate first action and then returning to greedy policy still won, improving 41596 -> 39883 or 39717 steps. A full step-0 scan across the 15-seed gate found 14 improved seed/action pairs, with best single-seed improvement 2458 steps.
- `rnn_targeted_action_patch.py` was added to train policy-head patches from harvested action trajectories plus anchors. A one-seed target-heavy probe improved seed 30034 to 38488 steps, but failed the broad gate by killing seed 30085 at score 36.
- `targeted_action_patch_multiseed_s134_20260506`: trained on the best step-0 improvement for every improved gate seed plus anchors for non-improved seeds. Weak settings preserved 15/15 but worsened mean_win_steps to 39983.4; stronger settings flipped some target actions but broke the hard gate.
- `rnn_early_head_patch.py` was added to train only the model's low-fill `early_policy_head`, leaving the base RNN and main policy head frozen. This keeps inference pure neural while localizing changes to length-3 initial-food behavior.
- `early_head_step0_s135_20260506/lr1p00e-03_ep1.pt` is the new path-efficiency frontier. It uses `early_head_max_fill=0.01`, passed the 15-seed gate at 15/15 with mean_win_steps=39582.8, and passed both 100-seed broad suites:
  - `20001-20100`: 100/100 wins, mean_win_steps=39830.03, p95_win_steps=41600.05.
  - `30001-30100`: 100/100 wins, mean_win_steps=39605.95, p95_win_steps=41546.6.
  - Combined broad mean: 39717.99 steps, improving the previous `shortcut_base_label_strong_s114c` combined mean of about 39795.32 by about 77.33 steps while preserving 200/200 deterministic wins.
- `single_deviation_s138_20260506/early_head_gate_lowfill_stride50.json` scanned the new early-head frontier for post-step-0 low-fill deviations. It found 8 genuine step-50 improvements, with best improvement 1379 steps on seed 20064.
- `early_head_step50_s139_20260506`: trained a second-round early head from those step-50 deviations. The best candidate preserved 15/15 but had mean_win_steps=39621.33, worse than the promoted `s135` frontier, so this second-round target set is not promoted.
- `single_deviation_s140_20260506/early_head_gate_score1_stride50.json` scanned length-4 / score-1 low-fill deviations and found 5 apparent improvements, including a 3893-step single-seed improvement on seed 30005.
- `early_head_score1_s141_20260506`: trained a wider early head with `early_head_max_fill=0.0126` from those score-1 deviations. The best small-gate candidate (`lr3p00e-04_ep1.pt`) improved the 15-seed gate to mean_win_steps=39512.27, but failed broad promotion: it preserved 200/200 wins with mean_win_steps=39826.71 on `20001-20100` and 39692.65 on `30001-30100`, for a combined mean of 39759.68. This is 41.69 steps worse than the `s135` frontier, so it is not promoted.
- `rnn_eval_seeds_batch.py` was added after sequential broad validation became too slow. It batches policy forward passes across exact seeds while keeping the same greedy RNN inference and independent `SnakeEnv` state transitions. A sanity check on seeds 30034 and 20008 matched the recorded gate step counts exactly.

Conclusion: global BC-style path-efficiency patches were not promotable, but a localized low-fill neural head can safely shift initial-food behavior and improve broad-suite path efficiency. The next branch should iterate from `early_head_step0_s135_20260506/lr1p00e-03_ep1.pt`, using the same 200-seed broad validation gate before promotion.

## 2026-05-06 Reliability Audit And Late-Failure Patch Attempts

The 200-seed broad gate was not enough to certify the refined mission. A fresh holdout check on `40001-40100` and `50001-50100` found that `early_head_step0_s135_20260506/lr1p00e-03_ep1.pt` and `early_head_broad_step0_top25_steponly_w50_s146_20260506/lr1p00e-04_ep1.pt` both score 197/200, failing the same three seeds: `40004`, `40014`, and `40099`. The pre-early-head clean RNN `shortcut_base_label_strong_s114c_20260505T120413/lr3p00e-07_ep10.pt` scores 196/200 on the same holdout, failing `40004`, `40055`, `40099`, and `50052`. The project is therefore not done under the reliability wording of `/goal`; the current best result is benchmark-slice mastery, not broad deterministic reliability.

`rnn_initial_action_scan_batch.py` was added to batch forced-initial-action scans. On the `s135` 200-seed broad suite, it found 118/200 seeds with at least one step-0 action improvement, with best single-seed improvement of 5079 steps. However, follow-up early-head training did not promote:

- `early_head_broad_step0_s143_20260506`: trained from all 118 broad step-0 improvements. Best result preserved 200/200 but had mean_win_steps=39748.54, worse than the `s135` mean of 39717.99.
- `early_head_broad_step0_top25_s144_20260506`: trained from the top 25 step-0 improvements. Best result preserved 200/200 but had mean_win_steps=39749.70, also worse than `s135`.
- `early_head_broad_step0_top25_steponly_s145_20260506`: step-0-only high-weight CE failed immediately on broad seeds.
- `early_head_broad_step0_top25_steponly_w50_s146_20260506`: lower-weight step-0-only CE technically improved the 200-seed in-slice mean to 39715.95 while preserving 200/200, but holdout validation stayed 197/200. This is not a reliability promotion.

The holdout failures are not irrecoverable by the base policy dynamics. `rnn_single_deviation_scan.py` found one-action winning continuations for all three `s135` holdout failures: `40004` needs action 0 instead of 1 at step 39782, `40014` needs action 2 instead of 1 at step 41800, and `40099` needs action 0 instead of 1 at step 42182. These are training-only labels; no inference-time deviation rule is allowed.

Late policy-head patching was tested and is currently falsified as a reliable route:

- `late_failfix_s149_20260506`: sparse terminal corrections plus sparse anchors. Low LR did not fix `40004`; medium LR fixed the target but introduced a new failure on `30086`; higher LR collapsed early.
- `late_failfix_dense_anchor_s150_20260506`: dense anchors on newly regressed seeds `30086` and `40043`. Low LR still missed `40004`; stronger settings caused early collapses.
- `late_failfix_traj_s151_20260506`: sampled the full winning deviation trajectories every 20 steps. Below the correction threshold it still missed `40004`; above the threshold it again introduced `30086` or early failures.

A hard-seed checkpoint scan over the relevant shortcut/early-head chain was saved to `experiments/hard_seed_checkpoint_scan_20260506.json`. No candidate cleared the discovered hard set `40004,40014,40099,40055,50052,30086,40043,20008`; best candidates reached only 5/8.

Conclusion: the current frontier remains `early_head_step0_s135_20260506/lr1p00e-03_ep1.pt` for broad path efficiency, but mission progress should now prioritize reliability on fresh deterministic seed suites. Single-action CE surgery has a real signal but is too brittle. The next defensible branch should use sequence-level or trust-region training on failure-conditioned late-game states, with a promotion gate that includes `20001-20100`, `30001-30100`, `40001-40100`, `50001-50100`, and the hard-seed set.

## 2026-05-15 Pause-State Reliability Audit

Training is paused. The active mission is still not complete under the refined reliability-plus-path-efficiency goal: a pure neural 20x20 policy must score 397/397 in deterministic greedy inference with no planner/search/rule fallback, and any path-efficiency gain must preserve 100% win rate on broad deterministic suites.

The strongest broad reliability candidate found in the later repair chain is currently `experiments/broad_anchor_40055_repair_s179_20260507/lr1p00e-04_ep1.pt`. It passed `20001-20100`, `30001-30100`, `40001-40100`, and `50001-50100` at 400/400 wins with mean_win_steps=39748.59, and also passed the 30-seed hard set at 30/30 wins with mean_win_steps=39699.17. It is not a completion candidate because the fresh `60001-60200` holdout was only 198/200, failing `60131` at score 396 and `60146` at score 393.

The exact-point repair candidate `experiments/exact_point_40043_repair_s172_20260507/lr1p00e-04_ep1.pt` passed its 29-seed hard audit, but a stop-on-first-failure broad audit from `20001` found a regression at seed `20250`: score 391, self-collision, step 38411. A one-action scan showed this failure is locally recoverable by taking action 2 instead of the policy's action 1 at step 38410, score 391; the greedy continuation then wins at step 38421. The saved label is `experiments/single_deviation_20250_s187_20260515/exact_point_s172_seed20250_step38410.json`.

Conclusion: the completion audit fails. The best known candidate is `broad_anchor_40055_repair_s179`, not the later exact-point repair, but it still has late holdout failures. The next resume-worthy branch should avoid another isolated point patch unless it is wrapped in broad sequence-level/trust-region anchoring. A concrete next experiment is sequence-level late-failure repair from `broad_anchor_40055_repair_s179/lr1p00e-04_ep1.pt` using targets for `60131` and `60146`, plus hard anchors including `40099`, `50085`, `50090`, `20099`, `40043`, `40004`, `30086`, `50052`, and the broad 20001-50100 suites. Promotion must require 600/600 wins on `20001-20100`, `30001-30100`, `40001-40100`, `50001-50100`, `60001-60200`, and no hard-set regressions.

`rnn_sequence_repair.py` was added as the resume-ready tool for that branch. Unlike `rnn_targeted_action_patch.py`, it trains through `forward_sequence` on fixed late-failure BPTT windows and applies a frozen-reference KL on broad anchor windows, so it is a sequence-level/trust-region repair rather than a cached-hidden-state head patch. The window packer was corrected to place real transitions at the beginning of short windows, not after left-padding, so padded timesteps cannot corrupt the recurrent state before useful context. A collect-only smoke test wrote `experiments/seq_repair_smoke_s188_20260515_anchor3/dataset_summary.json` and verified the intended data path: target windows for `60131` and `60146`, plus a fallback high-fill anchor window for `40099`. A tiny synthetic CPU smoke exercised `_train_sequence_candidate` on one random window and completed with finite CE/KL metrics, verifying the train loop shape/gradient path without launching a real Snake training run. No training run was launched during the pause.

Resume command template:

```bash
python -u rnn_sequence_repair.py \
  --base experiments/broad_anchor_40055_repair_s179_20260507/lr1p00e-04_ep1.pt \
  --trajectory-json experiments/final_layer_60131_60146_s184_20260508/points.json \
  --out-dir experiments/seq_repair_60131_60146_s188_20260515 \
  --board-size 20 --hidden-size 512 --device mps \
  --anchor-ranges 20001:100,30001:100,40001:100,50001:100 \
  --anchor-seeds 40099,50085,50090,20099,40043,40004,30086,50052 \
  --gate-seeds 60131,60146,40099,50085,50090,20099,40043,40004,30086,50052 \
  --lrs 1e-7,3e-7,1e-6 --epochs 1,2,4 \
  --window-len 544 --target-context-before 512 --target-context-after 32 \
  --anchor-stride 1000 --anchor-min-fill 0.9 --max-anchor-windows 400 \
  --target-weight 50 --trajectory-weight 0.1 --anchor-weight 0.1 \
  --trajectory-kl-weight 0.1 --anchor-kl-weight 1.0 --kl-coef 2.0 \
  --batch-size 8 --fail-fast --save-all
```

`rnn_promotion_audit.py` was added to turn the refined goal into a repeatable completion check. It runs deterministic greedy RNN inference only, writes per-suite and combined summaries, and emits a checklist covering pure neural inference, no planner/search/rule fallback, required perfect score, evaluated episode count, reliability pass/fail, optional mean/p95 win-step path-efficiency gates, and promotion pass/fail. A smoke audit on `broad_anchor_40055_repair_s179/lr1p00e-04_ep1.pt` with `good=40099:1,known_fail=60131:1` wrote `experiments/promotion_audit_smoke_s188_20260515.json`; it correctly passed `40099`, failed `60131` at score 396, and marked `promotion_passed=false`. A second path-gate smoke wrote `experiments/promotion_audit_pathgate_smoke_s188_20260515.json`; it passed reliability on `40099` but failed an intentionally strict `--max-mean-win-steps 39000` threshold, confirming that path efficiency can block promotion separately from win rate.

A current machine-readable completion audit was written to `experiments/current_completion_audit_20260515.json`. It records the objective, current frontier evidence, the `60131`/`60146` reliability gap, and the exact next action when training resumes. The tracked conclusion remains unchanged: status is `not_complete`.

Three bounded sequence-repair pilots were run from `broad_anchor_40055_repair_s179/lr1p00e-04_ep1.pt` using the `60131`/`60146` labels. `seq_repair_60131_60146_s188_pilot_20260515` used broad mini-anchors, target_weight=50, and kl_coef=2.0. Its best candidate was `lr3p00e-07_ep2.pt`: it fixed `60131` and preserved the hard anchors, but still failed `60146` at score 389, so it is not promoted. `seq_repair_60131_60146_s189_strongtarget_hard_20260515` increased target pressure to 150 with hard anchors only. This was worse: candidates fixed `60131` but regressed hard seeds such as `40043`, `40004`, or collapsed `50090` early. `seq_repair_60131_60146_s190_residual_hard_20260515` added a zero-initialized residual-head-only mode to localize updates. Safe residual settings preserved anchors but did not fix `60131`; the strongest setting regressed `40004`. Conclusion: the labels are real, but both stronger point pressure and residual-only locality are insufficient as currently configured. The next adjustment should not simply increase CE; it should use denser late-game anchors around `40043`/`40004`/`50090`, try a 60146-focused trust-region repair, or switch to a richer sequence objective that preserves pre-deviation behavior while changing only the terminal branch.

Promotion audit template:

```bash
python -u rnn_promotion_audit.py \
  experiments/<candidate>.pt \
  --board-size 20 --hidden-size 512 --device mps \
  --ranges 20001:100,30001:100,40001:100,50001:100,60001:200 \
  --hard-seeds 40099,50085,50090,20099,40043,40004,30086,50052,60131,60146 \
  --max-mean-win-steps <threshold> --max-p95-win-steps <threshold> \
  --max-steps 100000 --stop-after-failures 1 \
  --out experiments/<candidate>_promotion_audit.json
```
