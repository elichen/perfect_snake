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
