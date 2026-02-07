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

## Key Insight: Policy Oscillation

**All architectures show the same pattern:**
1. Score improves → finds good strategy
2. Keeps training → overwrites good weights
3. Score crashes → "forgets" what worked
4. Sometimes recovers, sometimes not

This is **policy oscillation** / catastrophic forgetting - a known PPO problem:
- No replay buffer (unlike DQN) - only learns from recent experience
- Once policy changes, old successful trajectories are gone
- Can "forget" good strategies mid-training

**The 39% plateau is NOT an architecture problem - it's training dynamics.**

---

## Key Findings

1. **Egocentric observation is critical** - Rotating grid so snake faces "up" reduces 4 direction cases to 1
2. **Symmetric augmentation helps** - Horizontal flip provides effective data augmentation
3. **Deterministic eval >> stochastic training** - 67% eval win rate vs 32% training win rate at same checkpoint
4. **HIGH VARIANCE on 10x10** - Same config can give 0% or 100% win rate. Try multiple seeds.
5. **20x20 plateaus at 39%** - All architectures (MLP, CNN, attention) hit ~155/397 wall
6. **Horizon=512 didn't help** - exp013 showed no improvement over horizon=128
7. **GRPO doesn't work for Snake** - 60-400x worse sample efficiency than PPO
8. **Stall handling didn't help** - exp022 showed same plateau
9. **Alpha decay made it worse** - exp023: 35% vs baseline 39%
10. **Tail channel hurt performance** - exp024/028: Added noise without useful signal
11. **CNN learns 6x faster on 10x10** - exp030: 100% win at 6M vs MLP at 40M
12. **CNN doesn't scale to 20x20** - exp031: Peaked at 75 vs MLP's 154
13. **Directional architectures all failed** - Ray-casting, 3-branch, attention all worse than MLP
14. **Policy oscillation is the core issue** - All models show same collapse pattern, not architecture-specific
15. **MLP 2x is optimal scale** - 1x collapses, 4x overfits, 2x best balance
16. **LR/entropy decay unreliable** - exp047-049: high variance, severe collapses
17. **Finetuning with low LR is most stable** - exp050: small variance, sustained ~110+
18. **Ultra-conservative finetune = new best** - exp056: peak 162.2 with LR=2.5e-5, 1 epoch
19. **CNN with coordinates still loses to MLP** - exp051-054: all <92 on 20x20

## Network Architectures

| Scale | Backbone | Policy Head | Value Head | Params (10x10) |
|-------|----------|-------------|------------|----------------|
| 1x | 1024→512→256→128 | 128→64→3 | 128→128→64→1 | 1.5M |
| 2x | 2048→1024→512→256 | 256→128→3 | 256→256→128→1 | 4.4M |
| 4x | 4096→2048→1024→512 | 512→256→3 | 512→512→256→1 | 14.5M |

## Commands

```bash
# Current experiment (alpha decay)
python train.py --board-size 20 --timesteps 100000000 --num-envs 256 --horizon 128 --minibatch-size 8192 --symmetric --network-scale 2 --device mps --eval-every-steps 5000000 --eval-deterministic --eval-episodes 10 --exp-name exp023_alpha_decay

# 10x10 baseline (for testing changes)
python train.py --board-size 10 --timesteps 50000000 --num-envs 256 --horizon 128 --minibatch-size 8192 --symmetric --device mps --eval-every-steps 5000000 --eval-deterministic --eval-episodes 10

# 20x20 baseline (before alpha decay)
python train.py --board-size 20 --timesteps 100000000 --num-envs 256 --horizon 128 --minibatch-size 8192 --symmetric --network-scale 2 --device mps --eval-every-steps 5000000 --eval-deterministic --eval-episodes 10

# Evaluate checkpoint
python eval.py experiments/checkpoint.pt --board-size 20 --episodes 100 --deterministic --device mps
```

## Next Experiments to Try

Core problem is **policy oscillation** — agent finds good strategies but overwrites them.

1. **Checkpoint ensembling / selection** - Save frequently, pick best checkpoint
2. **EWC / weight regularization** - Penalize moving away from good weights
3. **Population-based training** - Multiple seeds, keep best
4. **Curriculum learning** - Start small board, gradually increase
5. **MCTS + policy network** - Search-augmented play at eval time
6. **Gamma 0.995** - Longer credit assignment horizon
7. **More eval episodes** - 50+ to reduce eval variance
