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

## Key Findings

1. **Egocentric observation is critical** - Rotating grid so snake faces "up" reduces 4 direction cases to 1
2. **Symmetric augmentation helps** - Horizontal flip provides effective data augmentation
3. **Deterministic eval >> stochastic training** - 67% eval win rate vs 32% training win rate at same checkpoint
4. **HIGH VARIANCE on 10x10** - Same config can give 0% or 100% win rate. Try multiple seeds.
5. **20x20 is much harder** - Best result: 39% of perfect score at 100M steps (exp012)
6. **Horizon=512 didn't help** - exp013 showed no improvement over horizon=128

## Network Architectures

| Scale | Backbone | Policy Head | Value Head | Params (10x10) |
|-------|----------|-------------|------------|----------------|
| 1x | 1024→512→256→128 | 128→64→3 | 128→128→64→1 | 1.5M |
| 2x | 2048→1024→512→256 | 256→128→3 | 256→256→128→1 | 4.4M |
| 4x | 4096→2048→1024→512 | 512→256→3 | 512→512→256→1 | 14.5M |

## Commands

```bash
# 10x10 baseline (recommended)
python train.py --board-size 10 --timesteps 50000000 --num-envs 256 --horizon 128 --minibatch-size 8192 --symmetric --device mps --eval-every-steps 1000000 --eval-deterministic --eval-episodes 10

# 20x20 with 2x network
python train.py --board-size 20 --timesteps 100000000 --num-envs 256 --horizon 128 --minibatch-size 8192 --symmetric --network-scale 2 --device mps --eval-every-steps 5000000 --eval-deterministic --eval-episodes 10

# Evaluate checkpoint
python eval.py experiments/checkpoint.pt --board-size 10 --episodes 100 --deterministic --device mps
```
