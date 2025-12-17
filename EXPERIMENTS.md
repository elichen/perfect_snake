# Experiment Log

**Goal:** 100% win rate on 20x20 Snake (perfect score = 397)

---

## Baseline Development (10x10)

Initial experiments on 10x10 grid to validate approach before scaling to 20x20.

### exp001 - Baseline Network
**Config:** board=10, obs=full, lr=3e-4, seed=42

**Network:**
- Backbone: 1024 → 512 → 256 → 128
- Policy: 128 → 64 → 3
- Value: 128 → 128 → 64 → 1

**Results @ 10M steps:** ~70-80% win rate, score ~80-90

**Notes:** High variance observed between runs even with same seed (multiprocessing non-determinism).

---

### exp002 - 2x Network Size
**Config:** Same as exp001, but doubled network widths

**Network:**
- Backbone: 2048 → 1024 → 512 → 256
- Policy: 256 → 128 → 3
- Value: 256 → 256 → 128 → 1

**Results @ 22M steps:** 0% win rate, score ~50

**Conclusion:** Larger network failed to learn. Likely needs lower LR or longer training.

---

### exp003 - 2x Network + Lower LR
**Config:** Same as exp002, lr=1e-4 (1/3 of baseline)

**Results @ 10M steps:** 0% win rate, score ~24

**Conclusion:** Lower LR made it worse. 2x network may be too large for this problem.

---

### exp004 - Symmetric Augmentation
**Config:** board=10, baseline network, 50% random horizontal flip per episode

**Implementation:**
- Flip observation horizontally
- Swap direction channels: right ↔ left
- Swap actions: left ↔ right

**Results @ 50M steps:** 0% win rate, score ~74-82, 43k SPS

**Conclusion:** Symmetric augmentation hurt performance significantly. Baseline achieved 70-80% wins at 10M steps; this achieved 0% wins at 50M steps. Possible issues:
- Per-episode flip may confuse learning (inconsistent world views across episodes)
- Implementation bug in flip logic?
- Augmentation may not be beneficial for this problem

---

## Key Findings

1. Baseline network (1024→512→256→128) achieves ~70-80% win rate on 10x10 at 10M steps
2. Doubling network size hurts performance (0% wins at 22M steps)
3. Symmetric augmentation hurts performance (0% wins at 50M steps)
4. High run-to-run variance despite fixed seed (MP non-determinism)

## Next Steps

- Run baseline to completion (50M steps) to establish true ceiling
- Scale to 20x20 once 10x10 is solved
