# CLAUDE.md

## Project Overview

**Perfect Snake** is a reinforcement learning project training AI agents to achieve perfect play on the classic Snake game. The goal is 100% win rate on a 20x20 grid with a perfect score of 397 (snake fills entire board).

**Current status:** 10x10 solved (100% win rate). 20x20 plateaus at ~40% (score ~155/397). Active experimentation ongoing.

## Quick Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install torch gymnasium numpy pufferlib psutil

# Current experiment (alpha decay - exp023)
python train.py --board-size 20 --timesteps 100000000 --num-envs 256 --horizon 128 --minibatch-size 8192 --symmetric --network-scale 2 --device mps --eval-every-steps 5000000 --eval-deterministic --eval-episodes 10 --exp-name exp023_alpha_decay

# 10x10 baseline (for testing changes)
python train.py --board-size 10 --timesteps 50000000 --num-envs 256 --horizon 128 --minibatch-size 8192 --symmetric --device mps --eval-every-steps 5000000 --eval-deterministic --eval-episodes 10

# List tracked experiments
python experiments.py list

# Inspect a specific run
python experiments.py show exp022

# Evaluate checkpoint
python eval.py experiments/checkpoint.pt --board-size 20 --episodes 100 --deterministic --device mps
```

## Code Structure

```
perfect_snake/
├── train.py              # Main training script (PPO via PufferLib)
├── snake_env.py          # Gymnasium Snake environment
├── eval.py               # Checkpoint evaluation script
├── experiment_tracker.py # Writes run metadata, metrics, checkpoints
├── experiments.py        # CLI to list/show tracked experiments
├── experiments.md        # Experiment log with findings (READ THIS)
├── CLAUDE.md             # This file - project overview
└── experiments/          # Training outputs
    ├── index.jsonl       # Append-only run index
    ├── {name}_{id}.pt    # Final checkpoints
    └── {name}_{id}/      # Run directories
        ├── run.json      # Config + metadata
        ├── metrics.jsonl # Train/eval events
        └── summary.json  # Best/last eval results
```

## Where to Find Experiment Learnings

**`experiments.md`** - Full experiment log with:
- All past experiments and their results
- Key findings and failed approaches
- Network architectures table
- Recommended commands

**`experiments/{name}/summary.json`** - Per-run results:
- `best_eval` - Best evaluation score achieved
- `last_eval` - Final evaluation
- `last_train` - Final training metrics

**`python experiments.py list`** - Quick overview of all runs

## Architecture

### Neural Network (SnakePolicy)
| Scale | Backbone | Params |
|-------|----------|--------|
| 1x | 1024→512→256→128 | 1.5M |
| 2x | 2048→1024→512→256 | 4.4M |
| 4x | 4096→2048→1024→512 | 14.5M |

### Environment (snake_env.py)
- **Observation**: 5-channel grid (board_size+2 x board_size+2), egocentric
  - Grid rotated so snake always faces "up"
  - Channels: head, body, food, normalized length, walls
- **Actions**: 3 discrete (turn left, straight, turn right)
- **Rewards**:
  - +1 food, -1 death, -1 stall (configurable)
  - Distance shaping with alpha decay: `alpha * (1 - progress)` where progress = length/board_area

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--board-size` | 20 | Grid size |
| `--timesteps` | 1M | Total training steps |
| `--num-envs` | 64 | Parallel environments |
| `--horizon` | 128 | Steps per env per epoch |
| `--minibatch-size` | auto | SGD minibatch size |
| `--symmetric` | off | Horizontal flip augmentation |
| `--network-scale` | 1 | Width multiplier (1, 2, or 4) |
| `--lr` | 3e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--alpha` | 0.2 | Distance shaping coefficient |
| `--stall-penalty` | -1.0 | Penalty for stalling |
| `--stall-terminates` | true | Stall ends episode (not truncate) |
| `--device` | cpu | `cpu`, `cuda`, or `mps` |
| `--eval-every-steps` | 0 | Eval frequency (0=disable) |
| `--eval-episodes` | 50 | Episodes per eval |

## Current 20x20 Plateau

Best result: **39% (score 155/397)** - both exp012 and exp022 hit this wall.

**Hypotheses being tested:**
1. ~~Stall handling~~ (exp022 - didn't help)
2. **Alpha decay** (exp023 - running) - reduce distance shaping late-game
3. Tail channel observation - help agent see where space opens up
4. Larger network (4x) or different gamma

## Results Summary

| Board | Best Result | Steps | Experiment |
|-------|-------------|-------|------------|
| 10x10 | 100% win | 26M | exp007 |
| 20x20 | 39% (155/397) | 100M | exp012, exp022 |

**Note:** High variance. Same config can give 0% or 100%. Try multiple seeds.
