# CLAUDE.md

## Project Overview

**Perfect Snake** is a reinforcement learning project training AI agents to achieve perfect play on the classic Snake game. The goal is 100% win rate on a 20x20 grid with a perfect score of 397 (snake fills entire board).

## Quick Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install torch gymnasium numpy pufferlib psutil

# Train (basic)
python train.py --board-size 10 --timesteps 10000000 --device mps

# Train (full 20x20 target)
python train.py --board-size 20 --timesteps 50000000 --num-envs 256 --backend mp --device mps

# Deterministic training (for reproducibility)
python train.py --backend serial --seed 42
```

## Architecture

### Files
- `train.py` - Main training script with PPO via PufferLib
- `snake_env.py` - Gymnasium Snake environment implementation

### Neural Network (SnakePolicy)
- FC backbone: 1024 → 512 → 256 → 128
- Policy head: 128 → 64 → 3 actions
- Value head: 128 → 128 → 64 → 1

### Environment
- **Observation**: 9-channel grid (board_size+2 x board_size+2)
  - Channels: head, body, food, direction (4), normalized length, walls
- **Actions**: 3 discrete (turn left, straight, turn right) - relative to current direction
- **Rewards**: +1 food, -1 death, -0.5 stall, distance shaping (alpha=0.2)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--board-size` | 20 | Grid size |
| `--timesteps` | 1M | Total training steps |
| `--num-envs` | 64 | Parallel environments |
| `--horizon` | 128 | Steps per env per epoch |
| `--lr` | 3e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--alpha` | 0.2 | Distance shaping coefficient |
| `--backend` | mp | `mp` (multiprocessing) or `serial` |
| `--device` | cpu | `cpu`, `cuda`, or `mps` |

## Known Issues

- **Non-determinism**: Multiprocessing backend (`--backend mp`) causes run-to-run variance even with same seed. Use `--backend serial` for reproducible experiments.
- **Variance**: Experiments show high variance (70-80% win rate in one run, 0% in another with same config)

## Experiment Tracking

See `EXPERIMENTS.md` for detailed experiment log. Key findings:
- Baseline 10x10: High variance, best run achieved 70-80% win rate
- Larger networks underperformed
- Symmetric augmentation hurt performance

## Output

Training outputs go to `experiments/` directory:
- `model_*.pt` - Periodic checkpoints
- `trainer_state.pt` - Training state snapshot
