# CLAUDE.md

## Project Overview

**Perfect Snake** is a reinforcement learning project training AI agents to achieve perfect play on the classic Snake game. The goal is 100% win rate on a 20x20 grid with a perfect score of 397 (snake fills entire board).

## Quick Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install torch gymnasium numpy pufferlib psutil

# Train 10x10 with symmetric augmentation (recommended)
python train.py --board-size 10 --timesteps 50000000 --num-envs 256 --horizon 128 --minibatch-size 8192 --symmetric --device mps

# Train 20x20 target
python train.py --board-size 20 --timesteps 100000000 --num-envs 256 --horizon 128 --minibatch-size 8192 --symmetric --device mps

# Resume training with optimizer/state (adds steps)
python train.py --board-size 20 --timesteps 50000000 --num-envs 256 --horizon 128 --minibatch-size 8192 --symmetric --network-scale 2 --device mps --resume-state experiments/exp016_20x20_2x_seed3_176618092705/trainer_state.pt --resume-add-steps --exp-name exp016_20x20_2x_seed3_resume

# List tracked experiments
python experiments.py list

# Inspect a specific run (prefix or full directory name)
python experiments.py show exp014_20x20_4x_176601447103

# Evaluate checkpoint
python eval.py experiments/checkpoint.pt --board-size 10 --episodes 100 --deterministic --device mps
```

## Architecture

### Files
- `train.py` - Main training script with PPO via PufferLib
- `experiment_tracker.py` - Writes run metadata, metrics, evals, and checkpoints
- `experiments.py` - CLI to list/show tracked experiments
- `snake_env.py` - Gymnasium Snake environment (egocentric observation)
- `eval.py` - Checkpoint evaluation script

### Neural Network (SnakePolicy)
- FC backbone: 1024 → 512 → 256 → 128
- Policy head: 128 → 64 → 3 actions
- Value head: 128 → 128 → 64 → 1

### Environment
- **Observation**: 5-channel grid (board_size+2 x board_size+2), snake-centric (egocentric)
  - Grid rotated so snake always faces "up"
  - Channels: head, body, food, normalized length, walls
- **Actions**: 3 discrete (turn left, straight, turn right) - relative to current direction
- **Rewards**: +1 food, -1 death, -0.5 stall, distance shaping (alpha=0.2)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--board-size` | 20 | Grid size |
| `--timesteps` | 1M | Total training steps |
| `--num-envs` | 64 | Parallel environments |
| `--horizon` | 128 | Steps per env per epoch |
| `--minibatch-size` | auto | SGD minibatch size |
| `--symmetric` | off | Enable horizontal flip augmentation |
| `--network-scale` | 1 | Network width multiplier (1, 2, or 4) |
| `--lr` | 3e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--alpha` | 0.2 | Distance shaping coefficient |
| `--backend` | mp | `mp` (multiprocessing) or `serial` |
| `--device` | cpu | `cpu`, `cuda`, or `mps` |
| `--eval-every-steps` | 0 | Run deterministic eval every N steps |
| `--eval-deterministic` | off | Use argmax for periodic eval |
| `--eval-episodes` | 50 | Episodes per eval |

## Results

Best config (10x10, egocentric + symmetric augmentation):
- **100% win rate** at ~26M steps
- **67% win rate** at 10M steps (eval with deterministic policy)

**Note:** High run-to-run variance due to MP non-determinism. Same config may produce 0% or 70%+ wins. Try different seeds if a run fails.

## Output

Training outputs go to `experiments/` directory:
- `{exp_name}_{run_id}.pt` - Final checkpoint (copied on close)
- `{exp_name}_{run_id}/` - Run directory with:
  - `run.json` (run metadata + config)
  - `metrics.jsonl` (train/eval/checkpoint events)
  - `summary.json` (best/last eval + final checkpoint)
  - `trainer_state.pt` and `model_*.pt` checkpoints
- `index.jsonl` - Append-only run index across experiments
