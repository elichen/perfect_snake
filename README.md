# Perfect Snake

Training RL agents to achieve perfect play on Snake.

**Goal:** 100% win rate on a 20x20 grid (perfect score = 397)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch gymnasium numpy pufferlib psutil
```

## Run

```bash
python train.py \
    --board-size 20 --obs-type full \
    --timesteps 50000000 --num-envs 256 --horizon 128 --minibatch-size 8192 \
    --backend mp --device mps --seed 42
```

## Files

- `train.py` - Training script (PufferLib PPO)
- `snake_env.py` - Snake environment
- `EXPERIMENTS.md` - Experiment log
