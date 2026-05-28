# AGENTS.md

## Project Overview

**Perfect Snake** is a reinforcement learning project training AI agents to achieve perfect play on the classic Snake game. The first goal is 100% win rate on a 20x20 grid with a perfect score of 397 (snake fills entire board). The refined goal is a pure neural policy that wins reliably while using fewer steps.

**Current status:** 10x10 solved (100% win rate). 20x20 is solved on benchmark seed slices by pure RNN policies, but the broad reliability-plus-path-efficiency mission is not complete. The current best reliability frontier is `experiments/broad_anchor_40055_repair_s179_20260507/lr1p00e-04_ep1.pt`: it passed 400/400 on `20001-50100` plus 30/30 hard seeds, but failed 2/200 on `60001-60200` (`60131`, `60146`). Training is paused; the next branch is sequence-level late-failure repair with strict promotion audit.

## Mission

- Train a **pure neural network** policy that achieves **100% deterministic win rate** on 20x20 Snake.
- Success means a **saved checkpoint** that scores **397/397** reliably in greedy play, not just occasional perfect training rollouts or stochastic samples.
- Refined success means the policy is also **path efficient**: among checkpoints that pass the win-rate gate, rank candidates by lower mean and tail steps-to-win on fixed deterministic seed suites.
- Treat win rate as a hard constraint. Do not accept fewer steps if the candidate introduces failures on the benchmark suite.
- Report `win_rate`, `mean_win_steps`, `p95_win_steps`, and `steps_per_food` for every serious RNN checkpoint comparison.
- Prioritize experiments that improve **stability, reproducibility, and branchability**, not just peak lucky-seed scores.
- Use the project as a scientific loop: form hypotheses, run controlled experiments, measure failure modes, and distinguish between **compute bottlenecks**, **optimization instability**, and **true dead ends**.

## Quick Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install torch gymnasium numpy pufferlib psutil

# 10x10 experiment (for quick arch iteration)
python train.py --board-size 10 --timesteps 50000000 --num-envs 256 --horizon 128 --minibatch-size 8192 --symmetric --device mps --eval-every-steps 1000000 --eval-deterministic --eval-episodes 10 --exp-name exp_name_here

# 20x20 experiment (full scale)
python train.py --board-size 20 --timesteps 100000000 --num-envs 256 --horizon 128 --minibatch-size 8192 --symmetric --network-scale 2 --device mps --eval-every-steps 1000000 --eval-deterministic --eval-episodes 10 --exp-name exp_name_here

# List tracked experiments
python experiments.py list

# Inspect a specific run
python experiments.py show exp022

# Evaluate checkpoint
python eval.py experiments/checkpoint.pt --board-size 20 --episodes 100 --deterministic --device mps

# Audit an RNN checkpoint against deterministic promotion gates
python rnn_promotion_audit.py experiments/checkpoint.pt --board-size 20 --hidden-size 512 --device mps --ranges 20001:100,30001:100,40001:100,50001:100,60001:200 --hard-seeds 40099,50085,50090,20099,40043,40004,30086,50052,60131,60146 --max-mean-win-steps THRESHOLD --max-p95-win-steps THRESHOLD --out experiments/checkpoint_promotion_audit.json
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
├── AGENTS.md             # This file - project overview
└── experiments/          # Training outputs
    ├── index.jsonl       # Append-only run index
    ├── {name}_{id}.pt    # Final checkpoints
    └── {name}_{id}/      # Run directories
        ├── run.json      # Config + metadata
        ├── metrics.jsonl # Train/eval events
        ├── summary.json  # Best/last eval results
        └── code/         # Archived source (snake_env.py, train.py)
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
  - Distance shaping: `-alpha * normalized_distance_to_food`

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
| `--eval-every-steps` | 1000000 | Eval every 1M steps (for quick feedback) |
| `--eval-episodes` | 10 | Episodes per eval (fast but sufficient) |

## Historical PPO 20x20 Plateau

Best result: **41% (score 162/397)** - exp056 (ultra-conservative finetune).

**Hypotheses tested:**
1. ~~Stall handling~~ (exp022 - didn't help)
2. ~~Alpha decay~~ (exp023 - made it worse: 35% vs 39%)
3. ~~Tail channel~~ (exp024/028 - hurt performance on both 20x20 and 10x10)
4. ~~Minimal 2-channel obs~~ (exp029 - worse: 57.5 on 10x10)
5. ~~Larger network 4x~~ (exp045 - worse: peak 83 vs 2x's 155)
6. ~~Smaller network 1x~~ (exp046 - peak 113 but collapsed)
7. ~~LR/entropy decay~~ (exp047-049 - unreliable, high variance)
8. ~~CNN with coordinates~~ (exp051-054 - all <92 on 20x20)
9. **Finetuning with low LR** (exp050/055/056 - most stable, new best peak 162)

**Core issue:** Policy oscillation / catastrophic forgetting — not architecture.

## Results Summary

| Board | Best Result | Steps | Experiment |
|-------|-------------|-------|------------|
| 10x10 | 100% win | 26M | exp007 |
| 20x20 PPO | 41% (162/397) | finetune | exp056 |
| 20x20 RNN | 400/400 broad + 30/30 hard, then 198/200 fresh holdout | ~39.7k mean win steps | broad_anchor_40055_repair_s179 |

**Note:** The PPO section above is historical. The active frontier is now pure RNN reliability repair and path-efficiency auditing.
