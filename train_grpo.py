"""Train Snake with GRPO (Group Relative Policy Optimization).

GRPO eliminates the value network by computing advantages from group statistics:
- Collect complete episodes from parallel environments
- Advantage = (episode_return - mean) / std across batch
- Single gradient update with clipped policy ratio

Reference: DeepSeekMath (https://arxiv.org/abs/2402.03300)
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from typing import List

import gymnasium as gym
import numpy as np
import psutil
import torch
import torch.nn as nn

from snake_env import SnakeEnv
from experiment_tracker import ExperimentTracker


# =============================================================================
# Environment Wrappers (reused from train.py)
# =============================================================================


class SnakeSymmetricAugmentation(gym.Wrapper):
    """Horizontal flip augmentation for Snake."""

    def __init__(self, env: gym.Env, flip_prob: float = 0.5, seed: int = 0):
        super().__init__(env)
        self.flip_prob = flip_prob
        self.rng = np.random.default_rng(seed + 1)
        self.flipped = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed + 1)
        self.flipped = self.rng.random() < self.flip_prob
        obs, info = self.env.reset(seed=seed, options=options)
        if self.flipped:
            obs = np.flip(obs, axis=2).copy()
        return obs, info

    def step(self, action):
        if self.flipped:
            if action == 0:
                action = 2
            elif action == 2:
                action = 0
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.flipped:
            obs = np.flip(obs, axis=2).copy()
        return obs, reward, terminated, truncated, info


class SnakeEpisodeStats(gym.Wrapper):
    """Episode stats wrapper that logs final Snake score correctly."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_return = 0.0
        self.episode_length = 0

    def reset(self, *, seed=None, options=None):
        self.episode_return = 0.0
        self.episode_length = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_return += float(reward)
        self.episode_length += 1

        if terminated or truncated:
            reason = info.get("reason", None)
            end_info = {
                "episode_return": self.episode_return,
                "episode_length": self.episode_length,
                "episode_score": int(info.get("score", 0)),
                "episode_win": 1 if reason == "win" else 0,
            }
            return obs, reward, terminated, truncated, end_info

        return obs, reward, terminated, truncated, {}


def make_snake_env(*, n: int, gamma: float, alpha: float, symmetric: bool = False, seed: int = 0):
    """Create a Snake environment with optional augmentation."""
    env = SnakeEnv(n=n, gamma=gamma, alpha=alpha, seed=seed)
    if symmetric:
        env = SnakeSymmetricAugmentation(env, flip_prob=0.5, seed=seed)
    env = SnakeEpisodeStats(env)
    return env


# =============================================================================
# GRPO Policy (no value head)
# =============================================================================


class SnakePolicyGRPO(nn.Module):
    """FC policy for Snake without value head (GRPO doesn't need it)."""

    def __init__(self, obs_shape: tuple, n_actions: int, scale: int = 1):
        super().__init__()

        n_input = int(np.prod(obs_shape))

        # Scale network width
        w = [1024, 512, 256, 128]
        if scale == 2:
            w = [2048, 1024, 512, 256]
        elif scale == 4:
            w = [4096, 2048, 1024, 512]

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input, w[0]),
            nn.LayerNorm(w[0]),
            nn.ReLU(),
            nn.Linear(w[0], w[1]),
            nn.LayerNorm(w[1]),
            nn.ReLU(),
            nn.Linear(w[1], w[2]),
            nn.LayerNorm(w[2]),
            nn.ReLU(),
            nn.Linear(w[2], w[3]),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(w[3], w[3] // 2),
            nn.ReLU(),
            nn.Linear(w[3] // 2, n_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, observations):
        """Return action logits only (no value)."""
        features = self.features(observations)
        logits = self.policy_head(features)
        return logits

    def get_action_and_logprob(self, observations, deterministic: bool = False):
        """Sample action and return log probability."""
        logits = self.forward(observations)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy()


# =============================================================================
# Episode Buffer
# =============================================================================


@dataclass
class Episode:
    """A complete episode with all transitions."""
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    total_return: float = 0.0
    score: int = 0
    win: bool = False
    advantage: float = 0.0  # Episode-level advantage (credit="episode")
    step_advantages: np.ndarray = None  # Per-step advantages (credit="step")

    def finalize(self):
        """Compute total return when episode is done."""
        self.total_return = sum(self.rewards)


@dataclass
class EnvState:
    """Track state for a single environment."""
    obs: np.ndarray = None
    episode: Episode = field(default_factory=Episode)


# =============================================================================
# Vectorized Environment (simple sync version)
# =============================================================================


class SyncVecEnv:
    """Simple synchronous vectorized environment."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

        # Get observation/action space from first env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, seed=None):
        observations = []
        infos = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            observations.append(obs)
            infos.append(info)
        return np.stack(observations), infos

    def step(self, actions):
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        return (
            np.stack(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()


# =============================================================================
# GRPO Trainer
# =============================================================================


class GRPOTrainer:
    """GRPO training loop."""

    def __init__(
        self,
        policy: SnakePolicyGRPO,
        vecenv: SyncVecEnv,
        device: str,
        lr: float = 3e-4,
        clip_coef: float = 0.2,
        ent_coef: float = 0.02,
        max_grad_norm: float = 0.5,
        minibatch_size: int = 256,
        gamma: float = 0.99,
        credit: str = "episode",  # "episode" or "step"
    ):
        self.policy = policy
        self.vecenv = vecenv
        self.device = device
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.credit = credit

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        # Episode tracking per env
        self.env_states = [EnvState() for _ in range(vecenv.num_envs)]

        # Stats
        self.total_steps = 0
        self.total_episodes = 0

    def collect_episodes(self, min_episodes: int) -> List[Episode]:
        """Collect at least min_episodes complete episodes."""
        completed_episodes = []

        # Reset all envs and initialize states
        observations, _ = self.vecenv.reset(seed=None)
        for i, obs in enumerate(observations):
            self.env_states[i].obs = obs
            self.env_states[i].episode = Episode()

        while len(completed_episodes) < min_episodes:
            # Get actions from policy
            obs_tensor = torch.as_tensor(
                np.stack([s.obs for s in self.env_states]),
                dtype=torch.float32,
                device=self.device,
            )

            self.policy.eval()
            with torch.no_grad():
                actions, log_probs, _ = self.policy.get_action_and_logprob(obs_tensor)

            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()

            # Step environments
            next_obs, rewards, terminateds, truncateds, infos = self.vecenv.step(actions_np)
            dones = terminateds | truncateds

            # Record transitions
            for i in range(self.vecenv.num_envs):
                ep = self.env_states[i].episode
                ep.observations.append(self.env_states[i].obs)
                ep.actions.append(int(actions_np[i]))
                ep.log_probs.append(float(log_probs_np[i]))
                ep.rewards.append(float(rewards[i]))

                if dones[i]:
                    # Finalize episode
                    ep.finalize()
                    info = infos[i]
                    ep.score = info.get("episode_score", 0)
                    ep.win = info.get("episode_win", 0) == 1
                    completed_episodes.append(ep)
                    self.total_episodes += 1

                    # Reset this env's tracking
                    self.env_states[i].episode = Episode()

                self.env_states[i].obs = next_obs[i]

            self.total_steps += self.vecenv.num_envs

        return completed_episodes

    def compute_advantages(self, episodes: List[Episode]) -> None:
        """Compute group-relative advantages for each episode.

        Two modes:
        - "episode": All steps get the same advantage (episode return normalized)
        - "step": Each step gets return-to-go, then normalized across batch
        """
        if self.credit == "episode":
            # Original GRPO: episode-level credit
            returns = np.array([ep.total_return for ep in episodes])
            mean_return = np.mean(returns)
            std_return = np.std(returns) + 1e-8

            for ep in episodes:
                ep.advantage = (ep.total_return - mean_return) / std_return
                ep.step_advantages = None  # Use episode-level
        else:
            # Step-level credit: compute return-to-go at each step
            all_returns_to_go = []
            for ep in episodes:
                # Compute discounted return-to-go for each step
                returns_to_go = np.zeros(len(ep.rewards))
                running_return = 0.0
                for t in reversed(range(len(ep.rewards))):
                    running_return = ep.rewards[t] + self.gamma * running_return
                    returns_to_go[t] = running_return
                all_returns_to_go.extend(returns_to_go)
                ep.step_advantages = returns_to_go  # Store per-step

            # Normalize across entire batch
            all_returns_to_go = np.array(all_returns_to_go)
            mean_rtg = np.mean(all_returns_to_go)
            std_rtg = np.std(all_returns_to_go) + 1e-8

            # Apply normalization to each episode's step advantages
            for ep in episodes:
                ep.step_advantages = (ep.step_advantages - mean_rtg) / std_rtg
                ep.advantage = None  # Not used in step mode

    def update(self, episodes: List[Episode]) -> dict:
        """Perform policy update on collected episodes."""
        # Flatten all transitions
        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []

        for ep in episodes:
            for i, (obs, action, log_prob) in enumerate(zip(ep.observations, ep.actions, ep.log_probs)):
                all_obs.append(obs)
                all_actions.append(action)
                all_old_log_probs.append(log_prob)
                # Use step-level advantages if available, else episode-level
                if ep.step_advantages is not None:
                    all_advantages.append(ep.step_advantages[i])
                else:
                    all_advantages.append(ep.advantage)

        # Convert to tensors
        obs_tensor = torch.as_tensor(np.stack(all_obs), dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(all_actions, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.as_tensor(all_old_log_probs, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.as_tensor(all_advantages, dtype=torch.float32, device=self.device)

        # Normalize advantages (already normalized per-episode, but normalize again across batch)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Single pass update (GRPO uses 1 epoch, not multiple like PPO)
        n_samples = len(all_obs)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        total_loss = 0.0
        total_pg_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        n_batches = 0

        self.policy.train()

        for start in range(0, n_samples, self.minibatch_size):
            end = min(start + self.minibatch_size, n_samples)
            batch_idx = indices[start:end]

            batch_obs = obs_tensor[batch_idx]
            batch_actions = actions_tensor[batch_idx]
            batch_old_log_probs = old_log_probs_tensor[batch_idx]
            batch_advantages = advantages_tensor[batch_idx]

            # Forward pass
            logits = self.policy(batch_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            # Policy ratio
            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            # Clipped surrogate loss
            pg_loss1 = -batch_advantages * ratio
            pg_loss2 = -batch_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Entropy bonus
            loss = pg_loss - self.ent_coef * entropy

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Stats
            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

            total_loss += loss.item()
            total_pg_loss += pg_loss.item()
            total_entropy += entropy.item()
            total_clip_frac += clip_frac.item()
            n_batches += 1

        return {
            "loss": total_loss / max(1, n_batches),
            "pg_loss": total_pg_loss / max(1, n_batches),
            "entropy": total_entropy / max(1, n_batches),
            "clip_frac": total_clip_frac / max(1, n_batches),
            "n_episodes": len(episodes),
            "n_transitions": n_samples,
            "mean_return": np.mean([ep.total_return for ep in episodes]),
            "mean_score": np.mean([ep.score for ep in episodes]),
            "win_rate": np.mean([ep.win for ep in episodes]),
        }


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def evaluate_policy(
    *,
    policy: nn.Module,
    device: str,
    board_size: int,
    episodes: int,
    seed: int,
    deterministic: bool,
    gamma: float,
    alpha: float,
) -> dict:
    """Evaluate policy on single-threaded environment."""
    env = SnakeEnv(n=board_size, gamma=gamma, alpha=alpha, seed=seed)

    perfect_score = board_size * board_size - 3
    scores = []
    wins = 0

    policy.eval()
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        last_info = info
        while not done:
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
            logits = policy(obs_t)
            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                action = int(torch.distributions.Categorical(logits=logits).sample().item())
            obs, _, terminated, truncated, last_info = env.step(action)
            done = terminated or truncated

        score = int(last_info.get("score", 0))
        scores.append(score)
        if score >= perfect_score:
            wins += 1

    return {
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "win_rate": float(wins / max(1, episodes)),
        "perfect_score": int(perfect_score),
        "episodes": int(episodes),
    }


# =============================================================================
# Main
# =============================================================================


def _format_command() -> str:
    parts = [sys.executable] + sys.argv
    return " ".join(shlex.quote(part) for part in parts)


def main():
    parser = argparse.ArgumentParser(description="GRPO training on Snake")
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--min-episodes", type=int, default=64, help="Min episodes per update")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--credit", type=str, default="episode", choices=["episode", "step"],
                        help="Credit assignment: 'episode' (all steps same) or 'step' (return-to-go)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symmetric", action="store_true", help="Enable symmetric augmentation")
    parser.add_argument("--network-scale", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--eval-every-steps", type=int, default=0, help="0 = disable")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-deterministic", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Updates between checkpoints")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="experiments")
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set thread count
    torch.set_float32_matmul_precision("high")
    try:
        physical = psutil.cpu_count(logical=False) or 1
    except Exception:
        physical = 1
    torch.set_num_threads(max(1, physical))

    # Create vectorized environment
    env_fns = [
        lambda i=i: make_snake_env(
            n=args.board_size,
            gamma=args.gamma,
            alpha=args.alpha,
            symmetric=args.symmetric,
            seed=args.seed + i,
        )
        for i in range(args.num_envs)
    ]
    vecenv = SyncVecEnv(env_fns)

    # Create policy
    obs_shape = vecenv.observation_space.shape
    n_actions = vecenv.action_space.n
    policy = SnakePolicyGRPO(obs_shape, n_actions, scale=args.network_scale).to(args.device)

    # Create trainer
    trainer = GRPOTrainer(
        policy=policy,
        vecenv=vecenv,
        device=args.device,
        lr=args.lr,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        credit=args.credit,
    )

    # Experiment tracking
    exp_name = args.exp_name or f"grpo_snake_{args.board_size}"
    run_id = str(int(time.time() * 1000) % 1_000_000_000_000)

    config = {
        "algorithm": "grpo",
        "credit": args.credit,
        "env": exp_name,
        "seed": args.seed,
        "device": args.device,
        "total_timesteps": args.timesteps,
        "num_envs": args.num_envs,
        "min_episodes": args.min_episodes,
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "clip_coef": args.clip_coef,
        "ent_coef": args.ent_coef,
        "max_grad_norm": args.max_grad_norm,
        "minibatch_size": args.minibatch_size,
        "symmetric": args.symmetric,
        "network_scale": args.network_scale,
    }

    tracker = None
    try:
        tracker = ExperimentTracker(
            exp_name=exp_name,
            run_id=run_id,
            data_dir=args.data_dir,
            args=vars(args),
            config=config,
            command=_format_command(),
            cwd=os.getcwd(),
        )
    except Exception as exc:
        print(f"experiment_tracker_disabled: {exc}", file=sys.stderr)

    # Training loop
    start_time = time.time()
    last_eval_at = 0
    update_count = 0
    last_log_time = start_time

    print(f"Starting GRPO training: {args.timesteps:,} steps, {args.num_envs} envs, {args.min_episodes} episodes/update, credit={args.credit}")

    while trainer.total_steps < args.timesteps:
        # Collect episodes
        episodes = trainer.collect_episodes(args.min_episodes)

        # Compute advantages
        trainer.compute_advantages(episodes)

        # Policy update
        logs = trainer.update(episodes)
        update_count += 1

        # Compute SPS
        elapsed = time.time() - last_log_time
        sps = trainer.total_steps / (time.time() - start_time)

        # Log
        print(
            f"steps={trainer.total_steps:,} | SPS={sps:,.0f} | "
            f"loss={logs['loss']:.4f} | entropy={logs['entropy']:.3f} | "
            f"score={logs['mean_score']:.2f} | win={logs['win_rate']*100:.1f}%"
        )

        if tracker is not None:
            tracker.log_train({
                "agent_steps": trainer.total_steps,
                "SPS": sps,
                "losses/policy_loss": logs["pg_loss"],
                "losses/entropy": logs["entropy"],
                "losses/total_loss": logs["loss"],
                "charts/clip_frac": logs["clip_frac"],
                "environment/episode_return": logs["mean_return"],
                "environment/episode_score": logs["mean_score"],
                "environment/episode_win": logs["win_rate"],
            })

        # Periodic evaluation
        if args.eval_every_steps > 0 and trainer.total_steps - last_eval_at >= args.eval_every_steps:
            last_eval_at = trainer.total_steps

            stats = evaluate_policy(
                policy=policy,
                device=args.device,
                board_size=args.board_size,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
                deterministic=args.eval_deterministic,
                gamma=args.gamma,
                alpha=args.alpha,
            )
            print(
                f"eval: mean_score={stats['mean_score']:.2f}/{stats['perfect_score']} "
                f"win_rate={stats['win_rate']*100:.1f}% ({stats['episodes']} eps)"
            )
            if tracker is not None:
                tracker.log_eval(
                    stats,
                    agent_steps=trainer.total_steps,
                    epoch=update_count,
                    deterministic=args.eval_deterministic,
                )

        # Checkpointing
        if tracker is not None and args.checkpoint_interval > 0:
            if update_count % args.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    tracker.run_dir, f"model_{exp_name}_{update_count:06d}.pt"
                )
                torch.save(policy.state_dict(), checkpoint_path)
                tracker.log_checkpoint(
                    checkpoint_path,
                    epoch=update_count,
                    agent_steps=trainer.total_steps,
                )

        last_log_time = time.time()

    # Final save
    elapsed = time.time() - start_time
    final_checkpoint = None
    if tracker is not None:
        final_checkpoint = os.path.join(tracker.run_dir, f"model_{exp_name}_final.pt")
        torch.save(policy.state_dict(), final_checkpoint)
        tracker.finalize(
            status="completed",
            final_checkpoint=final_checkpoint,
            elapsed_seconds=elapsed,
        )

    vecenv.close()
    print(f"Training complete: {trainer.total_steps:,} steps in {elapsed:.1f}s ({trainer.total_steps/elapsed:,.0f} SPS)")


if __name__ == "__main__":
    main()
