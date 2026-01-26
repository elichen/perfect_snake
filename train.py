"""Train Snake with PufferLib PPO."""

from __future__ import annotations

import multiprocessing
import sys

# Use 'spawn' to avoid PyTorch/MPS crashes on worker exit (macOS)
# Must be set before any multiprocessing happens
try:
    if sys.platform == 'darwin':
        multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # Already set

import argparse
import os
import shlex
import time

import gymnasium as gym
import numpy as np
import psutil
import torch
import torch.nn as nn

import pufferlib
import pufferlib.emulation
import pufferlib.vector
from pufferlib import pufferl

from snake_env import SnakeEnv
from experiment_tracker import ExperimentTracker


class SnakeSymmetricAugmentation(gym.Wrapper):
    """Horizontal flip augmentation for Snake.

    With probability `flip_prob`, flips the observation horizontally each episode.
    When flipped:
      - Observation columns are reversed
      - Actions swapped: left (0) ↔ right (2)
    """

    def __init__(self, env: gym.Env, flip_prob: float = 0.5, seed: int = 0):
        super().__init__(env)
        self.flip_prob = flip_prob
        self.rng = np.random.default_rng(seed + 1)  # Different seed from env
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
            # Swap left (0) and right (2), keep straight (1)
            if action == 0:
                action = 2
            elif action == 2:
                action = 0
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.flipped:
            obs = np.flip(obs, axis=2).copy()
        return obs, reward, terminated, truncated, info


class SnakeEpisodeStats(gym.Wrapper):
    """Episode stats wrapper that logs final Snake score correctly.

    PufferLib's generic `EpisodeStats` sums all per-step `info` values, which is
    wrong for cumulative fields like `score` and `length`. This wrapper emits
    episode-level aggregates only on termination/truncation:
      - `episode_return`: sum of rewards
      - `episode_length`: number of env steps
      - `episode_score`: final `info["score"]`
      - `episode_win`: 1 if `info["reason"] == "win"` else 0
    """

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


def make_snake_env(
    *,
    n: int,
    gamma: float,
    alpha: float,
    symmetric: bool = False,
    stall_penalty: float = -1.0,
    stall_terminates: bool = True,
    max_no_food_base: int = None,
    buf=None,
    seed=None,
):
    seed = 0 if seed is None else int(seed)
    env = SnakeEnv(
        n=n,
        gamma=gamma,
        alpha=alpha,
        seed=seed,
        stall_penalty=stall_penalty,
        stall_terminates=stall_terminates,
        max_no_food_base=max_no_food_base,
    )
    if symmetric:
        env = SnakeSymmetricAugmentation(env, flip_prob=0.5, seed=seed)
    env = SnakeEpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env, buf=buf)


class SnakePolicy(nn.Module):
    """FC policy for Snake.

    Scale 1x (default): obs -> 1024 -> 512 -> 256 -> 128
    Scale 2x: obs -> 2048 -> 1024 -> 512 -> 256
    """

    def __init__(self, env, scale: int = 1):
        super().__init__()

        obs_space = getattr(env, "single_observation_space", env.observation_space)
        act_space = getattr(env, "single_action_space", env.action_space)
        obs_shape = obs_space.shape
        n_input = int(np.prod(obs_shape))
        n_actions = act_space.n

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

        self.value_head = nn.Sequential(
            nn.Linear(w[3], w[3]),
            nn.ReLU(),
            nn.Linear(w[3], w[3] // 2),
            nn.ReLU(),
            nn.Linear(w[3] // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward_eval(self, observations, state=None):
        features = self.features(observations)
        logits = self.policy_head(features)
        values = self.value_head(features)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


class SnakeCNNPolicy(nn.Module):
    """CNN policy for Snake - better at spatial patterns."""

    def __init__(self, env, scale: int = 1):
        super().__init__()

        obs_space = getattr(env, "single_observation_space", env.observation_space)
        act_space = getattr(env, "single_action_space", env.action_space)
        obs_shape = obs_space.shape  # (C, H, W)
        n_channels = obs_shape[0]
        n_actions = act_space.n

        # Scale conv channels: 1x=small, 2x=medium, 4x=large
        if scale >= 4:
            c = [128, 256, 256]
            hidden = 1024
        elif scale >= 2:
            c = [64, 128, 128]
            hidden = 512
        else:
            c = [32, 64, 64]
            hidden = 256

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, c[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c[0], c[1], kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(c[1], c[2], kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            conv_out = self.conv(dummy).shape[1]

        self.features = nn.Sequential(
            nn.Linear(conv_out, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward_eval(self, observations, state=None):
        x = self.conv(observations)
        features = self.features(x)
        logits = self.policy_head(features)
        values = self.value_head(features)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


def _auto_num_workers(num_envs: int) -> int:
    try:
        physical = psutil.cpu_count(logical=False) or 1
    except Exception:
        physical = 1
    max_workers = min(num_envs, physical)
    for w in range(max_workers, 0, -1):
        if num_envs % w == 0:
            return w
    return 1


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
            logits, _ = policy.forward_eval(obs_t, state=None)
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


def _format_command() -> str:
    parts = [sys.executable] + sys.argv
    return " ".join(shlex.quote(part) for part in parts)


def _safe_int(value, default: int = 0) -> int:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return int(value)
    except Exception:
        return int(default)


def _get_agent_steps(logs, trainer) -> int:
    if logs is not None and "agent_steps" in logs:
        return _safe_int(logs["agent_steps"], 0)
    return _safe_int(getattr(trainer, "global_step", 0), 0)


def main():
    parser = argparse.ArgumentParser(description="PufferLib PPO on Snake")
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0, help="0 = auto")
    parser.add_argument(
        "--backend",
        type=str,
        default="mp",
        choices=["mp", "serial"],
        help="Vector backend (default: mp)",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.0,
        help="Final LR ratio for cosine anneal (0.0 = decay to 0, 0.1 ~= 3e-4->3e-5)",
    )
    parser.add_argument("--no-anneal-lr", action="store_true", help="Disable LR annealing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=128, help="Steps per env per epoch")
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=0,
        help="SGD minibatch size (0 = auto; must be divisible by --horizon)",
    )
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--ent-coef-final", type=float, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=0, help="0 = disable")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-deterministic", action="store_true")
    parser.add_argument("--perfect-patience", type=int, default=0, help="0 = disable early stop")
    parser.add_argument("--symmetric", action="store_true", help="Enable symmetric augmentation (50%% horizontal flip)")
    parser.add_argument("--network-scale", type=int, default=1, choices=[1, 2, 4], help="Network width multiplier (1=base, 2=2x, 4=4x)")
    parser.add_argument("--cnn", action="store_true", help="Use CNN policy instead of MLP")
    parser.add_argument("--stall-penalty", type=float, default=-1.0, help="Penalty for stalling (default: -1.0, same as death)")
    parser.add_argument("--stall-terminates", action="store_true", default=True, help="Stall ends episode (terminated=True, not truncated)")
    parser.add_argument("--no-stall-terminates", action="store_false", dest="stall_terminates", help="Stall truncates instead of terminates")
    parser.add_argument("--max-no-food-base", type=int, default=None, help="Override max steps without food (default: dynamic based on length)")
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name (default: auto-generated)")
    parser.add_argument("--data-dir", type=str, default="experiments")
    parser.add_argument("--prio-alpha", type=float, default=0.8)
    parser.add_argument("--prio-beta0", type=float, default=0.2)
    parser.add_argument("--resume", type=str, default=None, help="Path to a saved .pt state_dict")
    parser.add_argument(
        "--resume-state",
        type=str,
        default=None,
        help="Path to trainer_state.pt (restores optimizer, epoch, global_step)",
    )
    parser.add_argument(
        "--resume-add-steps",
        action="store_true",
        help="When resuming, add --timesteps to the saved global_step",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Enable PufferLib rich dashboard output (off by default for benchmarking)",
    )
    args = parser.parse_args()

    if args.num_envs < 1:
        raise SystemExit("--num-envs must be >= 1")
    if args.horizon < 1:
        raise SystemExit("--horizon must be >= 1")

    resume_state = None
    resume_steps = 0
    resume_epoch = 0
    resume_state_path = args.resume_state
    if resume_state_path is not None:
        if not os.path.exists(resume_state_path):
            raise SystemExit(f"--resume-state not found: {resume_state_path}")
        try:
            resume_state = torch.load(resume_state_path, map_location="cpu")
        except Exception as exc:
            raise SystemExit(f"--resume-state load failed: {exc}") from exc
        resume_steps = int(resume_state.get("global_step", 0))
        resume_epoch = int(resume_state.get("update", 0))

    backend = pufferlib.vector.Multiprocessing if args.backend == "mp" else pufferlib.vector.Serial
    num_workers = int(args.num_workers)
    if backend is pufferlib.vector.Multiprocessing:
        if num_workers <= 0:
            num_workers = _auto_num_workers(args.num_envs)
        if args.num_envs % num_workers != 0:
            raise SystemExit(
                f"--num-envs ({args.num_envs}) must be divisible by --num-workers ({num_workers})"
            )

    torch.set_float32_matmul_precision("high")
    try:
        physical = psutil.cpu_count(logical=False) or 1
    except Exception:
        physical = 1
    torch.set_num_threads(max(1, physical - (num_workers or 0)))

    env_kwargs = dict(
        n=args.board_size,
        gamma=args.gamma,
        alpha=args.alpha,
        symmetric=args.symmetric,
        stall_penalty=args.stall_penalty,
        stall_terminates=args.stall_terminates,
        max_no_food_base=args.max_no_food_base,
    )
    vec_kwargs = dict(
        num_envs=args.num_envs,
        seed=args.seed,
        backend=backend,
        env_kwargs=env_kwargs,
    )
    if backend is pufferlib.vector.Multiprocessing:
        vec_kwargs["num_workers"] = num_workers
    vecenv = pufferlib.vector.make(make_snake_env, **vec_kwargs)

    if args.cnn:
        policy = SnakeCNNPolicy(vecenv.driver_env, scale=args.network_scale).to(args.device)
    else:
        policy = SnakePolicy(vecenv.driver_env, scale=args.network_scale).to(args.device)

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        policy.load_state_dict(state, strict=True)
    elif resume_state is not None:
        model_name = resume_state.get("model_name")
        if model_name:
            resume_model = os.path.join(os.path.dirname(resume_state_path), model_name)
            if os.path.exists(resume_model):
                state = torch.load(resume_model, map_location="cpu")
                policy.load_state_dict(state, strict=True)
            else:
                print(f"warning: resume model not found: {resume_model}", file=sys.stderr)

    batch_size = args.num_envs * args.horizon
    if args.minibatch_size > 0:
        if args.minibatch_size % args.horizon != 0:
            raise SystemExit("--minibatch-size must be divisible by --horizon")
        minibatch_size = min(int(args.minibatch_size), int(batch_size))
    else:
        minibatch_segments = max(1, 256 // args.horizon)
        minibatch_size = minibatch_segments * args.horizon
        if minibatch_size > batch_size:
            minibatch_size = batch_size

    total_timesteps = int(args.timesteps)
    if resume_state is not None and args.resume_add_steps:
        total_timesteps = resume_steps + total_timesteps
    elif resume_state is not None and resume_steps >= total_timesteps:
        print(
            f"warning: resume global_step ({resume_steps}) >= total_timesteps ({total_timesteps})",
            file=sys.stderr,
        )

    config = {
        "env": args.exp_name if args.exp_name else f"snake_{args.board_size}",
        "seed": args.seed,
        "torch_deterministic": True,
        "cpu_offload": False,
        "device": args.device,
        "optimizer": "adam",
        "precision": "float32",
        "total_timesteps": int(total_timesteps),
        "learning_rate": float(args.lr),
        "anneal_lr": not bool(args.no_anneal_lr),
        "min_lr_ratio": float(args.min_lr_ratio),
        "gamma": float(args.gamma),
        "gae_lambda": 0.95,
        "update_epochs": int(args.update_epochs),
        "clip_coef": 0.1,
        "vf_coef": 1.0,
        "vf_clip_coef": 0.2,
        "max_grad_norm": 0.5,
        "ent_coef": float(args.ent_coef),
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "batch_size": int(batch_size),
        "minibatch_size": int(minibatch_size),
        "max_minibatch_size": 32768,
        "bptt_horizon": int(args.horizon),
        "compile": False,
        "compile_mode": "max-autotune-no-cudagraphs",
        "compile_fullgraph": True,
        "vtrace_rho_clip": 1.0,
        "vtrace_c_clip": 1.0,
        "prio_alpha": float(args.prio_alpha),
        "prio_beta0": float(args.prio_beta0),
        "use_rnn": False,
        "checkpoint_interval": int(args.checkpoint_interval),
        "data_dir": str(args.data_dir),
    }

    trainer = pufferl.PuffeRL(config, vecenv, policy)
    if not args.dashboard:
        trainer.print_dashboard = lambda *_, **__: None

    if resume_state is not None:
        if "optimizer_state_dict" in resume_state:
            trainer.optimizer.load_state_dict(resume_state["optimizer_state_dict"])

        trainer.global_step = resume_steps
        trainer.epoch = resume_epoch
        trainer.last_log_step = resume_steps
        trainer.last_log_time = time.time()

        # Properly restore or recreate scheduler
        if args.resume_add_steps:
            # When extending training, recreate scheduler with new T_max
            # This creates a fresh cosine schedule that continues from where we left off
            trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer,
                T_max=trainer.total_epochs,
                last_epoch=resume_epoch - 1,  # -1 because step() will increment
            )
            # Force LR update to correct value
            trainer.scheduler.step()
        elif "scheduler_state_dict" in resume_state:
            # Normal resume - restore exact scheduler state
            trainer.scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        else:
            # Fallback: just set last_epoch (may not work correctly)
            try:
                trainer.scheduler.last_epoch = resume_epoch
            except Exception:
                pass

        lr = trainer.optimizer.param_groups[0]["lr"]
        print(
            f"resume_state: steps={resume_steps} epoch={resume_epoch} "
            f"lr={lr:.2e} T_max={trainer.scheduler.T_max}",
            file=sys.stderr,
        )

    tracker = None
    try:
        tracker = ExperimentTracker(
            exp_name=config["env"],
            run_id=trainer.logger.run_id,
            data_dir=config["data_dir"],
            args=vars(args),
            config=config,
            command=_format_command(),
            cwd=os.getcwd(),
        )
    except Exception as exc:
        print(f"experiment_tracker_disabled: {exc}", file=sys.stderr)

    start_time = time.time()
    last_logs = None
    last_eval_at = 0
    perfect_streak = 0
    while trainer.epoch < trainer.total_epochs:
        trainer.evaluate()
        logs = trainer.train()
        if logs is not None:
            last_logs = logs
            sps = logs.get("SPS", 0)
            agent_steps = _get_agent_steps(logs, trainer)
            if tracker is not None:
                tracker.log_train(logs)
            ep_ret = logs.get("environment/episode_return", None)
            ep_len = logs.get("environment/episode_length", None)
            ep_score = logs.get("environment/episode_score", None)
            win_rate = logs.get("environment/episode_win", None)

            if args.ent_coef_final is not None and args.timesteps > 0:
                progress = min(1.0, agent_steps / float(args.timesteps))
                ent = float(args.ent_coef + progress * (float(args.ent_coef_final) - float(args.ent_coef)))
                trainer.config["ent_coef"] = ent

            extra = []
            if ep_ret is not None:
                extra.append(f"ep_ret={ep_ret:.2f}")
            if ep_len is not None:
                extra.append(f"ep_len={ep_len:.1f}")
            if ep_score is not None:
                extra.append(f"ep_score={ep_score:.2f}")
            if win_rate is not None:
                extra.append(f"win={win_rate*100:.1f}%")
            extra = (" | " + " ".join(extra)) if extra else ""
            print(f"steps={agent_steps:,} | SPS={sps:,.0f}{extra}")

            if args.eval_every_steps > 0 and agent_steps - last_eval_at >= args.eval_every_steps:
                last_eval_at = agent_steps

                stats = evaluate_policy(
                    policy=policy,
                    device=args.device,
                    board_size=args.board_size,
                    episodes=args.eval_episodes,
                    seed=args.seed + 10_000,
                    deterministic=args.eval_deterministic,
                    gamma=float(args.gamma),
                    alpha=float(args.alpha),
                )
                mean_score = stats["mean_score"]
                win_rate = stats["win_rate"]
                perfect_score = stats["perfect_score"]
                print(
                    f"eval: mean_score={mean_score:.2f}/{perfect_score} win_rate={win_rate*100:.1f}% "
                    f"({stats['episodes']} eps)"
                )
                if tracker is not None:
                    tracker.log_eval(
                        stats,
                        agent_steps=agent_steps,
                        epoch=trainer.epoch,
                        deterministic=args.eval_deterministic,
                    )

                if args.perfect_patience > 0 and win_rate >= 1.0:
                    perfect_streak += 1
                    if perfect_streak >= args.perfect_patience:
                        print(
                            f"early_stop: perfect win_rate for {perfect_streak}/{args.perfect_patience} evals"
                        )
                        break
                else:
                    perfect_streak = 0

        if tracker is not None:
            checkpoint_interval = int(args.checkpoint_interval)
            if checkpoint_interval > 0 and (
                trainer.epoch % checkpoint_interval == 0 or trainer.epoch >= trainer.total_epochs
            ):
                agent_steps = _get_agent_steps(last_logs, trainer)
                checkpoint_path = os.path.join(
                    tracker.run_dir, f"model_{config['env']}_{trainer.epoch:06d}.pt"
                )
                tracker.log_checkpoint(
                    checkpoint_path,
                    epoch=trainer.epoch,
                    agent_steps=agent_steps,
                )

    final_checkpoint = trainer.close()
    elapsed = time.time() - start_time
    if tracker is not None:
        tracker.finalize(
            status="completed",
            final_checkpoint=final_checkpoint,
            elapsed_seconds=elapsed,
        )
    if last_logs is not None and elapsed > 0:
        agent_steps = _safe_int(last_logs.get("agent_steps", 0), 0)
        print(f"avg_SPS={agent_steps/elapsed:,.0f} (steps={agent_steps:,}, seconds={elapsed:.1f})")


if __name__ == "__main__":
    main()
