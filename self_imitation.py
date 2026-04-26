"""Self-imitation finetuning from elite stochastic episodes.

Collects episodes from a source checkpoint, keeps elite episodes by score,
filters to late-game states, then behavior-clones those actions into an
initial target checkpoint. Intended for the regime where PPO can sample
near-perfect play but does not make it the greedy default.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from eval import SnakePolicy, evaluate_checkpoint
from snake_env import SnakeEnv


@dataclass
class Episode:
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    score: int = 0
    win: bool = False
    terminal_length: int = 0


def episode_rank(ep: Episode) -> tuple[int, int, int]:
    return (int(ep.win), int(ep.score), int(ep.terminal_length))


def make_policy(
    checkpoint_path: str,
    *,
    board_size: int,
    device: str,
    network_scale: int,
    flood_fill: bool,
    aux_flood_fill: bool,
    head_centered: bool,
) -> SnakePolicy:
    n_channels = 5 + int(flood_fill)
    policy = SnakePolicy(
        board_size=board_size,
        scale=network_scale,
        n_channels=n_channels,
        aux_flood_fill=aux_flood_fill,
        aux_cycle_target=False,
        head_centered=head_centered,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if not aux_flood_fill:
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("flood_decoder")}
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy


def collect_episodes(
    policy: SnakePolicy,
    *,
    board_size: int,
    seed: int,
    episodes: int,
    num_envs: int,
    elite_count: int,
    device: str,
    flood_fill: bool,
    head_centered: bool,
    temperature: float,
    store_min_fill: float,
    progress_every: int = 0,
) -> tuple[list[Episode], np.ndarray, int]:
    envs = [
        SnakeEnv(
            n=board_size,
            gamma=0.999,
            alpha=0.2,
            seed=seed + i,
            flood_fill_obs=flood_fill,
            head_centered=head_centered,
        )
        for i in range(num_envs)
    ]
    obs_batch = []
    episode_states: list[Episode] = []
    elite_pool: list[Episode] = []
    completed_scores: list[int] = []
    completed_wins = 0

    for i, env in enumerate(envs):
        obs, _ = env.reset(seed=seed + i)
        obs_batch.append(obs)
        episode_states.append(Episode())

    next_seed = seed + num_envs
    while len(completed_scores) < episodes:
        obs_tensor = torch.as_tensor(np.stack(obs_batch), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = policy(obs_tensor)
            if temperature != 1.0:
                logits = logits / temperature
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy()

        for i, env in enumerate(envs):
            ep = episode_states[i]
            fill = float(obs_batch[i][3, 0, 0])
            if fill >= store_min_fill:
                ep.observations.append(obs_batch[i])
                ep.actions.append(int(actions[i]))

            next_obs, _, terminated, truncated, info = env.step(int(actions[i]))
            if terminated or truncated:
                ep.score = int(info.get("score", 0))
                ep.terminal_length = int(info.get("length", ep.score + 3))
                ep.win = info.get("reason") == "win" or ep.score >= (board_size * board_size - 3)
                completed_scores.append(ep.score)
                completed_wins += int(ep.win)

                elite_pool.append(ep)
                elite_pool.sort(key=episode_rank, reverse=True)
                if len(elite_pool) > elite_count:
                    elite_pool = elite_pool[:elite_count]
                if progress_every > 0 and len(completed_scores) % progress_every == 0:
                    elite_top = max((elite.score for elite in elite_pool), default=-1)
                    print(
                        f"collect_progress: episodes={len(completed_scores)}/{episodes} "
                        f"wins={completed_wins} mean_score={np.mean(completed_scores):.2f} "
                        f"best_score={max(completed_scores)} elite_top={elite_top}"
                    )

                if len(completed_scores) >= episodes:
                    break
                next_obs, _ = env.reset(seed=next_seed)
                next_seed += 1
                episode_states[i] = Episode()
            obs_batch[i] = next_obs

    return elite_pool, np.asarray(completed_scores[:episodes], dtype=np.int32), completed_wins


def build_dataset(
    episodes: list[Episode],
    *,
    min_fill: float,
    board_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    obs_list: list[np.ndarray] = []
    action_list: list[int] = []
    for ep in episodes:
        for obs, action in zip(ep.observations, ep.actions):
            fill = float(obs[3, 0, 0])
            if fill >= min_fill:
                obs_list.append(obs)
                action_list.append(action)

    if not obs_list:
        raise RuntimeError("No dataset samples passed the min_fill filter")

    observations = np.stack(obs_list).astype(np.float32)
    actions = np.asarray(action_list, dtype=np.int64)
    return observations, actions


def behavior_clone(
    policy: SnakePolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    device: str,
    lr: float,
    epochs: int,
    batch_size: int,
    policy_head_only: bool = False,
) -> None:
    params = list(policy.parameters())
    if policy_head_only:
        for param in params:
            param.requires_grad_(False)
        for param in policy.policy_head.parameters():
            param.requires_grad_(True)
        trainable_params = list(policy.policy_head.parameters())
    else:
        trainable_params = params
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    n_samples = observations.shape[0]
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    act_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)

    policy.train()
    for epoch in range(epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for start in range(0, n_samples, batch_size):
            idx = indices[start:start + batch_size]
            mb_obs = obs_tensor[idx]
            mb_actions = act_tensor[idx]

            logits, _ = policy(mb_obs)
            loss = F.cross_entropy(logits, mb_actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * len(idx)
            total_correct += int((logits.argmax(dim=-1) == mb_actions).sum().item())
            total_seen += len(idx)

        mean_loss = total_loss / max(1, total_seen)
        acc = total_correct / max(1, total_seen)
        print(f"bc_epoch={epoch+1}/{epochs} loss={mean_loss:.4f} acc={acc*100:.1f}%")

    policy.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-imitation finetuning from elite episodes")
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--network-scale", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--episodes", type=int, default=200, help="Stochastic episodes to collect")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--elite-count", type=int, default=32)
    parser.add_argument("--elite-min-score", type=int, default=390)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-fill", type=float, default=0.90)
    parser.add_argument("--bc-lr", type=float, default=1e-5)
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-batch-size", type=int, default=2048)
    parser.add_argument("--policy-head-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--flood-fill", action="store_true")
    parser.add_argument("--aux-flood-fill", action="store_true")
    parser.add_argument("--head-centered", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-num-envs", type=int, default=8)
    args = parser.parse_args()

    source_policy = make_policy(
        args.source_checkpoint,
        board_size=args.board_size,
        device=args.device,
        network_scale=args.network_scale,
        flood_fill=args.flood_fill,
        aux_flood_fill=args.aux_flood_fill,
        head_centered=args.head_centered,
    )
    target_policy = make_policy(
        args.init_checkpoint,
        board_size=args.board_size,
        device=args.device,
        network_scale=args.network_scale,
        flood_fill=args.flood_fill,
        aux_flood_fill=args.aux_flood_fill,
        head_centered=args.head_centered,
    )

    print(
        f"collect: episodes={args.episodes} num_envs={args.num_envs} "
        f"elite_count={args.elite_count} elite_min_score={args.elite_min_score} "
        f"temperature={args.temperature:.2f}"
    )
    episodes = collect_episodes(
        source_policy,
        board_size=args.board_size,
        seed=args.seed,
        episodes=args.episodes,
        num_envs=args.num_envs,
        elite_count=args.elite_count,
        device=args.device,
        flood_fill=args.flood_fill,
        head_centered=args.head_centered,
        temperature=args.temperature,
        store_min_fill=args.min_fill,
        progress_every=args.progress_every,
    )
    elites, scores, wins = episodes
    print(
        f"collected: wins={wins}/{len(scores)} mean_score={scores.mean():.2f} "
        f"median_score={np.median(scores):.1f} max_score={scores.max()}"
    )

    elites = [ep for ep in elites if ep.score >= args.elite_min_score] or elites
    elites = sorted(elites, key=episode_rank, reverse=True)[:args.elite_count]
    elite_scores = np.asarray([ep.score for ep in elites], dtype=np.int32)
    print(
        f"elite: count={len(elites)} wins={sum(ep.win for ep in elites)} "
        f"mean_score={elite_scores.mean():.2f} min_score={elite_scores.min()} "
        f"max_score={elite_scores.max()}"
    )

    observations, actions = build_dataset(
        elites,
        min_fill=args.min_fill,
        board_size=args.board_size,
    )
    print(
        f"dataset: samples={len(actions)} min_fill={args.min_fill:.2f} "
        f"action_hist={np.bincount(actions, minlength=3).tolist()}"
    )

    behavior_clone(
        target_policy,
        observations,
        actions,
        device=args.device,
        lr=args.bc_lr,
        epochs=args.bc_epochs,
        batch_size=args.bc_batch_size,
        policy_head_only=args.policy_head_only,
    )

    os.makedirs(os.path.dirname(args.output_checkpoint) or ".", exist_ok=True)
    torch.save(target_policy.state_dict(), args.output_checkpoint)
    print(f"saved: {args.output_checkpoint}")

    stats = evaluate_checkpoint(
        checkpoint_path=args.output_checkpoint,
        board_size=args.board_size,
        episodes=args.eval_episodes,
        seed=args.seed + 1000,
        deterministic=True,
        device=args.device,
        network_scale=args.network_scale,
        flood_fill=args.flood_fill,
        aux_flood_fill=args.aux_flood_fill,
        head_centered=args.head_centered,
        num_envs=max(1, min(args.eval_num_envs, args.eval_episodes)),
    )
    print(
        f"eval: mean_score={stats['mean_score']:.2f}/{stats['perfect_score']} "
        f"win_rate={stats['win_rate']*100:.1f}% median={stats['median_score']:.1f} "
        f"lt20={stats['phase_lt20_rate']*100:.1f}% 95+={stats['phase_gte95_rate']*100:.1f}%"
    )


if __name__ == "__main__":
    main()
