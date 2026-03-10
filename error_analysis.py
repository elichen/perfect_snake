"""Detailed error analysis of Snake policy deaths."""

import sys
import numpy as np
import torch
import torch.nn as nn
from snake_env import SnakeEnv

# Import policy from eval.py
from eval import SnakePolicy


def analyze_deaths(checkpoint_path, board_size=20, episodes=50, device="mps",
                   network_scale=2, seed=12345, head_centered=False):
    n_channels = 6  # flood-fill obs
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy = SnakePolicy(board_size, scale=network_scale, n_channels=n_channels,
                         aux_flood_fill=True, head_centered=head_centered).to(device)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    env = SnakeEnv(n=board_size, gamma=0.99, alpha=0.2, seed=seed, flood_fill_obs=True, head_centered=head_centered)
    perfect_score = board_size * board_size - 3

    results = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        steps = 0
        last_info = info

        # Track action history for final moves
        action_history = []
        flood_fill_history = []

        while not done:
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, values = policy(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())

            # Record flood-fill reachability (channel 5) for last N steps
            ff_channel = obs[5]  # flood-fill channel
            total_cells = board_size * board_size
            if head_centered:
                # Head-centered: no wall padding to strip, count non-wall cells
                wall_channel = obs[4]
                reachable = int(np.sum((ff_channel > 0) & (wall_channel == 0)))
                body_cells = int(np.sum((obs[1] > 0) & (wall_channel == 0)))
            else:
                inner_ff = ff_channel[1:-1, 1:-1]  # strip walls
                reachable = int(np.sum(inner_ff > 0))
                body_cells = int(np.sum(obs[1][1:-1, 1:-1] > 0))  # body channel
            empty_cells = total_cells - body_cells - 1  # -1 for head
            reachable_pct = reachable / max(empty_cells, 1) * 100

            action_history.append({
                'action': action,
                'logits': logits[0].cpu().numpy().tolist(),
                'value': values[0].item(),
                'reachable': reachable,
                'empty': empty_cells,
                'reachable_pct': reachable_pct,
                'body_cells': body_cells,
            })

            obs, _, terminated, truncated, last_info = env.step(action)
            done = terminated or truncated
            steps += 1

        score = int(last_info.get("score", 0))
        snake_len = int(last_info.get("length", score + 3))
        reason = last_info.get("reason", "unknown")
        fill_pct = snake_len / (board_size * board_size) * 100

        # Get stats from last 10 steps before death
        last_n = action_history[-min(10, len(action_history)):]

        results.append({
            'ep': ep,
            'score': score,
            'snake_len': snake_len,
            'fill_pct': fill_pct,
            'reason': reason,
            'steps': steps,
            'last_reachable': last_n[-1]['reachable'] if last_n else 0,
            'last_reachable_pct': last_n[-1]['reachable_pct'] if last_n else 0,
            'last_empty': last_n[-1]['empty'] if last_n else 0,
            'last_value': last_n[-1]['value'] if last_n else 0,
            'last_logits': last_n[-1]['logits'] if last_n else [],
            'reachable_trend': [s['reachable'] for s in last_n],
            'reachable_pct_trend': [s['reachable_pct'] for s in last_n],
        })

        marker = " WIN!" if score >= perfect_score else ""
        print(f"  Ep {ep+1:3d}: score={score:3d}/{perfect_score}  fill={fill_pct:4.1f}%  "
              f"reason={reason:5s}  reachable={last_n[-1]['reachable'] if last_n else 0:3d}/"
              f"{last_n[-1]['empty'] if last_n else 0:3d} ({last_n[-1]['reachable_pct'] if last_n else 0:.0f}%)  "
              f"steps={steps:6d}{marker}", flush=True)

    # Analysis
    print("\n" + "=" * 70)
    print("DEATH ANALYSIS")
    print("=" * 70)

    scores = [r['score'] for r in results]
    print(f"\nScore stats: mean={np.mean(scores):.1f}  median={np.median(scores):.0f}  "
          f"std={np.std(scores):.1f}  min={np.min(scores)}  max={np.max(scores)}")

    # Death reasons
    print("\nDeath reasons:")
    reason_counts = {}
    reason_scores = {}
    for r in results:
        reason = r['reason']
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reason not in reason_scores:
            reason_scores[reason] = []
        reason_scores[reason].append(r['score'])
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        avg = np.mean(reason_scores[reason])
        print(f"  {reason:8s}: {count:3d} ({count/len(results)*100:.0f}%)  avg_score={avg:.1f}")

    # Fill % distribution at death
    fills = [r['fill_pct'] for r in results]
    print(f"\nFill % at death: mean={np.mean(fills):.1f}%  median={np.median(fills):.1f}%")
    buckets = [0]*10
    for f in fills:
        b = min(int(f // 10), 9)
        buckets[b] += 1
    print("  Distribution:")
    for i, c in enumerate(buckets):
        bar = "#" * c
        print(f"    {i*10:3d}-{(i+1)*10:3d}%: {c:3d} {bar}")

    # Reachability at death
    print("\nReachability at death (what % of empty cells were reachable from head):")
    reach_pcts = [r['last_reachable_pct'] for r in results if r['reason'] != 'win']
    if reach_pcts:
        print(f"  mean={np.mean(reach_pcts):.1f}%  median={np.median(reach_pcts):.1f}%")
        low_reach = [r for r in results if r['last_reachable_pct'] < 50 and r['reason'] != 'win']
        high_reach = [r for r in results if r['last_reachable_pct'] >= 80 and r['reason'] != 'win']
        print(f"  Deaths with <50% reachable: {len(low_reach)} ({len(low_reach)/len(results)*100:.0f}%)")
        print(f"  Deaths with >=80% reachable: {len(high_reach)} ({len(high_reach)/len(results)*100:.0f}%)")

    # Self-trapping analysis: reachable cells dropping to 0 or near-0
    print("\nSelf-trapping analysis (reachable trend in last 10 steps before death):")
    trap_deaths = []
    non_trap_deaths = []
    for r in results:
        if r['reason'] in ('self', 'wall'):
            trend = r['reachable_trend']
            if len(trend) >= 3 and trend[-1] <= 3:
                trap_deaths.append(r)
            else:
                non_trap_deaths.append(r)

    print(f"  Trapped (<=3 reachable at death): {len(trap_deaths)}")
    print(f"  Non-trapped (>3 reachable at death): {len(non_trap_deaths)}")

    if trap_deaths:
        trap_scores = [r['score'] for r in trap_deaths]
        print(f"  Trapped avg score: {np.mean(trap_scores):.1f}")
    if non_trap_deaths:
        nt_scores = [r['score'] for r in non_trap_deaths]
        print(f"  Non-trapped avg score: {np.mean(nt_scores):.1f}")

    # Show worst and best episodes
    sorted_results = sorted(results, key=lambda x: x['score'])
    print("\n5 WORST episodes:")
    for r in sorted_results[:5]:
        print(f"  Ep {r['ep']+1}: score={r['score']:3d}  fill={r['fill_pct']:.1f}%  "
              f"reason={r['reason']}  reachable_at_death={r['last_reachable']}/{r['last_empty']} "
              f"({r['last_reachable_pct']:.0f}%)")
        print(f"    Reachable trend (last 10): {[f'{x:.0f}%' for x in r['reachable_pct_trend']]}")

    print("\n5 BEST episodes:")
    for r in sorted_results[-5:]:
        print(f"  Ep {r['ep']+1}: score={r['score']:3d}  fill={r['fill_pct']:.1f}%  "
              f"reason={r['reason']}  reachable_at_death={r['last_reachable']}/{r['last_empty']} "
              f"({r['last_reachable_pct']:.0f}%)")

    # Score vs reachability correlation
    death_results = [r for r in results if r['reason'] != 'win']
    if death_results:
        scores_d = np.array([r['score'] for r in death_results])
        reaches_d = np.array([r['last_reachable_pct'] for r in death_results])
        if len(scores_d) > 2:
            corr = np.corrcoef(scores_d, reaches_d)[0, 1]
            print(f"\nCorrelation(score, reachability_at_death) = {corr:.3f}")

    # Deaths by fill % bucket — which ones are self-traps vs which are not
    print("\nDeath mode by fill bucket:")
    for i in range(10):
        lo, hi = i*10, (i+1)*10
        bucket_deaths = [r for r in results if lo <= r['fill_pct'] < hi and r['reason'] != 'win']
        if bucket_deaths:
            trapped = sum(1 for r in bucket_deaths if r['last_reachable'] <= 3)
            avg_reach = np.mean([r['last_reachable_pct'] for r in bucket_deaths])
            print(f"  {lo:3d}-{hi:3d}%: {len(bucket_deaths):3d} deaths, "
                  f"{trapped} trapped ({trapped/len(bucket_deaths)*100:.0f}%), "
                  f"avg reachability={avg_reach:.0f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Error analysis of Snake policy deaths")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--board-size", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--network-scale", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--head-centered", action="store_true")
    args = parser.parse_args()
    analyze_deaths(args.checkpoint, board_size=args.board_size, episodes=args.episodes,
                   device=args.device, network_scale=args.network_scale, seed=args.seed,
                   head_centered=args.head_centered)
