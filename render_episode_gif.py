"""Render a recorded episode (web_build/verify_episode.json) to a GIF."""
import argparse
import json

import imageio.v2 as imageio
import numpy as np


def render(ep, out_path, target_frames=360, cell=18, pad=2):
    n = ep["board_size"]
    frames = ep["frames"]
    total = len(frames)
    step = max(1, total // target_frames)
    idxs = list(range(0, total, step))
    if idxs[-1] != total - 1:
        idxs.append(total - 1)

    size = n * cell
    imgs = []
    BG = np.array([18, 18, 22], np.uint8)
    GRID = np.array([32, 34, 40], np.uint8)
    BODY = np.array([46, 160, 70], np.uint8)
    HEAD = np.array([130, 240, 140], np.uint8)
    FOOD = np.array([230, 60, 60], np.uint8)

    for i in idxs:
        f = frames[i]
        img = np.zeros((size, size, 3), np.uint8)
        img[:] = BG
        # subtle grid
        for g in range(0, size, cell):
            img[g, :] = GRID
            img[:, g] = GRID

        def paint(r, c, color):
            y0, x0 = r * cell + pad, c * cell + pad
            img[y0:y0 + cell - pad, x0:x0 + cell - pad] = color

        if f["food"] and f["food"][0] >= 0:
            paint(f["food"][0], f["food"][1], FOOD)
        for j, (r, c) in enumerate(f["snake"]):
            paint(r, c, HEAD if j == 0 else BODY)
        imgs.append(img)

    # Hold the final (won) frame longer
    durations = [0.04] * len(imgs)
    durations[-1] = 2.0
    imageio.mimsave(out_path, imgs, duration=durations, loop=0)
    print(f"wrote {out_path}: {len(imgs)} frames (every {step} steps of {total}), "
          f"won={ep['won']} score={ep['score']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="web_build/verify_episode.json")
    ap.add_argument("--out", default="web_build/example_win.gif")
    args = ap.parse_args()
    with open(args.episode) as f:
        ep = json.load(f)
    render(ep, args.out)
