"""Sample and visualize levels from a diffusion checkpoint.

Usage:
    python -m minimax.add.sample_viz
    python -m minimax.add.sample_viz --ckpt checkpoints/diffusion/step_0050000.pkl --n 16
"""

import argparse
import glob
import os
import pickle

import jax
import numpy as np
import matthewplotlib as mp

from minimax.add.unet import UNet
from minimax.add.diffusion import make_schedule, ddim_sample_theta
from minimax.add.theta import decode_level


WALL_COLOR = (0.2, 0.2, 0.25)
OPEN_COLOR = (0.92, 0.92, 0.88)
AGENT_COLOR = (0.2, 0.6, 1.0)
GOAL_COLOR = (1.0, 0.3, 0.3)


def render_maze(wall_map, agent_pos, goal_pos):
    """Render a maze as an RGB image for matthewplotlib."""
    h, w = wall_map.shape
    img = np.where(
        wall_map[:, :, None],
        np.array(WALL_COLOR),
        np.array(OPEN_COLOR),
    )
    img[agent_pos[0], agent_pos[1]] = AGENT_COLOR
    img[goal_pos[0], goal_pos[1]] = GOAL_COLOR
    return img


def find_latest_ckpt(ckpt_dir="checkpoints/diffusion"):
    pkls = sorted(
        glob.glob(os.path.join(ckpt_dir, "step_*.pkl"))
    )
    if pkls:
        return pkls[-1]
    final = os.path.join(ckpt_dir, "final.pkl")
    if os.path.exists(final):
        return final
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()

    ckpt_path = args.ckpt or find_latest_ckpt()
    if ckpt_path is None:
        print("No checkpoint found.")
        return

    print(f"Loading {ckpt_path}...")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    params = jax.device_put(ckpt["ema_params"])
    step = ckpt.get("step", "?")
    print(f"Checkpoint step: {step}")

    model = UNet()
    schedule = make_schedule()

    def model_fn(params, x, t):
        return model.apply(params, x, t)

    print(
        f"Sampling {args.n} levels"
        f" ({args.ddim_steps} DDIM steps)..."
    )
    rng = jax.random.PRNGKey(args.seed)
    thetas = ddim_sample_theta(
        model_fn, params,
        (args.n, 16, 16, 3), rng, schedule, args.ddim_steps,
    )

    decoded = jax.vmap(decode_level)(thetas)
    wall_maps, agent_pos, goal_pos, agent_dirs = decoded
    wm = np.array(wall_maps)
    ap = np.array(agent_pos)
    gp = np.array(goal_pos)

    wall_counts = wm.reshape(args.n, -1).sum(axis=1)
    print(f"Wall counts: {wall_counts.tolist()}")
    print(
        f"Mean walls:"
        f" {wall_counts.mean():.1f} +/- {wall_counts.std():.1f}"
    )

    cols = min(args.cols, args.n)
    plots = []
    for i in range(args.n):
        img = render_maze(wm[i], ap[i], gp[i])
        plots.append(mp.image(img))
    grid = mp.wrap(*plots, cols=cols)
    print(grid)


if __name__ == "__main__":
    main()
