"""Stage 1: Train DDPM on procedurally generated maze levels.

Usage:
    tpu-device 0 python -m minimax.add.train_diffusion
    tpu-device 0 python -m minimax.add.train_diffusion --total_steps 1000 --log_every 50
"""

import argparse
import os
import time
import pickle
from collections import deque

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from minimax.add.theta import sample_random_theta, decode_level
from minimax.add.unet import UNet
from minimax.add.diffusion import make_schedule, compute_loss, ddim_sample_theta


def has_path_bfs(wall_map_np, start, goal):
    if wall_map_np[start] or wall_map_np[goal]:
        return False
    visited = set([start])
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 13 and 0 <= nc < 13 and (nr, nc) not in visited and not wall_map_np[nr, nc]:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


def save_checkpoint(path, state, ema_params, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "params": jax.device_get(state.params),
        "ema_params": jax.device_get(ema_params),
        "opt_state": jax.device_get(state.opt_state),
        "step": step,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_checkpoint(path, state):
    with open(path, "rb") as f:
        data = pickle.load(f)
    state = state.replace(
        params=jax.device_put(data["params"]),
        opt_state=jax.device_put(data["opt_state"]),
        step=data["step"],
    )
    ema_params = jax.device_put(data["ema_params"])
    return state, ema_params, data["step"]


def main():
    parser = argparse.ArgumentParser(description="Train DDPM on maze levels")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--ema_rate", type=float, default=0.9999)
    parser.add_argument("--total_steps", type=int, default=300_000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--sample_every", type=int, default=5000)
    parser.add_argument("--sample_size", type=int, default=64)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/diffusion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    rng = jax.random.PRNGKey(args.seed)
    schedule = make_schedule()
    model = UNet()

    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.ones((1, 16, 16, 3)), jnp.array([0]))
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"UNet parameters: {n_params:,}")

    tx = optax.adamw(args.lr, b1=0.9, b2=0.999, weight_decay=args.weight_decay)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )
    ema_params = state.params
    start_step = 0

    if args.resume:
        state, ema_params, start_step = load_checkpoint(args.resume, state)
        print(f"Resumed from step {start_step}")

    # --- JIT-compiled functions (close over model, schedule, config) ---

    @jax.jit
    def train_step(state, batch, rng):
        def loss_fn(params):
            return compute_loss(model.apply, params, batch, rng, schedule)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def gen_batch(rng):
        return jax.vmap(sample_random_theta)(jax.random.split(rng, args.batch_size))

    ema_rate = args.ema_rate

    @jax.jit
    def update_ema(ema, new):
        return jax.tree.map(lambda e, n: ema_rate * e + (1 - ema_rate) * n, ema, new)

    @jax.jit
    def sample_and_decode(params, rng):
        def model_fn(p, x, t):
            return model.apply(p, x, t)
        thetas = ddim_sample_theta(
            model_fn, params,
            (args.sample_size, 16, 16, 3), rng, schedule, args.ddim_steps,
        )
        wall_maps, agent_pos, goal_pos, agent_dirs = jax.vmap(decode_level)(thetas)
        return wall_maps, agent_pos, goal_pos

    def evaluate(params, rng):
        t0 = time.time()
        wall_maps, agent_pos, goal_pos = sample_and_decode(params, rng)
        wm = np.array(wall_maps)
        ap = np.array(agent_pos)
        gp = np.array(goal_pos)
        n = wm.shape[0]
        wall_counts = wm.reshape(n, -1).sum(axis=1)
        solvable = sum(
            has_path_bfs(wm[i], tuple(ap[i]), tuple(gp[i])) for i in range(n)
        )
        elapsed = time.time() - t0
        return {
            "solvable": solvable / n,
            "walls_mean": float(wall_counts.mean()),
            "walls_std": float(wall_counts.std()),
            "sample_time": elapsed,
        }

    # --- Training loop ---

    losses = []
    t0 = time.time()
    print(f"Training for {args.total_steps} steps, batch_size={args.batch_size}")

    for step in range(start_step, args.total_steps):
        rng, rng_batch, rng_train = jax.random.split(rng, 3)
        batch = gen_batch(rng_batch)
        state, loss = train_step(state, batch, rng_train)
        ema_params = update_ema(ema_params, state.params)
        losses.append(float(loss))

        if (step + 1) % args.log_every == 0:
            avg_loss = np.mean(losses[-args.log_every :])
            elapsed = time.time() - t0
            steps_done = step + 1 - start_step
            sps = steps_done / elapsed
            eta = (args.total_steps - step - 1) / sps if sps > 0 else 0
            print(
                f"step {step+1:>7d}/{args.total_steps} | "
                f"loss {avg_loss:.4f} | "
                f"{sps:.1f} steps/s | "
                f"ETA {eta/3600:.1f}h"
            )

        if (step + 1) % args.sample_every == 0:
            rng, rng_eval = jax.random.split(rng)
            metrics = evaluate(ema_params, rng_eval)
            print(
                f"  eval | solvable {metrics['solvable']:.1%} | "
                f"walls {metrics['walls_mean']:.1f}±{metrics['walls_std']:.1f} | "
                f"sampled in {metrics['sample_time']:.1f}s"
            )

        if (step + 1) % args.save_every == 0:
            path = os.path.join(args.ckpt_dir, f"step_{step+1:07d}.pkl")
            save_checkpoint(path, state, ema_params, step + 1)
            print(f"  saved {path}")

    path = os.path.join(args.ckpt_dir, "final.pkl")
    save_checkpoint(path, state, ema_params, args.total_steps)
    total_time = time.time() - t0
    print(f"Training complete in {total_time/3600:.1f}h. Saved {path}")


if __name__ == "__main__":
    main()
