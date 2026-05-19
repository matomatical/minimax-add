"""Stage 1: Train DDPM on procedurally generated maze levels.

Runs on whatever devices are visible to JAX (`jax.local_device_count()`):
  - `tpu-device 0 python -m minimax.add.train_diffusion`        → single-chip
  - `tpu-device 0,1,2,3 python -m minimax.add.train_diffusion`  → 4-chip pmap

`--batch_size` is the *effective* (global) batch; per-device batch is
`batch_size // n_devices`. Must divide evenly.
"""

import argparse
import os
import time
import pickle
from collections import deque
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from flax.training import train_state

from minimax.add.theta import GRID_SIZE, sample_random_theta, decode_level
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


def save_checkpoint(path, state, ema_params, step, unet_kwargs, data_kwargs):
    """Unreplicate-then-pickle. `state` and `ema_params` are device-replicated."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state_host = jax_utils.unreplicate(state)
    ema_host = jax_utils.unreplicate(ema_params)
    data = {
        "params": jax.device_get(state_host.params),
        "ema_params": jax.device_get(ema_host),
        "opt_state": jax.device_get(state_host.opt_state),
        "step": step,
        "unet_kwargs": unet_kwargs,
        "data_kwargs": data_kwargs,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_checkpoint(path, state):
    """Load single-device state and replicate across visible devices."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    state_host = state.replace(
        params=jax.device_put(data["params"]),
        opt_state=jax.device_put(data["opt_state"]),
        step=data["step"],
    )
    ema_host = jax.device_put(data["ema_params"])
    state = jax_utils.replicate(state_host)
    ema_params = jax_utils.replicate(ema_host)
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
    parser.add_argument("--num_heads", type=int, default=None,
                        help="Fix UNet attention head count (matches PyTorch reference, =4). "
                             "Default None preserves legacy max(1, channels // 64).")
    parser.add_argument("--min_walls", type=int, default=0,
                        help="Lower bound (inclusive) of uniform wall-count sampling.")
    parser.add_argument("--max_walls", type=int, default=60,
                        help="Upper bound (exclusive) of uniform wall-count sampling. "
                             "Default 60 matches the ADD paper. Wall indices are sampled "
                             "with replacement, so realised wall counts are coupon-collector "
                             "skewed below the nominal n_walls draw, especially near 169.")
    args = parser.parse_args()
    if not 0 <= args.min_walls < args.max_walls <= GRID_SIZE * GRID_SIZE:
        raise ValueError(
            f"need 0 <= min_walls ({args.min_walls}) < max_walls ({args.max_walls}) <= 169"
        )

    n_devices = jax.local_device_count()
    if args.batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size {args.batch_size} must divide n_devices {n_devices}"
        )
    per_device_batch = args.batch_size // n_devices

    print(f"JAX devices: {jax.devices()} (n={n_devices}, per_device_batch={per_device_batch})")
    rng = jax.random.PRNGKey(args.seed)
    schedule = make_schedule()

    unet_kwargs = {}
    if args.num_heads is not None:
        unet_kwargs["num_heads"] = args.num_heads
    print(f"UNet kwargs (non-default): {unet_kwargs}")
    model = UNet(**unet_kwargs)

    data_kwargs = {"min_walls": args.min_walls, "max_walls": args.max_walls}
    print(f"Data kwargs: {data_kwargs}")
    sample_theta_fn = partial(
        sample_random_theta,
        min_walls=args.min_walls,
        max_walls=args.max_walls,
    )

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

    # Replicate state and EMA params across devices.
    state = jax_utils.replicate(state)
    ema_params = jax_utils.replicate(ema_params)

    if args.resume:
        # load_checkpoint returns already-replicated state/EMA.
        state, ema_params, start_step = load_checkpoint(args.resume, jax_utils.unreplicate(state))
        print(f"Resumed from step {start_step}")

    # --- Pmapped training functions (close over model, schedule, config) ---

    @partial(jax.pmap, axis_name="device")
    def train_step(state, batch, rng):
        def loss_fn(params):
            return compute_loss(model.apply, params, batch, rng, schedule)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = jax.lax.pmean(grads, axis_name="device")
        loss = jax.lax.pmean(loss, axis_name="device")
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.pmap
    def gen_batch(rng):
        # rng shape (n_devices, 2). Per-device: split → vmap.
        return jax.vmap(sample_theta_fn)(jax.random.split(rng, per_device_batch))

    ema_rate = args.ema_rate

    @jax.pmap
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

    def evaluate(replicated_params, rng):
        # Run sampling on a single device using the unreplicated copy.
        params = jax_utils.unreplicate(replicated_params)
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
        rng_batch_d = jax.random.split(rng_batch, n_devices)
        rng_train_d = jax.random.split(rng_train, n_devices)
        batch = gen_batch(rng_batch_d)
        state, loss = train_step(state, batch, rng_train_d)
        ema_params = update_ema(ema_params, state.params)
        # `loss` is shape (n_devices,) with identical values across devices (pmean).
        losses.append(float(loss[0]))

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
            save_checkpoint(path, state, ema_params, step + 1, unet_kwargs, data_kwargs)
            print(f"  saved {path}")

    path = os.path.join(args.ckpt_dir, "final.pkl")
    save_checkpoint(path, state, ema_params, args.total_steps, unet_kwargs, data_kwargs)
    total_time = time.time() - t0
    print(f"Training complete in {total_time/3600:.1f}h. Saved {path}")


if __name__ == "__main__":
    main()
