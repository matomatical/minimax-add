"""DDPM forward process, training loss, and DDIM deterministic sampling."""

from typing import Callable, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp


# --- Schedule ---


class DiffusionSchedule(NamedTuple):
    betas: jnp.ndarray              # (T,)
    alphas: jnp.ndarray             # (T,)
    alpha_bars: jnp.ndarray         # (T,)
    sqrt_alpha_bars: jnp.ndarray    # (T,)
    sqrt_one_minus_alpha_bars: jnp.ndarray  # (T,)


def make_schedule(
    T: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> DiffusionSchedule:
    # Compute in float64 with numpy for precision, then store as jax float32.
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return DiffusionSchedule(
        betas=jnp.array(betas, dtype=jnp.float32),
        alphas=jnp.array(alphas, dtype=jnp.float32),
        alpha_bars=jnp.array(alpha_bars, dtype=jnp.float32),
        sqrt_alpha_bars=jnp.array(np.sqrt(alpha_bars), dtype=jnp.float32),
        sqrt_one_minus_alpha_bars=jnp.array(np.sqrt(1.0 - alpha_bars), dtype=jnp.float32),
    )


# --- Diffusion space conversion ---


def theta_to_diffusion(theta: jnp.ndarray) -> jnp.ndarray:
    return 2.0 * theta - 1.0


def diffusion_to_theta(x: jnp.ndarray) -> jnp.ndarray:
    return (x + 1.0) / 2.0


# --- Helpers ---


def _extract(arr: jnp.ndarray, t: jnp.ndarray, shape: tuple) -> jnp.ndarray:
    """Index into 1-D array `arr` with batch of indices `t`, broadcast to `shape`."""
    vals = arr[t]  # (B,)
    return vals.reshape(t.shape[0], *((1,) * (len(shape) - 1)))


# --- Forward process ---


def q_sample(
    x_0: jnp.ndarray,
    t: jnp.ndarray,
    noise: jnp.ndarray,
    schedule: DiffusionSchedule,
) -> jnp.ndarray:
    """Sample x_t ~ q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise."""
    sqrt_ab = _extract(schedule.sqrt_alpha_bars, t, x_0.shape)
    sqrt_1m_ab = _extract(schedule.sqrt_one_minus_alpha_bars, t, x_0.shape)
    return sqrt_ab * x_0 + sqrt_1m_ab * noise


# --- Training loss ---


ModelFn = Callable  # (params, x_t, t) -> epsilon_pred


def training_loss(
    model_fn: ModelFn,
    params,
    x_0: jnp.ndarray,
    t: jnp.ndarray,
    noise: jnp.ndarray,
    schedule: DiffusionSchedule,
) -> jnp.ndarray:
    """MSE between true noise and predicted noise, averaged over batch."""
    x_t = q_sample(x_0, t, noise, schedule)
    eps_pred = model_fn(params, x_t, t)
    return jnp.mean((noise - eps_pred) ** 2)


def compute_loss(
    model_fn: ModelFn,
    params,
    batch_theta: jnp.ndarray,
    rng: jax.Array,
    schedule: DiffusionSchedule,
) -> jnp.ndarray:
    """Full training loss: convert theta to diffusion space, sample t and noise."""
    x_0 = theta_to_diffusion(batch_theta)
    B = x_0.shape[0]
    T = schedule.betas.shape[0]
    rng_t, rng_noise = jax.random.split(rng)
    t = jax.random.randint(rng_t, (B,), 0, T)
    noise = jax.random.normal(rng_noise, x_0.shape)
    return training_loss(model_fn, params, x_0, t, noise, schedule)


# --- DDIM sampling ---


def _make_ddim_timesteps(T: int, num_steps: int) -> jnp.ndarray:
    """Create DDIM sub-sequence of timesteps, from high to low.

    Returns array of shape (num_steps,) with integer timestep indices.
    """
    step_indices = jnp.linspace(0, T - 1, num_steps + 1, dtype=jnp.int32)
    timesteps = step_indices[1:][::-1]  # (num_steps,) from T-1 down
    return timesteps


def ddim_sample(
    model_fn: ModelFn,
    params,
    shape: tuple,
    rng: jax.Array,
    schedule: DiffusionSchedule,
    num_steps: int = 50,
) -> jnp.ndarray:
    """DDIM deterministic sampling (eta=0). Returns x_0 in [-1, 1]."""
    T = schedule.betas.shape[0]
    timesteps = _make_ddim_timesteps(T, num_steps)

    # alpha_bars extended: alpha_bar[-1] = 1.0 for the "clean" end.
    alpha_bars_ext = jnp.concatenate([jnp.array([1.0]), schedule.alpha_bars])
    # Now alpha_bars_ext[0] = 1.0, alpha_bars_ext[t+1] = alpha_bars[t].

    x = jax.random.normal(rng, shape)

    def body(i, x):
        t_cur = timesteps[i]
        # t_prev: the next timestep in the subsequence (one step closer to clean).
        # For the last step (i == num_steps - 1), t_prev index goes to -1,
        # but we use alpha_bars_ext so index 0 gives alpha_bar = 1.0.
        t_prev = jnp.where(i < num_steps - 1, timesteps[i + 1], -1)

        ab_t = alpha_bars_ext[t_cur + 1]
        ab_prev = alpha_bars_ext[t_prev + 1]

        # Broadcast t_cur to batch dimension for model call.
        B = shape[0]
        t_batch = jnp.full((B,), t_cur, dtype=jnp.int32)
        eps_pred = model_fn(params, x, t_batch)

        # Predict x_0 and clip.
        x0_pred = (x - jnp.sqrt(1.0 - ab_t) * eps_pred) / jnp.sqrt(ab_t)
        x0_pred = jnp.clip(x0_pred, -1.0, 1.0)

        # DDIM deterministic update (eta=0).
        x_prev = jnp.sqrt(ab_prev) * x0_pred + jnp.sqrt(1.0 - ab_prev) * eps_pred
        return x_prev

    x = jax.lax.fori_loop(0, num_steps, body, x)
    return x


def ddim_sample_theta(
    model_fn: ModelFn,
    params,
    shape: tuple,
    rng: jax.Array,
    schedule: DiffusionSchedule,
    num_steps: int = 50,
) -> jnp.ndarray:
    """DDIM sampling, returning result in [0, 1] (theta space)."""
    x = ddim_sample(model_fn, params, shape, rng, schedule, num_steps)
    return jnp.clip(diffusion_to_theta(x), 0.0, 1.0)
