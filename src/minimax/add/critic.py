"""Environment critic for ADD: predicts categorical return distribution."""

from typing import Sequence

import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from minimax.add.unet import sinusoidal_embedding, ResBlock, AttentionBlock, AttentionPool2d
from minimax.add.diffusion import DiffusionSchedule, theta_to_diffusion, q_sample


# --- Categorical return distribution utilities ---


def returns_to_categorical(
    returns: jnp.ndarray,
    num_bins: int = 100,
    min_return: float = 0.0,
    max_return: float = 1.0,
) -> jnp.ndarray:
    """Two-hot encode returns into a categorical distribution. (B,) -> (B, num_bins)."""
    bin_width = (max_return - min_return) / num_bins
    bin_idx = jnp.floor((returns - min_return) / bin_width).astype(jnp.int32)
    bin_idx = jnp.clip(bin_idx, 0, num_bins - 2)
    remainder = (returns - min_return - bin_idx * bin_width) / bin_width
    remainder = jnp.clip(remainder, 0.0, 1.0)
    target = jnp.zeros((returns.shape[0], num_bins))
    target = target.at[jnp.arange(returns.shape[0]), bin_idx].add(1.0 - remainder)
    target = target.at[jnp.arange(returns.shape[0]), bin_idx + 1].add(remainder)
    return target


def _return_to_two_hot(
    ret: jnp.ndarray,
    num_bins: int = 100,
    min_return: float = 0.0,
    max_return: float = 1.0,
) -> jnp.ndarray:
    """Two-hot encode a single scalar return. () -> (num_bins,)."""
    bin_width = (max_return - min_return) / num_bins
    ret = jnp.clip(ret, min_return, max_return - 1e-8)
    bin_idx = jnp.floor((ret - min_return) / bin_width).astype(jnp.int32)
    bin_idx = jnp.clip(bin_idx, 0, num_bins - 2)
    remainder = (ret - min_return - bin_idx * bin_width) / bin_width
    remainder = jnp.clip(remainder, 0.0, 1.0)
    target = jnp.zeros(num_bins)
    target = target.at[bin_idx].add(1.0 - remainder)
    target = target.at[bin_idx + 1].add(remainder)
    return target


def rollout_to_target(
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    num_bins: int = 100,
    min_return: float = 0.0,
    max_return: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build a distributional critic target from one worker's rollout.

    Scans through timesteps accumulating undiscounted reward. On each
    episode completion (done=True), two-hot encodes the episode return and
    adds it to a running distribution. The final distribution is normalised
    by the episode count, giving an equal-weight mixture over all completed
    episodes.

    Args:
        rewards: (n_steps,)
        dones:   (n_steps,) uint8

    Returns:
        target:      (num_bins,) distributional target (sums to 1, or all
                     zeros if no episodes completed)
        n_episodes:  () int32, number of completed episodes
        total_return: () float32, sum of all episode returns
    """
    def scan_fn(carry, step):
        cumulative, target_accum, ep_count, total_ret = carry
        reward, done = step

        cumulative = cumulative + reward
        done_bool = done.astype(jnp.bool_)

        two_hot = _return_to_two_hot(
            cumulative, num_bins, min_return, max_return,
        )
        target_accum = jnp.where(done_bool, target_accum + two_hot, target_accum)
        ep_count = ep_count + done_bool.astype(jnp.int32)
        total_ret = jnp.where(done_bool, total_ret + cumulative, total_ret)
        cumulative = jnp.where(done_bool, 0.0, cumulative)

        return (cumulative, target_accum, ep_count, total_ret), None

    init = (
        jnp.float32(0.0),
        jnp.zeros(num_bins),
        jnp.int32(0),
        jnp.float32(0.0),
    )
    (_, target_accum, n_episodes, total_return), _ = jax.lax.scan(
        scan_fn, init, (rewards, dones),
    )
    target = target_accum / jnp.maximum(n_episodes, 1)
    return target, n_episodes, total_return


def batch_rollout_to_targets(
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    num_bins: int = 100,
    min_return: float = 0.0,
    max_return: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build distributional critic targets for all workers via vmap.

    Args:
        rewards: (n_steps, n_workers)
        dones:   (n_steps, n_workers) uint8

    Returns:
        targets:     (n_workers, num_bins)
        n_episodes:  (n_workers,)
        mean_return: () scalar mean return across all episodes and workers
    """
    targets, n_episodes, total_returns = jax.vmap(
        lambda r, d: rollout_to_target(r, d, num_bins, min_return, max_return),
        in_axes=(1, 1),
    )(rewards, dones)

    mean_return = total_returns.sum() / jnp.maximum(n_episodes.sum(), 1)
    return targets, n_episodes, mean_return


def categorical_mean(
    logits: jnp.ndarray,
    num_bins: int = 100,
    min_return: float = 0.0,
    max_return: float = 1.0,
) -> jnp.ndarray:
    """Expected return from categorical logits. (B, num_bins) -> (B,)."""
    bin_width = (max_return - min_return) / num_bins
    prob = jax.nn.softmax(logits, axis=-1)
    support = jnp.linspace(min_return, max_return - bin_width, num_bins) + bin_width / 2
    return jnp.sum(prob * support, axis=-1)


def categorical_cvar(
    logits: jnp.ndarray,
    alpha: float = 0.15,
    num_bins: int = 100,
    min_return: float = 0.0,
    max_return: float = 1.0,
) -> jnp.ndarray:
    """CVaR from right tail (approximates max). (B, num_bins) -> (B,)."""
    bin_width = (max_return - min_return) / num_bins
    prob = jax.nn.softmax(logits, axis=-1)
    cumprob = jnp.cumsum(prob, axis=-1)
    x = jnp.clip(cumprob - (1.0 - alpha), 0.0, prob)
    bin_left = jnp.linspace(min_return, max_return - bin_width, num_bins)
    cvar = jnp.sum(x * (bin_left + bin_width * (1.0 - x / (2.0 * prob + 1e-8))), axis=-1) / alpha
    return cvar


def regret(
    logits: jnp.ndarray,
    alpha: float = 0.15,
    num_bins: int = 100,
    min_return: float = 0.0,
    max_return: float = 1.0,
) -> jnp.ndarray:
    """Regret = CVaR - mean. Differentiable w.r.t. logits. (B, num_bins) -> (B,)."""
    return categorical_cvar(logits, alpha, num_bins, min_return, max_return) - \
        categorical_mean(logits, num_bins, min_return, max_return)


# --- Environment critic model ---


class EnvCritic(nn.Module):
    """Predicts categorical return distribution from a level image.

    Matches guided-diffusion's EncoderUNetModel: UNet encoder with FiLM
    conditioning (scale_shift_norm), learned downsampling (resblock_updown),
    and CLIP-style attention pooling.
    """
    base_channels: int = 128
    channel_mults: Sequence[int] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attention_resolutions: Sequence[int] = (16, 8, 4)
    num_head_channels: int = 64
    dropout_rate: float = 0.0
    num_bins: int = 100

    @nn.compact
    def __call__(
        self, x: jax.Array, t: jax.Array, *, train: bool = False
    ) -> jax.Array:
        base_ch = self.base_channels
        emb_channels = 4 * base_ch

        temb = sinusoidal_embedding(t, base_ch)
        temb = nn.Dense(emb_channels)(temb)
        temb = nn.silu(temb)
        temb = nn.Dense(emb_channels)(temb)

        def use_attention(spatial_size: int) -> bool:
            return spatial_size in self.attention_resolutions

        def num_heads(channels: int) -> int:
            return max(1, channels // self.num_head_channels)

        def res_block(ch_out, down=False):
            return ResBlock(
                out_channels=ch_out,
                emb_channels=emb_channels,
                dropout_rate=self.dropout_rate,
                use_scale_shift_norm=True,
                down=down,
            )

        def attn_block(channels):
            return AttentionBlock(num_heads=num_heads(channels))

        ch = int(self.channel_mults[0] * base_ch)
        h = nn.Conv(ch, kernel_size=(3, 3), padding="SAME")(x)

        current_res = x.shape[1]
        for level, mult in enumerate(self.channel_mults):
            ch_out = int(mult * base_ch)
            for _ in range(self.num_res_blocks):
                h = res_block(ch_out)(h, temb, train=train)
                if use_attention(current_res):
                    h = attn_block(ch_out)(h)
            if level != len(self.channel_mults) - 1:
                h = res_block(ch_out, down=True)(h, temb, train=train)
                current_res //= 2

        mid_ch = int(self.channel_mults[-1] * base_ch)
        h = res_block(mid_ch)(h, temb, train=train)
        h = attn_block(mid_ch)(h)
        h = res_block(mid_ch)(h, temb, train=train)

        # Attention pooling -> output logits.
        h = nn.GroupNorm(num_groups=min(32, mid_ch))(h)
        h = nn.silu(h)
        h = AttentionPool2d(
            spatial_dim=current_res,
            embed_dim=mid_ch,
            num_head_channels=self.num_head_channels,
            output_dim=self.num_bins,
        )(h)
        return h


# --- Replay buffer ---


class CriticBuffer:
    """Fixed-size circular buffer holding (theta, target_dist) pairs."""

    def __init__(
        self,
        capacity: int,
        theta_shape: tuple[int, ...] = (16, 16, 3),
        num_bins: int = 100,
    ):
        self.capacity = capacity
        self.thetas = np.zeros((capacity, *theta_shape), dtype=np.float32)
        self.targets = np.zeros((capacity, num_bins), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, thetas: np.ndarray, targets: np.ndarray) -> None:
        """Add a batch of (theta, target) pairs."""
        B = thetas.shape[0]
        if self.ptr + B <= self.capacity:
            self.thetas[self.ptr:self.ptr + B] = thetas
            self.targets[self.ptr:self.ptr + B] = targets
            self.ptr = self.ptr + B
        else:
            first = self.capacity - self.ptr
            self.thetas[self.ptr:] = thetas[:first]
            self.targets[self.ptr:] = targets[:first]
            rest = B - first
            self.thetas[:rest] = thetas[first:]
            self.targets[:rest] = targets[first:]
            self.ptr = rest
        self.size = min(self.size + B, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Sample a random batch."""
        idx = rng.choice(self.size, size=batch_size, replace=False)
        return self.thetas[idx], self.targets[idx]

    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity


# --- Training ---


def make_critic_train_step(critic_model: EnvCritic, optimizer: optax.GradientTransformation):
    """Create a JIT-compiled critic training step (call once at setup)."""

    @jax.jit
    def critic_step(params, opt_state, x_t, t, targets):
        def loss_fn(p):
            logits = critic_model.apply(p, x_t, t, train=True)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            return -jnp.sum(targets * log_probs, axis=-1).mean()

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    return critic_step


def train_critic(
    critic_step,
    critic_params,
    opt_state: optax.OptState,
    buffer: CriticBuffer,
    rng: jax.Array,
    schedule: DiffusionSchedule,
    num_iterations: int = 5,
    batch_size: int = 128,
) -> tuple:
    """Train the critic on buffered (theta, target) pairs.

    Returns:
        (critic_params, opt_state, mean_loss)
    """
    np_rng = np.random.default_rng(int(jax.random.randint(rng, (), 0, 2**30)))
    T = schedule.betas.shape[0]
    total_loss = 0.0

    for i in range(num_iterations):
        rng, rng_t, rng_noise = jax.random.split(rng, 3)

        theta_batch, target_batch = buffer.sample(batch_size, np_rng)
        theta_batch = jnp.array(theta_batch)
        target_batch = jnp.array(target_batch)

        x_0 = theta_to_diffusion(theta_batch)
        t = jax.random.randint(rng_t, (batch_size,), 0, T)
        noise = jax.random.normal(rng_noise, x_0.shape)
        x_t = q_sample(x_0, t, noise, schedule)

        critic_params, opt_state, loss = critic_step(
            critic_params, opt_state, x_t, t, target_batch,
        )
        total_loss += float(loss)

    mean_loss = total_loss / num_iterations
    return critic_params, opt_state, mean_loss
