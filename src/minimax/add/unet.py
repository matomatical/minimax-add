"""UNet denoising model for DDPM on 16x16x3 maze level images."""

import math
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


def sinusoidal_embedding(t: jax.Array, dim: int, max_period: int = 10000) -> jax.Array:
    """Sinusoidal timestep embedding. Matches guided-diffusion convention."""
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
    )
    args = t[:, None].astype(jnp.float32) * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate(
            [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
        )
    return embedding


class ResBlock(nn.Module):
    """Residual block with timestep conditioning.

    Attributes:
        use_scale_shift_norm: If True, timestep embedding produces scale+shift
            for FiLM conditioning. If False, additive conditioning only.
        down: If True, apply 2x spatial downsampling inside this block
            (resblock_updown from guided-diffusion).
    """
    out_channels: int
    emb_channels: int
    dropout_rate: float = 0.0
    use_scale_shift_norm: bool = False
    down: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, temb: jax.Array, *, train: bool = False):
        # x: (B, H, W, C_in), temb: (B, emb_channels)
        C_in = x.shape[-1]
        C_out = self.out_channels

        h = nn.GroupNorm(num_groups=min(32, C_in))(x)
        h = nn.silu(h)
        if self.down:
            h = nn.avg_pool(h, (2, 2), strides=(2, 2))
            x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        h = nn.Conv(C_out, kernel_size=(3, 3), padding="SAME")(h)

        # Timestep projection.
        temb_proj = nn.silu(temb)
        if self.use_scale_shift_norm:
            temb_proj = nn.Dense(2 * C_out)(temb_proj)
            scale, shift = jnp.split(temb_proj[:, None, None, :], 2, axis=-1)
            h = nn.GroupNorm(num_groups=min(32, C_out))(h)
            h = h * (1 + scale) + shift
        else:
            temb_proj = nn.Dense(C_out)(temb_proj)
            h = h + temb_proj[:, None, None, :]
            h = nn.GroupNorm(num_groups=min(32, C_out))(h)

        h = nn.silu(h)
        h = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(h)
        h = nn.Conv(
            C_out,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(h)

        if C_in != C_out:
            x = nn.Conv(C_out, kernel_size=(1, 1))(x)

        return x + h


class AttentionBlock(nn.Module):
    """Multi-head self-attention over spatial positions."""
    num_heads: int

    @nn.compact
    def __call__(self, x: jax.Array):
        B, H, W, C = x.shape
        num_heads = self.num_heads

        h = nn.GroupNorm(num_groups=min(32, C))(x)

        # Reshape to sequence: (B, H*W, C)
        h = h.reshape(B, H * W, C)

        # QKV projection
        qkv = nn.Dense(3 * C)(h)  # (B, H*W, 3*C)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        head_dim = C // num_heads
        # Reshape for multi-head: (B, num_heads, H*W, head_dim)
        q = q.reshape(B, H * W, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, H * W, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, H * W, num_heads, head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(head_dim)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

        # Merge heads: (B, H*W, C)
        out = out.transpose(0, 2, 1, 3).reshape(B, H * W, C)

        # Output projection, zero-initialized.
        out = nn.Dense(
            C,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(out)

        # Residual and reshape back to spatial.
        return (x.reshape(B, H * W, C) + out).reshape(B, H, W, C)


class AttentionPool2d(nn.Module):
    """CLIP-style attention pooling (from Dhariwal & Nichol 2021).

    Prepends a learnable [CLS]-like mean token, adds positional embeddings,
    then runs one multi-head attention layer. Returns the [CLS] output,
    projected to output_dim.
    """
    spatial_dim: int
    embed_dim: int
    num_head_channels: int
    output_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        B, H, W, C = x.shape
        seq_len = H * W
        num_heads = C // self.num_head_channels

        x = x.reshape(B, seq_len, C)
        cls_token = x.mean(axis=1, keepdims=True)
        x = jnp.concatenate([cls_token, x], axis=1)  # (B, seq_len+1, C)

        pos_emb = self.param(
            "positional_embedding",
            nn.initializers.normal(stddev=1.0 / math.sqrt(C)),
            (seq_len + 1, C),
        )
        x = x + pos_emb[None, :, :]

        qkv = nn.Dense(3 * C)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        head_dim = C // num_heads
        q = q.reshape(B, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, num_heads, head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(head_dim)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

        out = out.transpose(0, 2, 1, 3).reshape(B, -1, C)
        out = nn.Dense(self.output_dim)(out[:, 0])  # CLS token only
        return out


class Downsample(nn.Module):
    """Strided 2x2 convolution for spatial downsampling."""
    out_channels: int

    @nn.compact
    def __call__(self, x: jax.Array):
        return nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
        )(x)


class Upsample(nn.Module):
    """Nearest-neighbor 2x upsample followed by 3x3 conv."""
    out_channels: int

    @nn.compact
    def __call__(self, x: jax.Array):
        B, H, W, C = x.shape
        x = jax.image.resize(x, (B, H * 2, W * 2, C), method="nearest")
        return nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            padding="SAME",
        )(x)


class UNet(nn.Module):
    """UNet denoising model for DDPM, following guided-diffusion conventions.

    Attributes:
        base_channels: Base channel count (model_channels).
        channel_mults: Per-level channel multipliers.
        num_res_blocks: Number of residual blocks per encoder level.
        attention_resolutions: Spatial resolutions at which to apply attention.
        dropout_rate: Dropout probability.
        out_channels: Output channels (defaults to 3).
    """
    base_channels: int = 128
    channel_mults: Sequence[int] = (1, 2, 2, 2)
    num_res_blocks: int = 3
    attention_resolutions: Sequence[int] = (16, 8)
    dropout_rate: float = 0.0
    use_scale_shift_norm: bool = True
    out_channels: int = 3
    num_heads: Optional[int] = None  # None = legacy max(1, channels // 64); int = fixed (matches PyTorch reference)

    @nn.compact
    def __call__(
        self, x: jax.Array, t: jax.Array, *, train: bool = False
    ) -> jax.Array:
        """
        Args:
            x: Noised image, shape (B, 16, 16, 3).
            t: Integer diffusion timesteps, shape (B,).
            train: Whether to enable dropout.

        Returns:
            Predicted noise, shape (B, 16, 16, 3).
        """
        base_ch = self.base_channels
        emb_channels = 4 * base_ch

        # --- Timestep embedding ---
        temb = sinusoidal_embedding(t, base_ch)
        temb = nn.Dense(emb_channels)(temb)
        temb = nn.silu(temb)
        temb = nn.Dense(emb_channels)(temb)

        def use_attention(spatial_size: int) -> bool:
            return spatial_size in self.attention_resolutions

        def num_heads(channels: int) -> int:
            if self.num_heads is not None:
                return self.num_heads
            return max(1, channels // 64)

        def res_block(ch_out):
            return ResBlock(
                out_channels=ch_out,
                emb_channels=emb_channels,
                dropout_rate=self.dropout_rate,
                use_scale_shift_norm=self.use_scale_shift_norm,
            )

        def attn_block(channels):
            return AttentionBlock(num_heads=num_heads(channels))

        # --- Input projection ---
        ch = int(self.channel_mults[0] * base_ch)
        h = nn.Conv(ch, kernel_size=(3, 3), padding="SAME")(x)

        # Track all skip connection outputs for the decoder.
        skips = [h]

        # --- Encoder ---
        current_res = x.shape[1]  # starts at 16
        for level, mult in enumerate(self.channel_mults):
            ch_out = int(mult * base_ch)
            for _ in range(self.num_res_blocks):
                h = res_block(ch_out)(h, temb, train=train)
                if use_attention(current_res):
                    h = attn_block(ch_out)(h)
                skips.append(h)
            # Downsample after all levels except the last.
            if level != len(self.channel_mults) - 1:
                h = Downsample(out_channels=ch_out)(h)
                current_res //= 2
                skips.append(h)

        # --- Middle ---
        mid_ch = int(self.channel_mults[-1] * base_ch)
        h = res_block(mid_ch)(h, temb, train=train)
        h = attn_block(mid_ch)(h)
        h = res_block(mid_ch)(h, temb, train=train)

        # --- Decoder ---
        for level, mult in reversed(list(enumerate(self.channel_mults))):
            ch_out = int(mult * base_ch)
            for i in range(self.num_res_blocks + 1):
                # Concatenate skip connection.
                skip = skips.pop()
                h = jnp.concatenate([h, skip], axis=-1)
                h = res_block(ch_out)(h, temb, train=train)
                if use_attention(current_res):
                    h = attn_block(ch_out)(h)
                # Upsample after the last resblock of each level (except 0).
                if level != 0 and i == self.num_res_blocks:
                    h = Upsample(out_channels=ch_out)(h)
                    current_res *= 2

        assert len(skips) == 0

        # --- Output ---
        h = nn.GroupNorm(num_groups=min(32, h.shape[-1]))(h)
        h = nn.silu(h)
        h = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(h)
        return h
