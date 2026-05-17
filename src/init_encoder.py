r"""InitEncoder_i - shallow per-modality 1-D convolutional feature extractor.

Maps  x_i \in R^{C_i x L}  ->  h_i \in R^{L' x C_i'}
The dimension transposition (channels-first input -> time-first output) is handled inside the module so downstream code always receives (L', C_i').
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class InitEncoder(nn.Module):
    """One per modality (weights are NOT shared across modalities).

    out_channels : int
        C_i' - width of the shallow representation.
    kernel_size : int
        Temporal convolution kernel size.
    stride : int
        Temporal stride (controls downsampling L -> L').
    """

    out_channels: int
    kernel_size: int = 5
    stride: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        """
        x : jnp.ndarray of shape (batch, C_i, L)
            Channels-first input for one modality.
        train : bool
            Forwarded to BatchNorm (controls running-stats update).

        Returns
        h : jnp.ndarray of shape (batch, L', C_i')
            Time-first hidden representation.
        """
        # (batch, C_i, L) -> (batch, L, C_i) - Flax Conv expects channels-last
        x = jnp.transpose(x, (0, 2, 1))

        # 1-D conv  (batch, L, C_i) -> (batch, L', C_i')
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding="SAME",
            name="conv1",
        )(x)

        # BatchNorm (axis=-1 normalises over the channel dimension)
        x = nn.BatchNorm(use_running_average=not train, name="batch_norm1")(x)

        # SiLU activation
        x = nn.silu(x)

        return x  # output shape: (B, L', C_i')

