"""Bridge native MPS operations into torch.compile on PyTorch 2.8.

The MPS SDPA operator has no FakeTensor implementation in 2.8. An opaque
custom operator supplies its output metadata while executing the original
attention kernel. Repetition-window detection also stays native to avoid Metal
codegen's boolean reduction bugs. These inference-only bridges preserve the
operations' math and do not fall back to CPU.
"""

import torch
from torch import Tensor
from torch.nn import functional as F


@torch.library.custom_op("fish_speech::mps_sdpa", mutates_args=(), device_types="mps")
def mps_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None,
    dropout_p: float,
    is_causal: bool,
) -> Tensor:
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    ).contiguous()


@mps_sdpa.register_fake
def _mps_sdpa_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None,
    dropout_p: float,
    is_causal: bool,
) -> Tensor:
    return query.new_empty((*query.shape[:-1], value.shape[-1]))


@torch.library.custom_op(
    "fish_speech::mps_in_window", mutates_args=(), device_types="mps"
)
def mps_in_window(history: Tensor, token: Tensor) -> Tensor:
    # PyTorch 2.8 Metal codegen cannot lower a symbolic any() reduction, and
    # its static any() result carries an invalid dtype into bitwise operations.
    return (history == token).any()


@mps_in_window.register_fake
def _mps_in_window_fake(history: Tensor, token: Tensor) -> Tensor:
    return history.new_empty((), dtype=torch.bool)
