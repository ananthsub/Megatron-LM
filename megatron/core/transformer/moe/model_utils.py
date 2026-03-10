# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


def initialize_moe_layer_metadata(layers: Iterable[object]) -> int:
    """Assign dense-to-sparse layer indices and propagate the total MoE layer count."""
    moe_layers = [layer for layer in layers if getattr(layer, "is_moe_layer", False)]

    for moe_layer_idx, layer in enumerate(moe_layers):
        assigned = layer.set_moe_layer_number(moe_layer_idx)
        if assigned is False:
            raise RuntimeError(
                f"MoE layer numbering should happen exactly once, but layer {layer!r} "
                f"rejected index {moe_layer_idx}."
            )

    num_moe_layers = len(moe_layers)
    for layer in moe_layers:
        layer.set_num_moe_layers(num_moe_layers)

    return num_moe_layers


def prepare_moe_topk_routing_replay_indices(
    moe_topk_routing_replay_indices: torch.Tensor | None,
    *,
    batch_size: int,
    seq_length: int,
    sequence_parallel: bool,
    scatter_to_sequence_parallel: bool,
    tp_group: Any,
    scatter_fn: Callable[..., torch.Tensor],
) -> torch.Tensor | None:
    """Normalize replay indices to `[seq, batch, num_moe_layers, topk]` and shard if needed."""
    if moe_topk_routing_replay_indices is None:
        return None

    if moe_topk_routing_replay_indices.dim() != 4:
        raise ValueError(
            "Expected `moe_topk_routing_replay_indices` to have shape "
            f"`[batch, seq, num_moe_layers, topk]` or `[seq, batch, num_moe_layers, topk]`, "
            f"but got {tuple(moe_topk_routing_replay_indices.shape)}."
        )

    leading_dims = tuple(moe_topk_routing_replay_indices.shape[:2])
    batch_first_dims = (batch_size, seq_length)
    seq_first_dims = (seq_length, batch_size)

    if leading_dims == batch_first_dims:
        moe_topk_routing_replay_indices = (
            moe_topk_routing_replay_indices.transpose(0, 1).contiguous()
        )
    elif leading_dims != seq_first_dims:
        raise ValueError(
            "Leading replay-index dimensions do not match either the batch-first or "
            f"sequence-first layout: got {leading_dims}, expected {batch_first_dims} "
            f"or {seq_first_dims}."
        )

    if sequence_parallel and scatter_to_sequence_parallel:
        moe_topk_routing_replay_indices = scatter_fn(
            moe_topk_routing_replay_indices, group=tp_group
        ).clone()

    return moe_topk_routing_replay_indices
