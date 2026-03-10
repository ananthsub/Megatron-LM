# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.model_utils import (
    initialize_moe_layer_metadata,
    prepare_moe_topk_routing_replay_indices,
)


class _DummyLayer:
    def __init__(self, is_moe_layer):
        self.is_moe_layer = is_moe_layer
        self.moe_layer_numbers = []
        self.num_moe_layers = []

    def set_moe_layer_number(self, moe_layer_idx):
        self.moe_layer_numbers.append(moe_layer_idx)
        return True

    def set_num_moe_layers(self, num_moe_layers):
        self.num_moe_layers.append(num_moe_layers)


def test_initialize_moe_layer_metadata_numbers_only_top_level_moe_layers():
    dense_a = _DummyLayer(is_moe_layer=False)
    moe_a = _DummyLayer(is_moe_layer=True)
    dense_b = _DummyLayer(is_moe_layer=False)
    moe_b = _DummyLayer(is_moe_layer=True)

    num_moe_layers = initialize_moe_layer_metadata([dense_a, moe_a, dense_b, moe_b])

    assert num_moe_layers == 2
    assert dense_a.moe_layer_numbers == []
    assert dense_b.moe_layer_numbers == []
    assert moe_a.moe_layer_numbers == [0]
    assert moe_b.moe_layer_numbers == [1]
    assert moe_a.num_moe_layers == [2]
    assert moe_b.num_moe_layers == [2]


def test_prepare_moe_topk_routing_replay_indices_normalizes_batch_first_layout():
    replay_indices = torch.arange(2 * 3 * 4 * 2).view(2, 3, 4, 2)

    prepared = prepare_moe_topk_routing_replay_indices(
        replay_indices,
        batch_size=2,
        seq_length=3,
        sequence_parallel=False,
        scatter_to_sequence_parallel=False,
        tp_group=None,
        scatter_fn=lambda tensor, group=None: tensor,
    )

    assert prepared.shape == (3, 2, 4, 2)
    assert torch.equal(prepared, replay_indices.transpose(0, 1).contiguous())


def test_prepare_moe_topk_routing_replay_indices_preserves_seq_first_layout():
    replay_indices = torch.arange(3 * 2 * 4 * 2).view(3, 2, 4, 2)

    prepared = prepare_moe_topk_routing_replay_indices(
        replay_indices,
        batch_size=2,
        seq_length=3,
        sequence_parallel=False,
        scatter_to_sequence_parallel=False,
        tp_group=None,
        scatter_fn=lambda tensor, group=None: tensor,
    )

    assert prepared.shape == (3, 2, 4, 2)
    assert torch.equal(prepared, replay_indices)


def test_prepare_moe_topk_routing_replay_indices_scatter_happens_after_normalization():
    replay_indices = torch.arange(2 * 4 * 3 * 2).view(2, 4, 3, 2)
    scatter_inputs = []

    def scatter_fn(tensor, group=None):
        scatter_inputs.append(tensor.clone())
        return tensor[:2]

    prepared = prepare_moe_topk_routing_replay_indices(
        replay_indices,
        batch_size=2,
        seq_length=4,
        sequence_parallel=True,
        scatter_to_sequence_parallel=True,
        tp_group="tp-group",
        scatter_fn=scatter_fn,
    )

    expected = replay_indices.transpose(0, 1).contiguous()
    assert len(scatter_inputs) == 1
    assert torch.equal(scatter_inputs[0], expected)
    assert torch.equal(prepared, expected[:2])


def test_prepare_moe_topk_routing_replay_indices_rejects_mismatched_shapes():
    replay_indices = torch.zeros(5, 6, 2, 1, dtype=torch.long)

    with pytest.raises(ValueError, match="Leading replay-index dimensions"):
        prepare_moe_topk_routing_replay_indices(
            replay_indices,
            batch_size=2,
            seq_length=3,
            sequence_parallel=False,
            scatter_to_sequence_parallel=False,
            tp_group=None,
            scatter_fn=lambda tensor, group=None: tensor,
        )
