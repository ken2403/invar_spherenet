from __future__ import annotations

from typing import Any

import pytest
import torch

from invarsphere.model import InvarianceSphereNet

param_align_initial_weight = [
    (True),
    (False),
]


@pytest.mark.parametrize("align_initial_weight", param_align_initial_weight)
def test_align_initial_weight(align_initial_weight: bool):
    n_blocks = 10
    model = InvarianceSphereNet(
        emb_size_atom=10,
        emb_size_edge=10,
        emb_size_rbf=10,
        emb_size_cbf=10,
        emb_size_sbf=10,
        n_neighbor_basis=10,
        n_blocks=n_blocks,
        n_targets=1,
        max_n=3,
        max_l=4,
        weight_init="glorotorthogonal",
        align_initial_weight=align_initial_weight,
    )

    int_block_dict: dict[str, Any] = {}
    out_block_dict: dict[str, Any] = {}
    for k, v in model.state_dict().items():
        if "int_blocks" in k:
            weight_key = ".".join(k.split(".")[2:])
            if int_block_dict.get(weight_key) is None:
                int_block_dict[weight_key] = [v]
            else:
                int_block_dict[weight_key].append(v)
        if "out_blocks" in k:
            weight_key = ".".join(k.split(".")[2:])
            if out_block_dict.get(weight_key) is None:
                out_block_dict[weight_key] = [v]
            else:
                out_block_dict[weight_key].append(v)

    # check weight value is same but id is different
    for k, v in int_block_dict.items():
        assert len(v) == n_blocks
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if align_initial_weight:
                    assert torch.allclose(v[i], v[j])
                assert id(v[i]) != id(v[j])

    for k, v in out_block_dict.items():
        assert len(v) == n_blocks + 1
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if align_initial_weight:
                    assert torch.allclose(v[i], v[j])
                assert id(v[i]) != id(v[j])
