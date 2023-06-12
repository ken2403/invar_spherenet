from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy import ndarray
from torch import Tensor

from invarsphere.utils.repeat_tensor import block_repeat, block_repeat_each

param_block_repeat = [
    # t, block_size, repeats, dim, return_index, expected
    # tensor test
    (
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([2, 2]),
        torch.tensor([2, 3]),
        0,
        False,
        torch.tensor([1, 2, 1, 2, 3, 4, 3, 4, 3, 4]),
    ),
    # index test
    (
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([2, 2]),
        torch.tensor([2, 3]),
        0,
        True,
        torch.tensor([0, 1, 0, 1, 2, 3, 2, 3, 2, 3]),
    ),
    # ndarray test
    (
        torch.tensor([1, 2, 3, 4]),
        np.array([2, 2]),
        np.array([2, 3]),
        0,
        False,
        torch.tensor([1, 2, 1, 2, 3, 4, 3, 4, 3, 4]),
    ),
    # ndarray index test
    (
        torch.tensor([1, 2, 3, 4]),
        np.array([2, 2]),
        np.array([2, 3]),
        0,
        True,
        torch.tensor([0, 1, 0, 1, 2, 3, 2, 3, 2, 3]),
    ),
    # dim test
    (
        torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        torch.tensor([2, 2]),
        torch.tensor([2, 3]),
        1,
        False,
        torch.tensor([[1, 2, 1, 2, 3, 4, 3, 4, 3, 4], [5, 6, 5, 6, 7, 8, 7, 8, 7, 8]]),
    ),
    # dim index test
    (
        torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        torch.tensor([2, 2]),
        torch.tensor([2, 3]),
        1,
        True,
        torch.tensor([0, 1, 0, 1, 2, 3, 2, 3, 2, 3]),
    ),
    # multi dimensional tensor, dim=0 test
    (
        torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        torch.tensor([1, 2]),
        torch.tensor([2, 1]),
        0,
        False,
        torch.tensor(
            [
                [1, 2, 3],
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        ),
    ),
    # multi dimensional tensor, dim=0 index test
    (
        torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        torch.tensor([1, 2]),
        torch.tensor([2, 1]),
        0,
        True,
        torch.tensor([0, 0, 1, 2]),
    ),
    # multi dimensional tensor, dim=1 test
    (
        torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        torch.tensor([2, 1]),
        torch.tensor([2, 3]),
        1,
        False,
        torch.tensor(
            [
                [1, 2, 1, 2, 3, 3, 3],
                [4, 5, 4, 5, 6, 6, 6],
                [7, 8, 7, 8, 9, 9, 9],
            ]
        ),
    ),
    # multi dimensional tensor, dim=1 index test
    (
        torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        torch.tensor([2, 1]),
        torch.tensor([2, 3]),
        1,
        True,
        torch.tensor([0, 1, 0, 1, 2, 2, 2]),
    ),
    # dimension value erro3
    (
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([1]),
        2,
        False,
        torch.tensor([1]),
    ),
    # block size value error
    (
        torch.tensor([1, 2, 3]),
        torch.tensor([1]),
        torch.tensor([1]),
        0,
        False,
        torch.tensor([1, 2, 3]),
    ),
]


@pytest.mark.parametrize("t, block_size, repeats, dim, return_index, expected", param_block_repeat)
def test_block_repeat(
    t: Tensor,
    block_size: ndarray | Tensor,
    repeats: ndarray | Tensor,
    dim: int,
    return_index: bool,
    expected: Tensor,
):
    if dim != 0 and dim != 1:
        with pytest.raises(ValueError) as e:
            out = block_repeat(t, block_size, repeats, dim, return_index)
        assert "Only dim 0 and 1 supported" in str(e.value)
        return

    if t.size(dim) != block_size.sum():
        with pytest.raises(ValueError) as e:
            out = block_repeat(t, block_size, repeats, dim, return_index)
        assert "Block sizes do not match tensor size" in str(e.value)
        return

    out = block_repeat(t, block_size, repeats, dim, return_index)
    if not return_index:
        assert len(t.size()) == len(out.size())
    assert torch.allclose(out, expected)


param_block_repeat_each = [
    # t, block_size, repeats, dim, return_index, expected
    # tensor test
    (
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([2, 2]),
        torch.tensor([2, 3]),
        False,
        torch.tensor([1, 1, 2, 2, 3, 3, 3, 4, 4, 4]),
    ),
    # index test
    (
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([2, 2]),
        torch.tensor([2, 3]),
        True,
        torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3]),
    ),
    # ndarray test
    (
        torch.tensor([1, 2, 3, 4]),
        np.array([2, 2]),
        np.array([2, 3]),
        False,
        torch.tensor([1, 1, 2, 2, 3, 3, 3, 4, 4, 4]),
    ),
    # ndarray index test
    (
        torch.tensor([1, 2, 3, 4]),
        np.array([2, 2]),
        np.array([2, 3]),
        True,
        torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3]),
    ),
]


@pytest.mark.parametrize("t, block_size, repeats, return_index, expected", param_block_repeat_each)
def test_block_repeat_each(
    t: Tensor,
    block_size: ndarray | Tensor,
    repeats: ndarray | Tensor,
    return_index: bool,
    expected: Tensor,
):
    out = block_repeat_each(t, block_size, repeats, return_index)
    if not return_index:
        assert len(t.size()) == len(out.size())
    assert torch.allclose(out, expected)
