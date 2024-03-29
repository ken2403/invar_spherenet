# from https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/gemnet/utils.py#L85

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch_scatter import segment_csr


def block_repeat(
    t: Tensor,
    block_size: np.ndarray | Tensor,
    repeats: np.ndarray | Tensor,
    dim: int = 0,
    return_index: bool = False,
) -> Tensor:
    """Repeat each block of the tensor separately.

    Args:
        t (torch.Tensor): Tensor to repeat
        block_size (np.ndarray | torch.Tensor): Size of each block to repeat
        repeats (np.ndarray | torch.Tensor): Number of times to repeat each block
        dim (int, optional): Dimension to repeat along. Defaults to `0`.
        return_index (bool, optional): Whether to return the indices used for. Defaults to `False`. If `True`, returns the indices only.

    Raises:
        ValueError: If block sizes do not match tensor size or if dim is not 0 and 1

    Example:
        t: [1,2,3,4,5,6,7]
        block_size: [2,3,2]
        repeats: [3,3,2]
        return: [
                    1, 2, 1, 2, 1, 2,
                    3, 4, 5, 3, 4, 5, 3, 4, 5,
                    6, 7, 6, 7,
                ]

        t: [1,2,3,4,5,6,7]
        block_size: [2,3,2]
        repeats: [3,3,2]
        return_index: True
        return: [
                    0, 1, 0, 1, 0, 1,
                    2, 3, 4, 2, 3, 4, 2, 3, 4,
                    5, 6, 5, 6,
                ]
    """  # noqa: E501
    if dim != 0 and dim != 1:
        raise ValueError("Only dim 0 and 1 supported")
    col_index = torch.arange(t.size(dim), device=t.device)
    if col_index.size(0) != block_size.sum():
        raise ValueError("Block sizes do not match tensor size")
    indices = []
    start = 0

    for i, b in enumerate(block_size):
        indices.append(torch.tile(col_index[start : start + b], (int(repeats[i].item()),)))  # noqa: E203
        start += b
    indices_tensor = torch.cat(indices, dim=0).to(t.device).long()
    if return_index:
        return indices_tensor
    else:
        return t[..., indices_tensor] if dim == 1 else t[indices_tensor]


def block_repeat_each(
    t: Tensor,
    block_size: np.ndarray | Tensor,
    repeats: np.ndarray | Tensor,
    return_index: bool = False,
) -> Tensor:
    """Repeat each block of the tensor separately. When repeating, repeat
    element by element. Corresponds to repetition in first dimension only.

    Args:
        t (torch.Tensor): Tensor to repeat
        block_size (np.ndarray | torch.Tensor): Size of each block to repeat
        repeats (np.ndarray | torch.Tensor): Number of times to repeat each block
        return_index (bool, optional): Whether to return the indices used for. Defaults to `False`. If `True`, returns the indices only.

    Example:
        t: [1,2,3,4,5,6,7]
        block_size: [2,3,2]
        repeats: [3,3,2]
        return: [
                    1, 1, 1, 2, 2, 2,
                    3, 3, 3, 4, 4, 4, 5, 5, 5,
                    6, 6, 7, 7,
                ]

        t: [1,2,3,4,5,6,7]
        block_size: [2,3,2]
        repeats: [3,2,3]
        return_index: True
        return: [
                    0, 0, 0, 1, 1, 1,
                    2, 2, 2, 3, 3, 3, 4, 4, 4,
                    5, 5, 6, 6,
                ]
    """  # noqa: E501
    # dim = 0 only
    col_index = torch.arange(t.size(0), device=t.device)
    indices = []
    start = 0

    for i, b in enumerate(block_size):
        indices.append(torch.repeat_interleave(col_index[start : start + b], repeats[i]))  # noqa: E203
        start += b
    indices_tensor = torch.cat(indices, dim=0).to(t.device).long()
    if return_index:
        return indices_tensor
    else:
        return t[indices_tensor]


def repeat_blocks(
    sizes,
    repeats,
    continuous_indexing=True,
    start_idx=0,
    block_inc=0,
    repeat_inc=0,
):
    """Repeat blocks of indices. Adapted from
    https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-
    repeat-blocks-of-consecutive-elements.

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Example:
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """  # noqa: E501
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(torch.arange(len(sizes), device=sizes.device), repeats)

    # Get total size of output array, as needed to initialize output indexing array
    N = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(block_inc[: r1[-1]], indptr, reduce="sum")
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    res = id_ar.cumsum(0)
    return res


def ragged_range(sizes: Tensor) -> Tensor:
    """Multiple concatenated ranges.

    Examples
    --------
        sizes = [1 4 2 3]
        Return: [0  0 1 2 3  0 1  0 1 2]
    """
    assert sizes.dim() == 1
    if sizes.sum() == 0:
        return sizes.new_empty(0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        sizes = torch.masked_select(sizes, sizes_nonzero)

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    id_steps = torch.ones(int(sizes.sum().item()), dtype=torch.long, device=sizes.device)
    id_steps[0] = 0
    insert_index = sizes[:-1].cumsum(0)
    insert_val = (1 - sizes)[:-1]

    # Assign index-offsetting values
    id_steps[insert_index] = insert_val

    # Finally index into input array for the group repeated o/p
    res = id_steps.cumsum(0)
    return res
