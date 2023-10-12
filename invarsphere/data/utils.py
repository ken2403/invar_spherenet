from __future__ import annotations

import logging

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Data


def get_triple_edge_idx(edge_src: ndarray, n_nodes: int) -> ndarray:
    first_col = edge_src.reshape(-1, 1)
    all_indices = np.arange(n_nodes).reshape(1, -1)
    n_bond_per_atom = np.count_nonzero(first_col == all_indices, axis=0)
    n_triple_i = n_bond_per_atom * (n_bond_per_atom - 1)
    n_triple = np.sum(n_triple_i)
    triple_edge_idx = np.empty((n_triple, 2), dtype=np.int64)

    start = 0
    cs = 0
    for n in n_bond_per_atom:
        if n > 0:
            """triple_bond_indices is generated from all pair permutations of
            atom indices.

            The
            numpy version below does this with much greater efficiency. The equivalent slow
            code is:
            ```
            for j, k in itertools.permutations(range(n), 2):
                triple_bond_indices[index] = [start + j, start + k]
            ```
            """
            r = np.arange(n)
            x, y = np.meshgrid(r, r)
            c = np.stack([y.ravel(), x.ravel()], axis=1)
            final = c[c[:, 0] != c[:, 1]]
            triple_edge_idx[start : start + (n * (n - 1)), :] = final + cs  # noqa: E203
            start += n * (n - 1)
            cs += n

    return triple_edge_idx


def nearest_vec2rot_mat_3d(nearest_vec: ndarray) -> ndarray:
    # nearest_vec is matrix with coordinate components arranged in the column direction
    if nearest_vec.shape != (3, 2):
        errm = f"nearest_vec must be (3, 2) shape, but got {nearest_vec.shape}"
        logging.error(errm)
        raise ValueError(errm)
    nearest_vec = np.array(nearest_vec, dtype=np.float64)
    # get first basis vector
    nearest_vec[:, 0] /= np.linalg.norm(nearest_vec[:, 0])
    # get second basis vector
    nearest_vec[:, 1] -= np.dot(nearest_vec[:, 0], nearest_vec[:, 1]) * nearest_vec[:, 0]
    nearest_vec[:, 1] /= np.linalg.norm(nearest_vec[:, 1])
    # get third basis vector
    cross = np.cross(nearest_vec[:, 0], nearest_vec[:, 1])
    cross /= np.linalg.norm(cross)
    # concatenate
    q = np.concatenate([nearest_vec, cross[:, np.newaxis]], axis=1)
    return q  # (3, 3) shape


def full_linked_graph(n_nodes: int) -> tuple[ndarray, ndarray]:
    # get all pair permutations of atom indices
    r = np.arange(n_nodes)
    x, y = np.meshgrid(r, r)
    ind = np.stack([y.ravel(), x.ravel()], axis=1)
    # remove self edge
    ind = ind[ind[:, 0] != ind[:, 1]]

    shift = np.zeros((ind.shape[0], 3), np.float32)

    return ind.T, shift


def _set_data(
    data: Data,
    k: str,
    v: int | float | ndarray | Tensor,
    add_dim: bool,
    add_batch: bool,
    dtype: torch.dtype,
):
    if add_dim:
        val = torch.tensor([v], dtype=dtype)
    else:
        val = torch.tensor(v, dtype=dtype)
    data[k] = val.unsqueeze(0) if add_batch else val


def set_properties(
    data: Data,
    k: str,
    v: int | float | str | ndarray | Tensor,
    add_batch: bool = True,
):
    if isinstance(v, int):
        _set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=torch.long)
    elif isinstance(v, float):
        _set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=torch.float32)
    elif isinstance(v, str):
        data[k] = v
    elif len(v.shape) == 0:
        # for 0-dim array
        if isinstance(v, ndarray):
            dtype = torch.long if v.dtype == int else torch.float32
        elif isinstance(v, Tensor):
            dtype = v.dtype
        else:
            raise ValueError(f"Unknown type of {v}")
        _set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=dtype)
    else:
        # for array-like
        if isinstance(v, ndarray):
            dtype = torch.long if v.dtype == int else torch.float32
        elif isinstance(v, Tensor):
            dtype = v.dtype
        else:
            raise ValueError(f"Unknown type of {v}")
        _set_data(data, k, v, add_dim=False, add_batch=add_batch, dtype=dtype)
