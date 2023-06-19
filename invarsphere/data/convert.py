from __future__ import annotations

import logging

import ase
import numpy as np
import torch
from ase.data import atomic_masses
from ase.neighborlist import neighbor_list
from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Data

from .keys import KEYS, GraphKeys


def atoms2graphdata(
    atoms: ase.Atoms,
    subtract_center_of_mass: bool,
    cutoff: float,
    max_n_neighbor_basis: int,
    basis_cutoff: float,
) -> Data:
    """Convert one `ase.Atoms` object to `torch_geometric.data.Data` with edge
    index information include pbc.

    Args:
        atoms (ase.Atoms): one atoms object

    Returns:
        data (torch_geometric.data.Data): one Data object with edge information include pbc and the rotation matrix.
    """
    if subtract_center_of_mass:
        masses = np.array(atomic_masses[atoms.numbers])
        pos = atoms.positions
        atoms.positions -= (masses[:, None] * pos).sum(0) / masses.sum()

    # edge information including pbc
    edge_src, edge_dst, dist, edge_vec, edge_shift = neighbor_list(
        "ijdDS",
        a=atoms,
        cutoff=basis_cutoff,
        self_interaction=False,
    )

    idx_s = []
    idx_t = []
    shift = []
    if max_n_neighbor_basis:
        check_index = int(max_n_neighbor_basis**0.5 + 1)
        rm = []
        basis_node_idx = []
        basis_edge_idx1 = []
        basis_edge_idx2 = []
        # basis_edge_idx3 = []

    n_ind = 0
    unique = np.unique(edge_src)
    for i in unique:
        center_mask = edge_src == i
        dist_i = dist[center_mask]
        sorted_ind = np.argsort(dist_i)
        dist_mask = (dist_i <= cutoff)[sorted_ind]
        # center_mask to retrieve information on central atom i
        # reorder by soreted_ind in order of distance
        # extract only the information within the cutoff radius with dist_mask
        idx_s_i = edge_src[center_mask][sorted_ind][dist_mask]
        idx_s.append(idx_s_i)
        idx_t.append(edge_dst[center_mask][sorted_ind][dist_mask])
        shift.append(edge_shift[center_mask][sorted_ind][dist_mask])

        # rotation matrix
        if max_n_neighbor_basis:
            # search neighbor basis only in cutoff radius
            triple_edge_idx = _get_triple_edge_idx(idx_s_i[:check_index], i + 1)
            i1 = 0
            cnt = 0
            while cnt < max_n_neighbor_basis:
                # nearest_vec has a coordinate component in the row direction
                try:
                    first_vec = edge_vec[center_mask][sorted_ind][triple_edge_idx[i1][0]]
                    second_vec = edge_vec[center_mask][sorted_ind][triple_edge_idx[i1][1]]
                except IndexError:
                    logging.info(f"only {cnt} neighbor_basis are found for {i}th atom in {atoms.symbols}")
                    break
                # coordinate component in the column direction.
                nearest_vec = np.stack([first_vec, second_vec], axis=1)
                q = _schmidt_3d(nearest_vec)
                # If two vectors are not first order independent, the q value become nan
                if np.isnan(q).any():
                    i1 += 1
                    continue

                # Transpose the original coordinates so that they can be transformed by matrix product
                rm.append(q.T)
                # The index of the edge is sorted,
                # so it stores how many indexes are in the range (i.e., i1)
                basis_node_idx.append(i)
                basis_edge_idx1.append(triple_edge_idx[i1][0] + n_ind)
                basis_edge_idx2.append(triple_edge_idx[i1][1] + n_ind)
                cnt += 1
                i1 += 1

            # keep n_ind for basis edge_index
            n_ind += idx_s[-1].shape[0]

    edge_src = np.concatenate(idx_s, axis=0)
    edge_dst = np.concatenate(idx_t, axis=0)
    edge_shift = np.concatenate(shift, axis=0)
    if max_n_neighbor_basis:
        rotation_matrix_arr = np.array(rm)
        basis_node_idx_arr = np.array(basis_node_idx)
        basis_edge_idx1_arr = np.array(basis_edge_idx1)
        basis_edge_idx2_arr = np.array(basis_edge_idx2)
        # basis_edge_idx3_arr = np.array(basis_edge_idx3)

    # edge_index order is "source_to_target"
    data = Data(edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0))
    # node info
    data[GraphKeys.Pos] = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    data[GraphKeys.Z] = torch.tensor(atoms.numbers, dtype=torch.long)
    # edge info
    data[GraphKeys.Edge_shift] = torch.tensor(edge_shift, dtype=torch.float32)

    if max_n_neighbor_basis:
        data[GraphKeys.Rot_mat] = torch.tensor(rotation_matrix_arr, dtype=torch.float32)
        data[GraphKeys.Basis_node_idx] = torch.tensor(basis_node_idx_arr, dtype=torch.long)
        data[GraphKeys.Basis_edge_idx1] = torch.tensor(basis_edge_idx1_arr, dtype=torch.long)
        data[GraphKeys.Basis_edge_idx2] = torch.tensor(basis_edge_idx2_arr, dtype=torch.long)
        # data[GraphKeys.Basis_edge_idx3] = torch.tensor(basis_edge_idx3_arr, dtype=torch.long)

    # graph info
    data[GraphKeys.Lattice] = torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0)
    data[GraphKeys.PBC] = torch.tensor(atoms.pbc, dtype=torch.long).unsqueeze(0)
    data[GraphKeys.Neighbors] = torch.tensor([edge_dst.shape[0]])

    return data


def _get_triple_edge_idx(edge_src: ndarray, n_nodes: int) -> ndarray:
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


def _schmidt_3d(nearest_vec: ndarray) -> ndarray:
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


def graphdata2atoms(data: Data) -> ase.Atoms:
    """Convert one `torch_geometric.data.Data` object to `ase.Atoms`.

    Args:
        data (torch_geometric.data.Data): one graph data object with edge information include pbc
    Returns:
        atoms (ase.Atoms): one Atoms object
    """
    pos = data[GraphKeys.Pos].numpy()
    atom_num = data[GraphKeys.Z].numpy()
    ce = data[GraphKeys.Lattice].numpy()[0]  # remove batch dimension
    pbc = data[GraphKeys.PBC].numpy()[0]  # remove batch dimension
    info = {}
    for k, v in data.items():
        if k not in KEYS:
            info[k] = v
    atoms = ase.Atoms(numbers=atom_num, positions=pos, pbc=pbc, cell=ce, info=info)
    return atoms
