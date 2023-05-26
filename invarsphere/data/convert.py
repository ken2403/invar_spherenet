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
    max_neighbors: int,
    n_neighbor_basis: int = 0,
    basis_cutoff: float = 15.0,
    properties: dict[str, int | float | str | ndarray | Tensor] = {},
    remove_batch_key: set[str] = {""},
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
        cutoff=basis_cutoff if n_neighbor_basis else cutoff,
        self_interaction=False,
    )

    # only max_neighbors
    idx_s = np.zeros(1, dtype=int) - 100
    idx_t = np.zeros(1, dtype=int) - 100
    s = np.zeros((1, 3)) - 100
    if n_neighbor_basis:
        rm = np.zeros((1, n_neighbor_basis, 3, 3))
        basis_idx_1 = np.zeros((1, n_neighbor_basis), dtype=int) - 100
        basis_idx_2 = np.zeros((1, n_neighbor_basis), dtype=int) - 100

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
        # indexing to take out only the max_neighbor neighborhoods
        idx_s = np.concatenate([idx_s, edge_src[center_mask][sorted_ind][dist_mask][:max_neighbors]], axis=0)
        idx_t = np.concatenate([idx_t, edge_dst[center_mask][sorted_ind][dist_mask][:max_neighbors]], axis=0)
        s = np.concatenate([s, edge_shift[center_mask][sorted_ind][dist_mask][:max_neighbors]], axis=0)

        # rotation matrix
        if n_neighbor_basis:
            rm_atom = np.zeros((1, 3, 3))
            basis_idx_1_atom = np.zeros(1, dtype=int)
            basis_idx_2_atom = np.zeros(1, dtype=int)
            i1 = 0
            cnt = 0
            while cnt < n_neighbor_basis:
                # nearest_vec has a coordinate component in the row direction
                try:
                    nearest_vec = edge_vec[center_mask][sorted_ind[[i1, i1 + 1]]]
                except IndexError:
                    errm = f"Cannot generate {i1+1}th nearest neighbor coordinate system of {atoms}, please increase {basis_cutoff}"  # noqa: E501
                    logging.error(errm)
                    raise IndexError(errm)
                # Transpose to have a coordinate component in the column direction.
                nearest_vec = nearest_vec.T
                q = _schmidt_3d(nearest_vec)
                # If two vectors are not first order independent, the q value become nan
                while np.isnan(q).any():
                    i1 += 1
                    try:
                        nearest_vec = edge_vec[center_mask][sorted_ind[[i1, i1 + 1]]]
                    except IndexError:
                        errm = f"Cannot generate {i1+1}th nearest neighbor coordinate system of {atoms}, please increase {basis_cutoff}"  # noqa: E501
                        logging.error(errm)
                        raise IndexError(errm)
                    # Transpose to have a coordinate component in the column direction.
                    nearest_vec = nearest_vec.T
                    q = _schmidt_3d(nearest_vec)
                # Transpose the original coordinates so that they can be transformed by matrix product
                rm_atom = np.concatenate([rm_atom, q.T[np.newaxis, ...]], axis=0)
                # The index of the edge is sorted,
                # so it stores how many indexes are in the range (i.e., i1)
                basis_idx_1_atom = np.concatenate([basis_idx_1_atom, np.array([i1 + n_ind])], axis=0)
                basis_idx_2_atom = np.concatenate([basis_idx_2_atom, np.array([i1 + n_ind + 1])], axis=0)
                cnt += 1
                i1 += 1

            rm = np.concatenate([rm, rm_atom[1:][np.newaxis, ...]], axis=0)
            basis_idx_1 = np.concatenate([basis_idx_1, basis_idx_1_atom[1:][np.newaxis, ...]], axis=0)
            basis_idx_2 = np.concatenate([basis_idx_2, basis_idx_2_atom[1:][np.newaxis, ...]], axis=0)

            # keep n_ind for basis edge_index
            n_ind = idx_s.shape[0] - 1

    edge_src = idx_s[1:]
    edge_dst = idx_t[1:]
    edge_shift = s[1:]
    if n_neighbor_basis:
        rotation_matrix = rm[1:]
        basis_idx_1 = basis_idx_1[1:]
        basis_idx_2 = basis_idx_2[1:]

    # order is "source_to_target"
    data = Data(edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0))

    data[GraphKeys.Neighbors] = torch.tensor([edge_dst.shape[0]])
    data[GraphKeys.Pos] = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    data[GraphKeys.Z] = torch.tensor(atoms.numbers, dtype=torch.long)
    if n_neighbor_basis:
        data[GraphKeys.Rot_mat] = torch.tensor(rotation_matrix, dtype=torch.float32)
        data[GraphKeys.Basis_edge_idx1] = torch.tensor(basis_idx_1, dtype=torch.long)
        data[GraphKeys.Basis_edge_idx2] = torch.tensor(basis_idx_2, dtype=torch.long)
    # add batch dimension
    data[GraphKeys.PBC] = torch.tensor(atoms.pbc, dtype=torch.long).unsqueeze(0)
    data[GraphKeys.Lattice] = torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0)
    data[GraphKeys.Edge_shift] = torch.tensor(edge_shift, dtype=torch.float32)

    # set properties
    for k, v in properties.items():
        add_batch = True
        if remove_batch_key is not None and k in remove_batch_key:
            add_batch = False
        set_properties(data, k, v, add_batch)

    return data


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
