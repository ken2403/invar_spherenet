from __future__ import annotations

import abc
import logging
import pathlib
import pickle

import ase
import numpy as np
import torch
from ase.data import atomic_masses
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data

from .keys import KEYS, GraphKeys
from .utils import (
    full_linked_graph,
    get_triple_edge_idx,
    nearest_vec2rot_mat_3d,
    set_properties,
)


class BaseDataConverter(abc.ABC):
    def __init__(self, cutoff: float, save_dir: str | pathlib.Path):
        self.cutoff = cutoff
        if isinstance(save_dir, str):
            self.save_dir = pathlib.Path(save_dir)
        else:
            self.save_dir = save_dir
        if not self.save_dir.exists():
            self.save_dir.mkdir(exist_ok=False)

    @abc.abstractmethod
    def convert(self, atoms_info):
        raise NotImplementedError


class ListDataConverter(BaseDataConverter):
    def __init__(
        self,
        cutoff: float,
        save_dir: str | pathlib.Path,
        subtract_center_of_mass: bool = False,
        max_neighbors: int = 32,
        max_n_neighbor_basis: int = 4,
        remove_batch_key: list[str] | None = None,
    ):
        super().__init__(cutoff, save_dir)

        self.subtract_center_of_mass = subtract_center_of_mass
        self.max_neighbors = max_neighbors
        self.max_n_neighbor_basis = max_n_neighbor_basis
        self.remove_batch_key = remove_batch_key

    def convert(self, atoms_list: list[ase.Atoms]):
        for i, at in enumerate(atoms_list):
            assert isinstance(at, ase.Atoms)
            data = atoms2graphdata(
                at,
                self.subtract_center_of_mass,
                self.cutoff,
                self.max_neighbors,
                self.max_n_neighbor_basis,
            )
            for k, v in at.info.items():
                add_batch = True
                if self.remove_batch_key is not None and k in self.remove_batch_key:
                    add_batch = False
                set_properties(data, k, v, add_batch)
            torch.save(data, f"{self.save_dir}/{i}.pt")


class FilesDataConverter(BaseDataConverter):
    def __init__(
        self,
        cutoff: float,
        save_dir: str | pathlib.Path,
        subtract_center_of_mass: bool = False,
        max_neighbors: int = 32,
        max_n_neighbor_basis: int = 4,
        remove_batch_key: list[str] | None = None,
    ):
        super().__init__(cutoff, save_dir)

        self.subtract_center_of_mass = subtract_center_of_mass
        self.max_neighbors = max_neighbors
        self.max_n_neighbor_basis = max_n_neighbor_basis
        self.remove_batch_key = remove_batch_key

    def convert(self, atoms_directory: str | pathlib.Path):
        if isinstance(atoms_directory, str):
            atoms_directory = pathlib.Path(atoms_directory)
        for i, at_file in enumerate(atoms_directory.iterdir()):
            with open(at_file, "rb") as f:
                at = pickle.load(f)
            assert isinstance(at, ase.Atoms)
            data = atoms2graphdata(
                at,
                self.subtract_center_of_mass,
                self.cutoff,
                self.max_neighbors,
                self.max_n_neighbor_basis,
            )
            for k, v in at.info.items():
                add_batch = True
                if self.remove_batch_key is not None and k in self.remove_batch_key:
                    add_batch = False
                set_properties(data, k, v, add_batch)
            torch.save(data, f"{self.save_dir}/{i}.pt")


# Main transformer to create edge information and rotation matrix
def atoms2graphdata(
    atoms: ase.Atoms,
    subtract_center_of_mass: bool,
    cutoff: float,
    max_neighbors: int,
    max_n_neighbor_basis: int,
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
        cutoff=cutoff,
        self_interaction=False,
    )

    idx_s = []
    idx_t = []
    shift = []
    if max_n_neighbor_basis:
        check_index = int(max_n_neighbor_basis**0.5 + 1)
        rm = []
        basis_node_idx = []

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
        idx_s_i = edge_src[center_mask][sorted_ind][dist_mask][:max_neighbors]
        idx_s.append(idx_s_i)
        idx_t.append(edge_dst[center_mask][sorted_ind][dist_mask][:max_neighbors])
        shift.append(edge_shift[center_mask][sorted_ind][dist_mask][:max_neighbors])

        # rotation matrix
        if max_n_neighbor_basis:
            # search neighbor basis only in cutoff radius
            triple_edge_idx = get_triple_edge_idx(idx_s_i[:check_index], i + 1)
            i1 = 0
            cnt = 0
            while cnt < max_n_neighbor_basis:
                # nearest_vec has a coordinate component in the row direction
                try:
                    first_vec = edge_vec[center_mask][sorted_ind][dist_mask][triple_edge_idx[i1][0]]
                    second_vec = edge_vec[center_mask][sorted_ind][dist_mask][triple_edge_idx[i1][1]]
                except IndexError:
                    logging.info(f"only {cnt} neighbor_basis are found for {i}th atom in {atoms.symbols}")
                    break
                # coordinate component in the column direction.
                nearest_vec = np.stack([first_vec, second_vec], axis=1)
                q = nearest_vec2rot_mat_3d(nearest_vec)
                # If two vectors are not first order independent, the q value become nan
                if np.isnan(q).any():
                    i1 += 1
                    continue

                # Transpose the original coordinates so that they can be transformed by matrix product
                rm.append(q.T)
                # The index of the edge is sorted,
                # so it stores how many indexes are in the range (i.e., i1)
                basis_node_idx.append(i)
                cnt += 1
                i1 += 1

    if len(idx_s) > 0:
        edge_src = np.concatenate(idx_s, axis=0)
        edge_dst = np.concatenate(idx_t, axis=0)
        edge_shift = np.concatenate(shift, axis=0)
    else:
        logging.warning(f"no neighbor is found in {atoms.symbols}. Make fully linked graph.")
        edge, edge_shift = full_linked_graph(atoms.numbers.shape[0])
        edge_src, edge_dst = edge[0], edge[1]

    if max_n_neighbor_basis:
        rotation_matrix_arr = np.array(rm)
        basis_node_idx_arr = np.array(basis_node_idx)

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

    # graph info
    data[GraphKeys.Lattice] = torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0)
    data[GraphKeys.PBC] = torch.tensor(atoms.pbc, dtype=torch.long).unsqueeze(0)
    data[GraphKeys.Neighbors] = torch.tensor([edge_dst.shape[0]])

    return data


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
