from __future__ import annotations

import logging

import ase
import ase.neighborlist
import numpy as np
import torch
from ase.data import atomic_masses
from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Data, Dataset

from .keys import GraphKeys


class BaseGraphDataset(Dataset):
    def __init__(
        self,
        cutoff: float,
        max_neighbors: int = 32,
        subtract_center_of_mass: bool = False,
        n_neighbor_basis: int = 4,
        basis_cutoff: float = 15,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.subtract_center_of_mass = subtract_center_of_mass
        self.n_neighbor_basis = n_neighbor_basis
        self.basis_cutoff = basis_cutoff

    def len(self) -> int:
        raise NotImplementedError

    def get(self, idx: int) -> Data:
        raise NotImplementedError

    @classmethod
    def load_from_pickle(cls, load_pth: str):
        import pickle

        with open(load_pth, "rb") as f:
            return pickle.load(f)

    def save(self, save_pth: str):
        import pickle

        with open(save_pth, "wb") as f:
            pickle.dump(self, f)

    def _atoms2graphdata(self, atoms: ase.Atoms) -> Data:
        """Helper function to convert one `Atoms` object to
        `torch_geometric.data.Data` with edge index information include pbc.

        Args:
            atoms (ase.Atoms): one atoms object

        Returns:
            data (torch_geometric.data.Data): one Data object with edge information include pbc
        """
        if self.subtract_center_of_mass:
            masses = np.array(atomic_masses[atoms.numbers])
            pos = atoms.positions
            atoms.positions -= (masses[:, None] * pos).sum(0) / masses.sum()

        # edge information including pbc
        edge_src, edge_dst, dist, edge_vec, edge_shift = ase.neighborlist.neighbor_list(
            "ijdDS",
            a=atoms,
            cutoff=self.basis_cutoff if self.n_neighbor_basis else self.cutoff,
            self_interaction=False,
        )

        # only max_neighbors
        idx_i = np.zeros(1, dtype=int) - 100
        idx_j = np.zeros(1, dtype=int) - 100
        s = np.zeros((1, 3)) - 100
        if self.n_neighbor_basis:
            rm = np.zeros((1, self.n_neighbor_basis, 3, 3))

        unique = np.unique(edge_src)
        for i in unique:
            center_mask = edge_src == i
            dist_i = dist[center_mask]
            sorted_ind = np.argsort(dist_i)
            dist_mask = (dist_i <= self.cutoff)[sorted_ind]
            idx_i = np.concatenate([idx_i, edge_src[center_mask][dist_mask][: self.max_neighbors]], axis=0)
            idx_j = np.concatenate([idx_j, edge_dst[center_mask][dist_mask][: self.max_neighbors]], axis=0)
            s = np.concatenate([s, edge_shift[center_mask][dist_mask][: self.max_neighbors]], axis=0)
            # rotation matrix
            rm_atom = np.zeros((1, 3, 3))
            for i1 in range(self.n_neighbor_basis):
                try:
                    nearest_vec = edge_vec[center_mask][sorted_ind[[i1, i1 + 1]]]
                except IndexError:
                    errm = f"Cannot generate {i1+1}th nearest neighbor coordinate system of {atoms}, please increase {self.basis_cutoff}"  # noqa: E501
                    logging.error(errm)
                    raise IndexError(errm)
                q = self._get_rot_matrix(nearest_vec)
                rm_atom = np.concatenate([rm_atom, q.T[np.newaxis, ...]], axis=0)
            rm = np.concatenate([rm, rm_atom[1:][np.newaxis, ...]], axis=0)
        edge_src = idx_i[1:]
        edge_dst = idx_j[1:]
        edge_shift = s[1:]
        if self.n_neighbor_basis:
            rotation_matrix = rm[1:]

        # order is "source_to_target" i.e. [index_j, index_i]
        data = Data(edge_index=torch.stack([torch.LongTensor(edge_dst), torch.LongTensor(edge_src)], dim=0))

        data[GraphKeys.Neighbors] = torch.tensor([edge_dst.size(0)])
        data[GraphKeys.Pos] = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        data[GraphKeys.Z] = torch.tensor(atoms.numbers, dtype=torch.long)
        if self.n_neighbor_basis:
            data[GraphKeys.Rot_mat] = torch.tensor(rotation_matrix, dtype=torch.float32)
        # add batch dimension
        data[GraphKeys.PBC] = torch.tensor(atoms.pbc, dtype=torch.long).unsqueeze(0)
        data[GraphKeys.Lattice] = torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0)
        data[GraphKeys.Edge_shift] = torch.tensor(edge_shift, dtype=torch.float32)
        return data

    def _get_rot_matrix(self, nearest_vec: ndarray) -> ndarray:
        if nearest_vec.shape != (2, 3):
            errm = f"nearest_vec must be (2, 3) shape, but got {nearest_vec.shape}"
            logging.error(errm)
            raise ValueError(errm)
        cross = np.cross(nearest_vec[0], nearest_vec[1])
        cross /= np.linalg.norm(cross)
        q = np.concatenate([nearest_vec, cross], axis=0)
        return q  # (3, 3) shape

    def _graphdata2atoms(self, data: Data) -> ase.Atoms:
        """Helper function to convert one `torch_geometric.data.Data` object to
        `ase.Atoms`.

        Args:
            data (torch_geometric.data.Data): one graph data object with edge information include pbc
        Returns:
            atoms (ase.Atoms): one Atoms object
        """
        pos = data[GraphKeys.Pos].numpy()
        atom_num = data[GraphKeys.Z].numpy()
        ce = data[GraphKeys.Lattice].numpy()[0]  # remove batch dimension
        pbc = data[GraphKeys.PBC].numpy()[0]  # remove batch dimension
        atoms = ase.Atoms(numbers=atom_num, positions=pos, pbc=pbc, cell=ce)
        return atoms

    def _set_data(
        self,
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

    def _set_properties(
        self,
        data: Data,
        k: str,
        v: int | float | str | ndarray | Tensor,
        add_batch: bool = True,
    ):
        if isinstance(v, int):
            self._set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=torch.long)
        elif isinstance(v, float):
            self._set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=torch.float32)
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
            self._set_data(data, k, v, add_dim=True, add_batch=add_batch, dtype=dtype)
        else:
            # for array-like
            if isinstance(v, ndarray):
                dtype = torch.long if v.dtype == int else torch.float32
            elif isinstance(v, Tensor):
                dtype = v.dtype
            else:
                raise ValueError(f"Unknown type of {v}")
            self._set_data(data, k, v, add_dim=False, add_batch=add_batch, dtype=dtype)


class List2GraphDataset(BaseGraphDataset):
    """Convert a list of structures or atoms into a graph dataset.

    During the conversion, the following information is computed:
    - Index of neighboring atoms within the cutoff radius considering PBC.
    - Lattice shift values taking into account PBC. (necessary to calculate inter atomic distances with atom in different cell images)
    - Rotation matrix of each node.
    """  # noqa: E501

    def __init__(
        self,
        structures: list[ase.Atoms],
        y_values: dict[str, list[int | float | str | ndarray | Tensor] | ndarray | Tensor],
        cutoff: float,
        max_neighbors: int = 32,
        subtract_center_of_mass: bool = False,
        n_neighbor_basis: int = 4,
        basis_cutoff: float = 15,
        remove_batch_key: list[str] | None = None,
    ):
        """
        Args:
           structures (list[ase.Atoms]): list of ase.Atoms object
           y_values (dict[str, list[int | float | str | ndarray | Tensor] | ndarray | Tensor]): dict of physical properties. The key is the name of the property, and the value is the corresponding value of the property.
           cutoff (float): the cutoff radius for computing the neighbor list
           max_neighbors (int, optional): Threshold of neighboring atoms to be considered. Defaults to `32`.
           subtract_center_of_mass (bool, optional): Whether to subtract the center of mass from the cartesian coordinates. Defaults to `False`.
           n_neighbor_basis (int, optional): the number of the neighbor basis for each node. Defaults to `0`.
           basis_cutoff (float, optional): the cutoff radius for computing neighbor basis
           remove_batch_key (list[str] | None, optional): List of property names that do not add dimension for batch. Defaults to `None`.
        """  # noqa: E501
        super().__init__(cutoff, max_neighbors, subtract_center_of_mass, n_neighbor_basis, basis_cutoff)
        self.graph_data_list: list[Data] = []
        self.remove_batch_key = remove_batch_key
        self._preprocess(structures, y_values)

    def _preprocess(
        self,
        atoms: list[ase.Atoms],
        y_values: dict[str, list[int | float | str | ndarray | Tensor] | ndarray | Tensor],
    ):
        for i, at in enumerate(atoms):
            data = self._atoms2graphdata(at)
            for k, v in y_values.items():
                add_batch = True
                if self.remove_batch_key is not None and k in self.remove_batch_key:
                    add_batch = False
                self._set_properties(data, k, v[i], add_batch)
            self.graph_data_list.append(data)

    def len(self) -> int:
        if len(self.graph_data_list) == 0:
            raise ValueError("The dataset is empty.")
        return len(self.graph_data_list)

    def get(self, idx: int) -> Data:
        return self.graph_data_list[idx]

    def get_atoms(self, idx: int) -> ase.Atoms:
        return self._graphdata2atoms(self.graph_data_list[idx])
