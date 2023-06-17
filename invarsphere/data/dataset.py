from __future__ import annotations

import os

import ase
import torch
from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Data, Dataset

from .convert import atoms2graphdata, graphdata2atoms, set_properties


class BaseGraphDataset(Dataset):
    def __init__(self, cutoff: float, save_dir: str):
        super().__init__()
        self.cutoff = cutoff
        self.save_dir = save_dir

    def len(self) -> int:
        raise NotImplementedError

    def get(self, idx: int) -> Data:
        raise NotImplementedError


class List2GraphDataset(BaseGraphDataset):
    """Convert a list of structures or atoms into a graph dataset.

    During the conversion, the following information is computed:
    - Index of neighboring atoms within the cutoff radius considering PBC.
    - Lattice shift values taking into account PBC. (necessary to calculate inter atomic distances with atom in different cell images)
    - Rotation matrix of each node.
    """  # noqa: E501

    def __init__(
        self,
        atoms_list: list[ase.Atoms],
        y_values: dict[str, list[int | float | str | ndarray | Tensor] | ndarray | Tensor],
        cutoff: float,
        save_dir: str,
        subtract_center_of_mass: bool = False,
        max_n_neighbor_basis: int = 4,
        basis_cutoff: float = 10,
        remove_batch_key: list[str] | None = None,
        preprocess: bool = True,
        inmemory: bool = False,
    ):
        """
        Args:
           atoms_list (list[ase.Atoms]): list of ase.Atoms object
           y_values (dict[str, list[int | float | str | ndarray | Tensor] | ndarray | Tensor]): dict of physical properties. The key is the name of the property, and the value is the corresponding value of the property.
           cutoff (float): the cutoff radius for computing the neighbor list
           save_dir (str): the directory to save the dataset
           subtract_center_of_mass (bool, optional): Whether to subtract the center of mass from the cartesian coordinates. Defaults to `False`.
           max_n_neighbor_basis (int): The maximum number of neighbors to be considered for each atom. Defaults to `4`.
           basis_cutoff (float): The cutoff radius for computing the neighbor basis. Defaults to `10`.
           remove_batch_key (list[str] | None, optional): List of property names that do not add dimension for batch. Defaults to `None`.
           preprocess (bool, optional): Whether to preprocess the dataset. Defaults to `True`.
           inmemory (bool, optional): Whether to load the dataset into memory. Defaults to `False`.
        """  # noqa: E501
        super().__init__(cutoff, save_dir)
        self.subtract_center_of_mass = subtract_center_of_mass
        self.max_n_neighbor_basis = max_n_neighbor_basis
        self.basis_cutoff = basis_cutoff
        self.remove_batch_key = remove_batch_key
        if preprocess:
            self._preprocess(atoms_list, y_values)
        self.inmemory = inmemory
        if inmemory:
            self._data_list = [None for _ in range(self.len())]

    def _preprocess(
        self,
        atoms_list: list[ase.Atoms],
        y_values: dict[str, list[int | float | str | ndarray | Tensor] | ndarray | Tensor],
    ):
        for i, at in enumerate(atoms_list):
            assert isinstance(at, ase.Atoms)
            data = atoms2graphdata(
                at,
                self.subtract_center_of_mass,
                self.cutoff,
                self.max_n_neighbor_basis,
                self.basis_cutoff,
            )
            for k, v in y_values.items():
                add_batch = True
                if self.remove_batch_key is not None and k in self.remove_batch_key:
                    add_batch = False
                set_properties(data, k, v[i], add_batch)
            torch.save(data, f"{self.save_dir}/{i}.pt")

    def len(self) -> int:
        length = len(os.listdir(self.save_dir))
        if length == 0:
            raise ValueError("The dataset is empty. Please set preprocess=True.")
        return length

    def get(self, idx: int) -> Data:
        if not self.inmemory:
            return torch.load(f"{self.save_dir}/{idx}.pt")
        if idx >= len(self._data_list):
            raise IndexError("index out of range")
        if self._data_list[idx] is None:
            try:
                self._data_list[idx] = torch.load(f"{self.save_dir}/{idx}.pt")
            except FileNotFoundError:
                raise IndexError("index out of range, please set proper index or preprocess=True")
        return self._data_list[idx]

    def get_atoms(self, idx: int) -> ase.Atoms:
        return graphdata2atoms(self.get(idx))
