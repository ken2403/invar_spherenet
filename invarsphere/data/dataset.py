from __future__ import annotations

import ase
from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Data, Dataset

from .convert import atoms2graphdata, graphdata2atoms, set_properties


class BaseGraphDataset(Dataset):
    def __init__(self, cutoff: float):
        super().__init__()

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
        super().__init__(cutoff)
        self.max_neighbors = max_neighbors
        self.subtract_center_of_mass = subtract_center_of_mass
        self.n_neighbor_basis = n_neighbor_basis
        self.basis_cutoff = basis_cutoff
        self.graph_data_list: list[Data] = []
        self.remove_batch_key = remove_batch_key
        self._preprocess(structures, y_values)

    def _preprocess(
        self,
        atoms: list[ase.Atoms],
        y_values: dict[str, list[int | float | str | ndarray | Tensor] | ndarray | Tensor],
    ):
        for i, at in enumerate(atoms):
            assert isinstance(at, ase.Atoms)
            data = atoms2graphdata(
                at,
                self.subtract_center_of_mass,
                self.cutoff,
                self.max_neighbors,
                self.n_neighbor_basis,
                self.basis_cutoff,
            )
            for k, v in y_values.items():
                add_batch = True
                if self.remove_batch_key is not None and k in self.remove_batch_key:
                    add_batch = False
                set_properties(data, k, v[i], add_batch)
            self.graph_data_list.append(data)

    def len(self) -> int:
        if len(self.graph_data_list) == 0:
            raise ValueError("The dataset is empty.")
        return len(self.graph_data_list)

    def get(self, idx: int) -> Data:
        return self.graph_data_list[idx]

    def get_atoms(self, idx: int) -> ase.Atoms:
        return graphdata2atoms(self.graph_data_list[idx])
