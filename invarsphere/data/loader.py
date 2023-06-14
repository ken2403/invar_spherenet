from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter

from .keys import GraphKeys


class EdgeBasisCollater:
    def __init__(self, follow_batch: list[str] | None, exclude_keys: list[str] | None):
        if follow_batch is None:
            follow_batch = []
        follow_batch += [
            GraphKeys.Basis_edge_idx1,
            GraphKeys.Basis_edge_idx2,
            GraphKeys.Basis_edge_idx3,
            GraphKeys.Edge_nb_idx,
            GraphKeys.Nb_edge_idx,
        ]
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        elem = batch[0]
        # custom loader for basis edge index
        if isinstance(elem, BaseData):
            # node_index increment
            data = Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
            return data

        # other is same as torch_geometric.loader.DataLoader
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: {type(elem)}")


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Dataset | Sequence[BaseData] | DatasetAdapter,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: list[str] | None = None,
        exclude_keys: list[str] | None = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,  # type: ignore # Since mypy cannot determine dataset is instance of torch.data.Dataset
            batch_size,
            shuffle,
            collate_fn=EdgeBasisCollater(follow_batch, exclude_keys),
            **kwargs,
        )
