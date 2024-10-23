from __future__ import annotations

import functools
import os
import random
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from chgnet import utils
from chgnet.graph import CrystalGraph, CrystalGraphConverter
from chgnet.data.dataset import collate_graphs

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing_extensions import Self
    from chgnet import TrainTask

warnings.filterwarnings("ignore")
TORCH_DTYPE = torch.float32


def get_train_val_test_loader(
    dataset: Dataset,
    *,
    batch_size: int = 64,
    indices_train: list,
    indices_val: list,
    indices_test: list,
    return_test: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Randomly partition a dataset into train, val, test loaders.

    Args:
        dataset (Dataset): The dataset to partition.
        batch_size (int): The batch size for the data loaders
            Default = 64
        train_ratio (float): The ratio of the dataset to use for training
            Default = 0.8
        val_ratio (float): The ratio of the dataset to use for validation
            Default: 0.1
        return_test (bool): Whether to return a test data loader
            Default = True
        num_workers (int): The number of worker processes for loading the data
            see torch Dataloader documentation for more info
            Default = 0
        pin_memory (bool): Whether to pin the memory of the data loaders
            Default: True

    Returns:
        train_loader, val_loader and optionally test_loader
    """
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=SubsetRandomSampler(indices=indices_train),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=SubsetRandomSampler(indices=indices_val),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if return_test:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            sampler=SubsetRandomSampler(indices=indices_test),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader
    return train_loader, val_loader