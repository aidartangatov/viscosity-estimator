from types import SimpleNamespace
from typing import List, Tuple, Union, Optional
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from quannet.utils import DEFAULT_CONFIG
from quannet.prepare import split_indices, generate_esp_grids


class QuanDataset(Dataset):
    def __init__(
        self,
        structure_paths: List[Union[str, Path]],
        artefacts_dir: Union[str, Path],
        target: Optional[List[float]] = None,
        args: SimpleNamespace = DEFAULT_CONFIG,
    ):
        self.structure_paths = structure_paths
        self.target = target
        self.artefacts_dir = artefacts_dir
        self.args = args
        self._prepare_dataset()

    def _prepare_dataset(self):
        self.structures = torch.FloatTensor(self.transforms(self.structure_paths))
        self.target = (
            torch.FloatTensor(self.target).view(-1, 1) if self.target is not None else torch.zeros(len(self.target), 1)
        )
        self.structure_ids = np.arange(len(self.target))

    def transforms(self, structure_paths: List[Union[str, Path]]) -> np.ndarray:
        sample_size = len(structure_paths) * self.args.num_augmentations
        data = np.zeros((sample_size, self.args.grid_dim, self.args.grid_dim, self.args.grid_dim))

        counter = 0
        for structure_path in structure_paths:
            structure_path = Path(structure_path)

            esp_grids = generate_esp_grids(
                str(structure_path),
                output_dir=self.artefacts_dir,
                grid_dim=self.args.grid_dim,
                grid_spacing=self.args.grid_spacing,
                shell_width=self.args.shell_width,
                num_augmentations=self.args.num_augmentations,
                processes=self.args.processes,
            )
            for grid in esp_grids:
                data[counter] = grid
                counter += 1

        # Convert format into [sample_size, channels, depth, height, width]
        return np.expand_dims(data, 1)

    def __getitem__(self, index):
        return (
            self.structures[index],
            self.target[index // self.args.num_augmentations],
        )

    def __len__(self):
        return len(self.structures)


class QuanSampler(Sampler):
    def __init__(self, structure_ids, batch_size):
        self.structure_ids = structure_ids
        self.batch_size = batch_size
        self.shuffled_indices = self.structure_shuffle(self.structure_ids)

    def __iter__(self):
        return iter(self.shuffled_indices)

    def __len__(self):
        return len(self.shuffled_indices)

    @staticmethod
    def structure_shuffle(structure_ids):
        unique_ids = np.unique(structure_ids)
        grouped_indices = {uid: list(np.where(structure_ids == uid)[0]) for uid in unique_ids}

        shuffled_indices = []
        while len(shuffled_indices) < len(structure_ids):
            batch_indices = []
            for uid in unique_ids:
                if len(grouped_indices[uid]) > 0:
                    chosen_index = np.random.choice(grouped_indices[uid])
                    batch_indices.append(chosen_index)
                    grouped_indices[uid].remove(chosen_index)
            shuffled_indices.extend(batch_indices)

        return shuffled_indices


def build_input(
    structure_paths: List[Union[str, Path]],
    target: List[float],
    artefacts_dir: Union[str, Path],
    train_val_split: bool = False,
    args: Optional[SimpleNamespace] = None,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Build a DataLoader (or two DataLoaders in case of train-val split)
    based on the given structure paths and target values.

    Args:
        structure_paths: List of paths to the structures.
        target: Target values.
        artefacts_dir: path to feature generation artefacts directory.
        train_val_split: Whether to split the data into training and validation sets.
        args: Arguments for the QuanDataset and DataLoader.

    Returns:
        DataLoader or a tuple of DataLoaders for training and validation.
    """

    # Use the provided args or the default configuration if none is provided
    if args is None:
        args = DEFAULT_CONFIG

    # If train-val split is required
    if train_val_split:
        train_index, val_index = split_indices(target, val_size=args.val_size, max_bins_stratify=args.max_bins_stratify)

        train_dataset = QuanDataset(
            structure_paths=[structure_paths[i] for i in train_index],
            target=[target[i] for i in train_index],
            artefacts_dir=artefacts_dir,
            args=args,
        )
        val_dataset = QuanDataset(
            structure_paths=[structure_paths[i] for i in val_index],
            target=[target[i] for i in val_index],
            artefacts_dir=artefacts_dir,
            args=args,
        )

        train_sampler = QuanSampler(train_dataset.structure_ids, batch_size=args.batch_size)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=None, shuffle=False)

        return train_loader, val_loader

    # If only one dataset is required (no train-val split)
    else:
        dataset = QuanDataset(structure_paths=structure_paths, target=target, artefacts_dir=artefacts_dir, args=args)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=None, shuffle=False)

        return loader
