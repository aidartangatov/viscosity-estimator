from types import SimpleNamespace
from typing import List, Tuple, Union, Optional
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from quannet.utils import LOGGER, DEFAULT_CONFIG, InsufficientDataError
from quannet.prepare import load_esp_grids, generate_esp_grids

ArrayLike = Union[list, pd.Series, np.ndarray]


class QuanDataset(Dataset):
    def __init__(
        self,
        structure_paths: List[Union[str, Path]],
        artefacts_dir: Union[str, Path],
        precomputed: bool = True,
        target: Optional[List[float]] = None,
        args: SimpleNamespace = DEFAULT_CONFIG,
    ):
        self.structure_paths = structure_paths
        self.target = target
        self.precomputed = precomputed
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
        for structure_path in tqdm(structure_paths):
            structure_path = Path(structure_path)

            if self.precomputed:
                structure_name = structure_path.with_suffix('').name
                esp_grids = load_esp_grids(artefacts_dir=Path(self.artefacts_dir, structure_name))
                size = len(esp_grids)
                if size < self.args.num_augmentations:
                    raise InsufficientDataError(
                        f'Only {size} ESP arrays were found, while {self.args.num_augmentations} expected.'
                    )
            else:
                esp_grids = generate_esp_grids(
                    str(structure_path),
                    artefacts_dir=self.artefacts_dir,
                    grid_dim=self.args.grid_dim,
                    grid_spacing=self.args.grid_spacing,
                    shell_width=self.args.shell_width,
                    num_augmentations=self.args.num_augmentations,
                    processes=self.args.processes,
                    remove_artefacts=self.args.remove_artefacts,
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


def find_stratification_bins(y: ArrayLike, max_bins_stratify: int = 5, min_samples_per_bin: int = 2):
    """
    Attempt to find a suitable number of bins to stratify the data into.

    Args:
        y: The array of labels to stratify by.
        max_bins_stratify: The maximum number of bins to try for stratification.
        min_samples_per_bin: Minimum number of samples required per bin.

    Returns:
        pandas.Series with the same length as `y` indicating the bin each sample belongs to, or
        None if no suitable stratification is possible.
    """
    for n_bins in range(max_bins_stratify, 1, -1):
        stratify = pd.Series(pd.qcut(y, q=n_bins, labels=False))
        if all(stratify.value_counts() >= min_samples_per_bin):
            return stratify
    return None


def split_indices(
    y: ArrayLike, val_size: float = 0.1, max_bins_stratify: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the dataset into training and validation sets.

    Args:
        y: The array of labels.
        val_size: The proportion of the dataset to include in the test split.
        max_bins_stratify: Maximum number of bins to use for stratification.

    Returns:
        Indices for training set.
        Indices for validation set.
    """

    if len(y) == 0:
        raise ValueError("Input array 'y' cannot be empty.")

    if val_size <= 0.0 or val_size >= 1.0:
        raise ValueError('val_size should be in the range [0.0, 1.0].')

    n_samples = len(y)
    indices = np.arange(n_samples)

    stratify = None
    # Attempt to find stratification bins
    if max_bins_stratify is not None:
        stratify = find_stratification_bins(y, max_bins_stratify=max_bins_stratify)

    train_index, val_index = [], []

    if stratify is not None:
        # Check that each stratification bin has enough samples
        bin_counts = np.bincount(stratify)
        min_bin_count = np.min(bin_counts)
        if min_bin_count < 2:
            raise ValueError('Each stratification bin must have at least 2 samples.')

        # Stratified sampling
        for bin in np.unique(stratify):
            bin_indices = indices[stratify == bin]
            n_train = round((1 - val_size) * len(bin_indices))
            train_indices_bin = np.random.choice(bin_indices, n_train, replace=False)
            val_indices_bin = list(set(bin_indices) - set(train_indices_bin))

            train_index.extend(train_indices_bin)
            val_index.extend(val_indices_bin)
    else:
        # Random sampling without stratification
        n_train = round((1 - val_size) * n_samples)
        train_index = np.random.choice(indices, n_train, replace=False)
        val_index = list(set(indices) - set(train_index))

    return np.array(train_index), np.array(val_index)


def build_input(
    structure_paths: List[Union[str, Path]],
    target: Optional[List[float]] = None,
    artefacts_dir: Union[str, Path] = '.',
    precomputed: bool = False,
    train_val_split: bool = False,
    args: Optional[SimpleNamespace] = None,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Build a DataLoader (or two DataLoaders in case of train-val split)
    based on the given structure paths and target values.

    Args:
        structure_paths: List of paths to the structures.
        target: Target values. Required when train_val_split = True.
        artefacts_dir: path to feature generation artefacts directory.
        precomputed: If True, load precomputed input array from files,
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
        if not target:
            raise ValueError("Non-empty 'target' required in train_val_split mode")

        train_index, val_index = split_indices(target, val_size=args.val_size, max_bins_stratify=args.max_bins_stratify)

        LOGGER.info(f'Preparing {len(train_index)} training input files ...')
        train_dataset = QuanDataset(
            structure_paths=[structure_paths[i] for i in train_index],
            target=[target[i] for i in train_index],
            artefacts_dir=artefacts_dir,
            precomputed=precomputed,
            args=args,
        )

        LOGGER.info(f'Preparing {len(val_index)} validation input files ...')
        val_dataset = QuanDataset(
            structure_paths=[structure_paths[i] for i in val_index],
            target=[target[i] for i in val_index],
            artefacts_dir=artefacts_dir,
            precomputed=precomputed,
            args=args,
        )

        train_sampler = QuanSampler(train_dataset.structure_ids, batch_size=args.batch_size)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False, num_workers=args.processes
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, sampler=None, shuffle=False, num_workers=args.processes
        )

        return train_loader, val_loader

    # If only one dataset is required (no train-val split)
    else:
        dataset = QuanDataset(
            structure_paths=structure_paths,
            target=target,
            artefacts_dir=artefacts_dir,
            precomputed=precomputed,
            args=args,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=None, shuffle=False, num_workers=args.processes)

        return loader
