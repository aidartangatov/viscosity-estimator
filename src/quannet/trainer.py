from typing import Tuple, Union, Optional
from pathlib import Path
from collections import Counter
from quannet.tasks import LitModel, BaseModel
from quannet.utils import (
    LOGGER,
    SETTINGS,
    ArrayLike,
    search_file,
    DATASETS_DIR,
    DEFAULT_CONFIG,
    IterableNamespace,
    DuplicatedDataError,
    InsufficientDataError,
)
from quannet.config import get_config
from quannet.preprocessor import Preprocessor
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import pandas as pd
import lightning as L


class Trainer(Preprocessor):
    """
    Trainer class for training QuanNet models

    Provides utility methods for model training including checkpointing, early stopping, and data loading.
    It makes use of Lightning for efficient model training and monitoring.
    """

    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        super().__init__(config=config, overrides=overrides)
        self.args = get_config(config, overrides)
        self.dataset = self.get_dataset()
        self.trainer = self.get_trainer()
        self.model = None
        self.best = None
        self.train_loader = None
        self.val_loader = None

    def get_dataset(self) -> IterableNamespace:
        """
        Retrieves and validates the dataset specified in the configuration.

        This method internally calls the check_dataset function to validate the dataset's integrity.
        It ensures that the number of valid structures with corresponding target values is sufficient
        and that the dataset's file paths are consistent with the records in its associated CSV file.

        Returns:
            IterableNamespace: An object containing validated paths and target values for the dataset.
                               The namespace contains two keys, 'structure_paths' and 'targets', which
                               hold lists of valid pdb paths and corresponding target values, respectively.
        """
        return check_dataset(
            self.args.dataset,
            path_csv_col=self.args.path_csv_col,
            target_csv_col=self.args.target_csv_col,
            min_structures=1,
        )

    def get_trainer(self) -> L.Trainer:
        """
        Configures and returns a PyTorch Lightning Trainer object.

        Returns:
            L.Trainer: A PyTorch Lightning Trainer configured according to the instance's settings.
        """

        checkpoint_callback = self._checkpoint_callback(self.save_dir)
        return L.Trainer(
            max_epochs=self.args.max_epochs,
            log_every_n_steps=1,
            accelerator=self.args.accelerator,
            devices=self.args.devices,
            callbacks=[self._early_stop_callback(), checkpoint_callback],
            deterministic=self.args.deterministic,
            logger=CSVLogger(save_dir=self.save_dir, name=SETTINGS['logs_dir_name']),
        )

    @staticmethod
    def _early_stop_callback():
        """Returns an early stopping callback configured to monitor validation loss."""
        return EarlyStopping(monitor='val_loss', patience=3, verbose=True, mode='min')

    @staticmethod
    def _checkpoint_callback(dirpath):
        """Returns a model checkpoint callback configured to monitor validation loss."""
        return ModelCheckpoint(
            dirpath=dirpath,
            filename='{epoch:02d}-{val_loss:.2f}',
            save_weights_only=False,
            monitor='val_loss',
            every_n_train_steps=1,
            save_top_k=1,
            mode='min',
            save_last=True,
        )

    def train(self):
        """
        Train the model using the PyTorch Lightning framework.
        """

        L.seed_everything(42, workers=True)
        lit_model = LitModel(model=self.model, args=self.args)
        train_index, val_index = split_indices(
            self.dataset.targets, val_size=self.args.val_size, max_bins_stratify=self.args.max_bins_stratify
        )
        self.train_loader = self.get_loader(
            [self.dataset.structure_paths[i] for i in train_index],
            [self.dataset.targets[i] for i in train_index],
            train_set=True,
        )
        self.val_loader = self.get_loader(
            [self.dataset.structure_paths[i] for i in val_index],
            [self.dataset.targets[i] for i in val_index],
            train_set=False,
        )
        self.trainer.fit(lit_model, self.train_loader, self.val_loader)
        best_model_path = str(self.save_dir / 'best_model.ckpt')
        self.trainer.save_checkpoint(best_model_path)
        self.best = best_model_path

    def get_model(self, model: BaseModel, weights=None, verbose=True):
        """
        Load the weights into the model and return it.

        Args:
            model: The model to which weights will be loaded.
            weights: Path to the model weights.
            verbose: Whether to print messages while loading the model.

        Returns:
            The model with loaded weights.
        """
        if weights:
            model.load(weights, verbose=verbose)
        return model


def check_dataset(
    dataset: Union[str, Path], path_csv_col: str = 'path', target_csv_col: str = 'target', min_structures: int = 1
) -> IterableNamespace:
    """
    Validates the integrity of a given dataset, ensuring the consistency between CSV records and file paths.

    Args:
        dataset: The directory path or dataset name containing structural data.
        path_csv_col: The CSV column name storing the structural file paths. Defaults to 'path'.
        target_csv_col: The CSV column name storing the target values. Defaults to 'target'.
        min_structures: The minimum required number of valid structures with associated target values. Defaults to 1.

    Returns:
        IterableNamespace: A namespace containing two keys, 'structure_paths' and 'targets', which hold lists
                           of valid pdb paths and their corresponding target values, respectively.

    Raises:
        InsufficientDataError: If the number of valid structures with targets is less than `min_structures`.
        DuplicatedDataError: If duplicated file paths are found in the dataset.
    """
    dataset = Path(dataset)
    data_dir = (dataset if dataset.is_dir() else (DATASETS_DIR / dataset)).resolve()
    target_csv_path = search_file('**/*.csv', data_dir)
    target_df = pd.read_csv(target_csv_path, usecols=[path_csv_col, target_csv_col])

    structure_relpaths_from_csv = target_df[path_csv_col].tolist()
    structure_paths = [Path(data_dir / path) for path in structure_relpaths_from_csv]
    structure_relpaths_pathlib_compatible = [str(path.relative_to(data_dir)) for path in structure_paths]

    # Check for duplicates
    duplicates = [item for item, count in Counter(structure_relpaths_pathlib_compatible).items() if count > 1]
    if duplicates:
        message = '\n'.join(duplicates)
        messsage = f'Found duplicate paths in CSV:\n{message}'
        raise DuplicatedDataError(messsage)
    path2str = dict(zip(structure_relpaths_pathlib_compatible, structure_relpaths_from_csv))

    # Check if paths in CSV actually exist
    nonexistent_paths = [path for path in structure_relpaths_pathlib_compatible if not (data_dir / path).exists()]
    if nonexistent_paths:
        message = '\n'.join(nonexistent_paths)
        LOGGER.warning(f'Paths from CSV that do not exist:\n{message}')

    all_pdb_files = [str(path.relative_to(data_dir)) for path in data_dir.rglob('*.pdb')]

    # Check which .pdb files in the directory aren't mentioned in the CSV
    not_in_csv = set(all_pdb_files) - set(structure_relpaths_pathlib_compatible)
    if not_in_csv:
        message = '\n'.join(not_in_csv)
        LOGGER.warning(f'.pdb files not found in CSV:\n{message}')

    # Check which .pdb files are mentioned in the CSV but aren't in the directory
    not_in_dir = set(structure_relpaths_pathlib_compatible) - set(all_pdb_files)
    if not_in_dir:
        message = '\n'.join(not_in_dir)
        LOGGER.warning(f'.pdb paths in CSV but not in directory:\n{message}')

    # Count valid .pdb files with targets
    valid_pdb_with_targets = len(set(all_pdb_files).intersection(set(structure_relpaths_pathlib_compatible)))

    LOGGER.info(f'Found {valid_pdb_with_targets} .pdb files with corresponding target values.')

    if valid_pdb_with_targets < min_structures:
        raise InsufficientDataError(f'Found fewer than {min_structures} .pdb files with corresponding target values.')

    valid_rel_paths = list(set(all_pdb_files).intersection(set(structure_relpaths_pathlib_compatible)))
    valid_targets = target_df[target_df[path_csv_col].isin([path2str[p] for p in valid_rel_paths])][
        target_csv_col
    ].tolist()
    valid_paths = [Path(data_dir / path).resolve() for path in valid_rel_paths]

    return IterableNamespace(**{'structure_paths': valid_paths, 'targets': valid_targets})


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
