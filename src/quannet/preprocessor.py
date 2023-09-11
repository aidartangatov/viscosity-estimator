from typing import Dict, List, Union, Optional
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from quannet.utils import LOGGER, SETTINGS, DEFAULT_CONFIG, InsufficientDataError, increment_path
from quannet.config import get_config
from quannet.dataset import QuanDataset, QuanSampler
from quannet.make_inputs import make_inputs, load_esp_arrays
from quannet.docker.make_inputs import run_make_inputs


class QuanPreprocessor:
    """
    Class for preparing input Electrostatic Potential (ESP) grids for QuanNet models.

    Provides methods to load or generate grids, convert molecular structure files to input tensors,
    and create PyTorch DataLoaders for model training.

    Attributes:
        args: A namespace or dictionary containing configuration settings.
        save_dir: Directory where model checkpoints and logs are saved.
        artefacts_dir: Directory to store the computed or loaded ESP grids.

    Methods:
        get_save_dir: Get or create the directory to save model checkpoints and logs.
        preprocess_inputs: Prepare the ESP grids for a list of molecular structures.
        get_loader: Create and return a DataLoader for ESP grids and optional targets.
    """

    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        self.args = get_config(config, overrides)
        self.save_dir = self.get_save_dir()
        self.artefacts_dir = self.get_artefacts_dir()

    def get_save_dir(self) -> Path:
        """
        Get or create the directory to save model checkpoints, logs, and ESP preparation outputs.

        Returns:
            The directory path.
        """
        if hasattr(self, 'save_dir'):
            return self.save_dir
        project = self.args.project or Path(SETTINGS['runs_dir'])
        name = self.args.name or f'{self.args.mode}'
        return increment_path(Path(project) / name, exist_ok=self.args.exist_ok, mkdir=True)

    def get_artefacts_dir(self) -> Union[Path, Dict[str, Path]]:
        if self.args.mode == 'train':
            train_artefacts_dir = self.save_dir / SETTINGS['artefacts_dir_name'] / 'train_set'
            val_artefacts_dir = self.save_dir / SETTINGS['artefacts_dir_name'] / 'val_set'
            train_artefacts_dir.mkdir(parents=True, exist_ok=self.args.exist_ok)
            val_artefacts_dir.mkdir(parents=True, exist_ok=self.args.exist_ok)
            return {'train': train_artefacts_dir, 'val': val_artefacts_dir}
        else:
            artefacts_dir = self.save_dir / SETTINGS['artefacts_dir_name'] / f'{self.args.mode}_set'
            artefacts_dir.mkdir(parents=True, exist_ok=self.args.exist_ok)
            return artefacts_dir

    def preprocess_inputs(
        self,
        structure_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        return_arrays: bool = True,
    ) -> Union[torch.Tensor, None]:
        """
        Prepare the ESP grids for a list of molecular structures.

        Args:
            structure_paths: List of paths to molecular structure files.
            output_dir: Directory to save preprocessing output files.
            return_arrays: If True, return the processed ESP grids.

        Returns:
            Tensor containing ESP grids for the provided molecular structures.
        """
        if self.args.precomputed_input:
            inputs = []
            for structure_path in tqdm(structure_paths):
                esp_arrays = load_esp_arrays(structure_path)
                size = len(esp_arrays)
                if size < self.args.num_augmentations:
                    raise InsufficientDataError(
                        f'Only {size} ESP arrays were found, while {self.args.num_augmentations} expected.'
                    )
                inputs.extend(esp_arrays)

        else:
            shared_params = {
                'structure_paths': structure_paths,
                'grid_dim': self.args.grid_dim,
                'grid_spacing': self.args.grid_spacing,
                'shell_width': self.args.shell_width,
                'num_augmentations': self.args.num_augmentations,
                'processes': self.args.processes,
                'artefacts_dir': output_dir,
                'remove_artefacts': self.args.remove_artefacts,
                'train_mode': self.args.mode == 'train',
            }
            if self.args.docker_image:
                inputs = run_make_inputs(**shared_params, image=self.args.docker_image)
            else:
                inputs = make_inputs(**shared_params, return_arrays=return_arrays)

        # Convert format into [sample_size, channels, depth, height, width]
        inputs = np.expand_dims(np.array(inputs), 1)

        return torch.FloatTensor(inputs)

    def get_loader(
        self,
        structure_paths: List[Union[str, Path]],
        targets: Optional[List[Union[str, Path]]] = None,
        train_set: bool = False,
    ) -> DataLoader:
        """
        Create a DataLoader containing ESP grids and, if provided, the targets.

        Args:
            structure_paths: List of paths to molecular structure files.
            targets: Optional list of target values for the molecular structures.
            train_set: If True, the DataLoader is for training, shuffling the dataset based on unique structures.
                       Requires `targets`.

        Returns:
            DataLoader object containing the ESP grids and optionally the targets.

        Raises:
            AssertionError: If 'train_set' is True but 'targets' is None.
        """
        LOGGER.info('Preparing  input files ...')
        if self.args.mode == 'train':
            output_dir = self.artefacts_dir['train' if train_set else 'val']
        else:
            output_dir = self.artefacts_dir
        inputs = self.preprocess_inputs(structure_paths, output_dir)
        if targets is not None:
            targets = torch.FloatTensor(np.repeat(targets, self.args.num_augmentations)).view(-1, 1)
        dataset = QuanDataset(inputs=inputs, targets=targets)
        sampler = None
        if train_set:
            assert targets is not None, "'targets' can not be empty while 'train_set' is True"
            structure_ids = np.arange(len(structure_paths))
            sampler = QuanSampler(structure_ids, batch_size=self.args.batch_size)
        loader = DataLoader(
            dataset, batch_size=self.args.batch_size, sampler=sampler, shuffle=False, num_workers=self.args.processes
        )
        return loader
