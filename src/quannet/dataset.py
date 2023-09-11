from typing import Tuple, Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from quannet.utils import ArrayLike


class QuanDataset(Dataset):
    """
    Custom dataset for QuanNet to hold model inputs and their corresponding targets.

    Attributes:
        inputs: A tensor containing model inputs.
        targets: A tensor containing the target values corresponding to the structures. None if not provided.
    """

    def __init__(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve an item from the dataset at the specified index.
        """
        sample = self.inputs[index]
        if self.targets is not None:
            target = self.targets[index]
            return sample, target
        return sample


class QuanSampler(Sampler):
    """
    Custom sampler for QuanNet to shuffle the dataset based on unique structure IDs.

    This class extends PyTorch's Sampler class and overrides the __iter__ and __len__
    methods to work with PyTorch's DataLoader for batch processing.

    Args:
        structure_ids: An array containing unique identifiers for each structure.
        batch_size: The size of each batch to be sampled from the DataLoader.

    """

    def __init__(self, structure_ids: ArrayLike, batch_size: int):
        self.structure_ids = structure_ids
        self.batch_size = batch_size
        self.shuffled_indices = self.structure_shuffle(self.structure_ids)

    def __iter__(self):
        return iter(self.shuffled_indices)

    def __len__(self):
        return len(self.shuffled_indices)

    @staticmethod
    def structure_shuffle(structure_ids: ArrayLike):
        """
        Shuffle the dataset indices based on unique structure IDs.

        Args:
            structure_ids: An array containing unique identifiers for each structure.

        Returns:
            list: A list of shuffled indices.
        """
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
