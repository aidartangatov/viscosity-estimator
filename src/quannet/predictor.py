from typing import Union
from pathlib import Path
from quannet.tasks import BaseModel
from quannet.utils import LOGGER, DEFAULT_CONFIG
from quannet.make_inputs import get_structure_paths
from quannet.preprocessor import Preprocessor

import numpy as np
import torch
import pandas as pd


class Predictor(Preprocessor):
    """
    Class for making predictions using the QuanModel.
    """

    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        super().__init__(config=config, overrides=overrides)
        self.model = None
        self.dataset = None
        self.loader = None

    @torch.no_grad()
    def predict(self, structures: Union[str, Path]) -> pd.Series:
        """
        Perform prediction based on the given structures directory path.

        Args:
            structures: Directory containing the .pdb files for prediction.

        Returns:
            Concatenated tensor of all predictions.
        """

        LOGGER.info('Starting inference ...')
        structure_paths = get_structure_paths(structures)
        self.loader = self.get_loader(structure_paths)

        was_training = self.model.training
        self.model.eval()
        try:
            all_predictions = []
            for x in self.loader:
                y_pred = self.model(x)
                all_predictions.append(y_pred.detach().cpu())
        finally:
            self.model.train(was_training)

        predictions = torch.cat(all_predictions, dim=0).numpy().reshape(-1)
        data = self._collapse_augmentations(predictions, n_structures=len(structure_paths))

        pd_output = pd.Series(index=[Path(p).with_suffix('').name for p in structure_paths], data=data)
        pd_output.to_csv(self.save_dir / 'prediction.csv')
        LOGGER.info('Inference finished')
        return pd_output

    @staticmethod
    def _collapse_augmentations(predictions: np.ndarray, n_structures: int) -> np.ndarray:
        """
        Collapse `n_structures * n_augs` predictions into `n_structures` by averaging
        across augmentations of the same structure. The data loader yields one
        prediction per (structure, augmentation) pair; the QuanSampler keeps the
        augmentations of each structure contiguous in the batch order.

        Args:
            predictions: 1-D array of length n_structures * n_augs.
            n_structures: Number of distinct structures.

        Returns:
            1-D array of length n_structures.

        Raises:
            ValueError: If `len(predictions)` is not a positive multiple of `n_structures`.
        """
        n_predictions = len(predictions)
        if n_predictions == n_structures:
            return predictions
        n_augs, remainder = divmod(n_predictions, n_structures)
        if remainder != 0 or n_augs == 0:
            raise ValueError(
                f'Got {n_predictions} predictions for {n_structures} structures; ' 'cannot collapse augmentations.'
            )
        return predictions.reshape(n_structures, n_augs).mean(axis=1)

    def get_model(self, model: BaseModel, weights=None, verbose=True):
        """
        Load the weights into the model.

        Eval-mode toggling is handled in `predict()` so that callers receive
        the model in whichever mode they passed it in.

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
