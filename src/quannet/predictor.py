from typing import List, Union
from pathlib import Path

import pandas as pd
import torch

from quannet.tasks import QuanModel
from quannet.utils import LOGGER, SETTINGS, DEFAULT_CONFIG, increment_path
from quannet.config import get_config
from quannet.dataset import build_input


class QuanPredictor:
    """
    Class for making predictions using the QuanModel.
    """

    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        self.args = get_config(config, overrides)
        self.model = None
        self.dataset = None
        self.loader = None

        project = self.args.project or Path(SETTINGS['runs_dir'])
        name = self.args.name or f'{self.args.mode}'
        if self.args.precomputed_input:
            self.save_dir = Path(project) / name
        else:
            self.save_dir = Path(increment_path(Path(project) / name, exist_ok=self.args.exist_ok))
        self.artefacts_dir = self.save_dir / SETTINGS['artefacts_dir_name']

    def _prepare_input(self, structures: List[Union[str, Path]]):
        """
        Prepare the input for the model using a list of structures.
        Generates DataLoader based on the structure paths provided.
        """
        self.args.num_augmentations = 1
        LOGGER.info('Started generating model input')
        loader = build_input(
            structures,
            artefacts_dir=self.artefacts_dir,
            precomputed=self.args.precomputed_input,
            train_val_split=False,
            args=self.args,
        )
        LOGGER.info('Finished generating model input')
        self.loader = loader

    @torch.no_grad()
    def predict(self, structures: Union[str, Path]):
        """
        Perform prediction based on the given structures directory path.

        Args:
            structures: Directory containing the .pdb files for prediction.

        Returns:
            Concatenated tensor of all predictions.
        """

        LOGGER.info('Starting inference ...')
        structures = [p for p in Path(structures).iterdir() if p.name.endswith('.pdb')]
        if not structures:
            raise FileNotFoundError('No .pdb files found in the specified directory.')
        self._prepare_input(structures)
        all_predictions = []
        for x, _ in self.loader:
            y_pred = self.model(x)
            all_predictions.append(y_pred.detach().cpu())
        concatenated_predictions = torch.cat(all_predictions, dim=0).numpy().squeeze()
        pd_output = pd.Series(index=[Path(p).with_suffix('').name for p in structures], data=concatenated_predictions)
        pd_output.to_csv(self.save_dir / 'prediction.csv')
        LOGGER.info('Inference finished')
        return pd_output

    def get_model(self, model: QuanModel, weights=None, verbose=True):
        """
        Load the weights into the model and set it to evaluation mode.

        Args:
            model: The model to which weights will be loaded.
            weights: Path to the model weights.
            verbose: Whether to print messages while loading the model.

        Returns:
            The model set in evaluation mode.
        """
        if weights:
            model.load(weights, verbose=verbose)
        model.eval()
        return model
