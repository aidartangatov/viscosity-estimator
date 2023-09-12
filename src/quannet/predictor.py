from typing import Union
from pathlib import Path
from quannet.tasks import QuanModel
from quannet.utils import LOGGER, DEFAULT_CONFIG
from quannet.config import get_config
from quannet.make_inputs import get_structure_paths
from quannet.preprocessor import Preprocessor

import torch
import pandas as pd


class Predictor(Preprocessor):
    """
    Class for making predictions using the QuanModel.
    """

    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        super().__init__(config=config, overrides=overrides)
        self.args = get_config(config, overrides)
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
        all_predictions = []
        for x in self.loader:
            y_pred = self.model(x)
            all_predictions.append(y_pred.detach().cpu())
        concatenated_predictions = torch.cat(all_predictions, dim=0).numpy().squeeze()
        pd_output = pd.Series(
            index=[Path(p).with_suffix('').name for p in structure_paths], data=concatenated_predictions
        )
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
