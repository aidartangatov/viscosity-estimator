from pathlib import Path

import pandas as pd
import torch

from quannet.tasks import QuanModel
from quannet.utils import LOGGER, SETTINGS, DEFAULT_CONFIG, increment_path
from quannet.config import get_config
from quannet.dataset import build_input


class QuanPredictor:
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

    def _prepare_input(self, structures):
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

    @torch.inference_mode()
    def predict(self, structures):
        structures = [p for p in Path(structures).iterdir() if p.name.endswith('.pdb')]
        self._prepare_input(structures)
        all_predictions = []
        for x, _ in self.loader:
            y_pred = self.model(x)
            all_predictions.append(y_pred.detach().cpu())
        concatenated_predictions = torch.cat(all_predictions, dim=0).numpy()
        pd.Series(index=[Path(p).with_suffix('').name for p in structures], data=concatenated_predictions).to_csv(
            self.save_dir / 'prediction.csv'
        )

    def get_model(self, model: QuanModel, weights=None, verbose=True):
        if weights:
            model.load(weights, verbose=verbose)
        model.eval()
        return model
