from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch

from quannet.dataset import build_input
from quannet.trainer import Trainer
from quannet.utils import DEFAULT_CONFIG_DICT, DEFAULT_CONFIG_KEYS, LOGGER, search_file

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_torch_model(model_path, device=None):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"'model_path' {model_path} does not exist")

    ckpt = torch.load(str(model_path), map_location='cpu')
    args = {**DEFAULT_CONFIG_DICT, **(ckpt.get('train_args', {}))}

    model = ckpt['model']
    if device:
        model = model.to(device)

    model.args = {k: v for k, v in args.items() if k in DEFAULT_CONFIG_KEYS}
    model.pt_path = str(model_path)

    return model


class Model:
    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        # self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None
        self.model = None
        self.trainer = None
        self.ckpt = None
        self.ckpt_path = None
        self.overrides = {}
        self.metrics = None

        if model_path is None:
            self._build_model()
        else:
            model_path = search_file(model_path, dir='models')
            self._load_model(model_path)
            self.overrides['model'] = str(model_path)

    def _load_model(self, model_path: Union[str, Path]):
        self.model = load_torch_model(model_path)
        self.ckpt = model_path
        self.ckpt_path = model_path
        self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)

    def _build_model(self, model=None):
        self.model = model or self.model_map['model']
        self.ckpt = None
        args = {**DEFAULT_CONFIG_DICT, **self.overrides}
        self.model.args = {k: v for k, v in args.items() if k in DEFAULT_CONFIG_DICT}
        # self.ckpt_path = self.pt_model.pt_path
        # self.overrides = self.model.args

    def __call__(self, structures=None, **kwargs):
        return self.predict(structures, **kwargs)

    def predict(self, structures, **kwargs):
        pass

    def train(self, **kwargs):
        overrides = self.overrides.copy()
        if kwargs.get('config'):
            LOGGER.info(f"overrides file passed. Overriding default params with {kwargs['overrides']}.")
            overrides = load_torch_model(search_file(kwargs['config']))
        overrides.update(kwargs)
        overrides['mode'] = 'train'

        train_df = pd.read_csv(overrides.data, usecols=['path', 'target'])
        target = list(train_df['target'])
        structure_paths = list(train_df['path'])

        train_loader, val_loader = build_input(structure_paths, target, train_val_split=True, args=overrides)
        trainer = Trainer(model=self.model, train_loader=train_loader, val_loader=val_loader)
        trainer.train()

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'grid_dim', 'grid_spacing', 'shell_width', 'num_augmentations'}
        return {k: v for k, v in args.items() if k in include}

    @property
    def model_map(self):
        raise NotImplementedError('Provide model map!')
