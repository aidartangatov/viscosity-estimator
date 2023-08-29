from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

from quannet.trainer import QuanTrainer
from quannet.utils import DEFAULT_CONFIG_DICT, DEFAULT_CONFIG_KEYS, LOGGER, load_yaml, search_file


def load_model_from_checkpoint(
    weights: Union[str, Path], device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, Dict]:
    """
    Load a PyTorch model from a checkpoint file.

    Args:
        weights: Path to the checkpoint file containing the model weights.
        device: Device to which the model should be loaded. If None, the model will remain on the CPU.

    Returns:
        A tuple containing the loaded model and the entire checkpoint dictionary.

    Raises:
        FileNotFoundError: If the provided 'weights' file path does not exist.
    """

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"'weights' {weights_path} does not exist")

    ckpt = torch.load(str(weights_path), map_location='cpu')
    args = {**DEFAULT_CONFIG_DICT, **(ckpt.get('train_args', {}))}

    model = ckpt['model']
    if device:
        model = model.to(device)

    model.args = {k: v for k, v in args.items() if k in DEFAULT_CONFIG_KEYS}
    model.pt_path = str(weights_path)

    return model, ckpt


class Model:
    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
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

    def _load_model(self, weights: Union[str, Path]):
        self.model, self.ckpt = load_model_from_checkpoint(weights)
        self.ckpt_path = self.model.pt_path
        self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)

    def _build_model(self, model=None):
        if model is None:
            model = self.model_map['model']
        overrides = self.overrides.copy()
        self.model = model(overrides=overrides)

    def __call__(self, structures=None, **kwargs):
        return self.predict(structures, **kwargs)

    def predict(self, structures, **kwargs):
        pass

    def train(self, **kwargs):
        overrides = self.overrides.copy()
        if kwargs.get('config'):
            LOGGER.info(f"overrides file passed. Overriding default params with {kwargs['overrides']}.")
            overrides = load_yaml(search_file(kwargs['config']))
        overrides.update(kwargs)
        overrides['mode'] = 'train'
        if not overrides.get('dataset'):
            raise AttributeError("Dataset required but missing, pass 'dataset=/path/to/dataset'")
        if overrides.get('resume'):
            overrides['resume'] = self.ckpt_path
        self.trainer = QuanTrainer(overrides=overrides)
        if not overrides.get('resume'):
            self.trainer.model = self.trainer.get_model(model=self.model, weights=self.model if self.ckpt else None)
            self.model = self.trainer.model
        self.trainer.train()
        # self.model, _ = load_model_from_checkpoint(str(self.trainer.best))
        self.overrides = self.model.args

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'grid_dim', 'grid_spacing', 'shell_width', 'num_augmentations'}
        return {k: v for k, v in args.items() if k in include}

    @property
    def model_map(self):
        raise NotImplementedError('Provide model map!')
