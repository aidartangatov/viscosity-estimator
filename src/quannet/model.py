from typing import Union, Optional
from pathlib import Path

from quannet.tasks import load_model_from_checkpoint
from quannet.utils import LOGGER, load_yaml, search_file
from quannet.trainer import QuanTrainer


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
        self.model, _ = load_model_from_checkpoint(str(self.trainer.best))
        self.overrides = self.model.args

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'grid_dim', 'grid_spacing', 'shell_width', 'num_augmentations'}
        return {k: v for k, v in args.items() if k in include}

    @property
    def model_map(self):
        raise NotImplementedError('Provide model map!')
