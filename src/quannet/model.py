from types import SimpleNamespace
from typing import Any, Dict, Union, Optional
from pathlib import Path

import pandas as pd
import torch.nn

from quannet.tasks import load_model_from_checkpoint
from quannet.utils import LOGGER, TEST_STRUCTURES, load_yaml, search_file
from quannet.trainer import QuanTrainer
from quannet.predictor import QuanPredictor
from quannet.make_inputs import get_structure_paths
from quannet.preprocessor import QuanPreprocessor


class Model:
    """
    Class for handling training and prediction of QuanNet models.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.predictor = None
        self.model = None
        self.trainer = None
        self.predictor = None
        self.preprocessor = None
        self.ckpt = None
        self.ckpt_path = None
        self.overrides = {}
        self.metrics = None

        if model_path is None:
            self._build_model()
        else:
            model_path = search_file(model_path, dir='models')
            self._load_model(model_path)

    def _load_model(self, weights: Union[str, Path]) -> None:
        """Load a pre-trained model from a checkpoint."""
        self.model, self.ckpt = load_model_from_checkpoint(weights)
        self.ckpt_path = self.model.pt_path
        self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)

    def _build_model(self, model: Optional[torch.nn.Module] = None) -> None:
        """Build a new model."""
        if model is None:
            model = self.model_map['model']
        overrides = self.overrides.copy()
        self.model = model(overrides=overrides)

    def __call__(self, structures: Optional[Union[str, Path]] = None, **kwargs):
        """Make predictions on structures."""
        return self.predict(structures, **kwargs)

    def preprocess(self, structures: Optional[Union[str, Path]] = None, **kwargs) -> None:
        if structures is None:
            structures = TEST_STRUCTURES
            LOGGER.warning(f"'structures' is missing, using 'structures={structures}'.")
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        overrides['mode'] = kwargs.get('mode', 'preprocess')
        self.preprocessor = QuanPreprocessor(overrides=overrides)
        structure_paths = get_structure_paths(structures)
        self.preprocessor.preprocess_inputs(structure_paths, return_arrays=False)

    def predict(self, structures: Optional[Union[str, Path]] = None, **kwargs) -> pd.Series:
        """
        Make predictions on given structures.

        Args:
            structures: Path to directory with .pdb structures to predict on.
            **kwargs: Additional keyword arguments.
        """
        if structures is None:
            structures = TEST_STRUCTURES
            LOGGER.warning(f"'structures' is missing, using 'structures={structures}'.")
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        overrides['mode'] = kwargs.get('mode', 'predict')
        self.predictor = QuanPredictor(overrides=overrides)
        self.predictor.model = self.predictor.get_model(model=self.model, weights=self.model if self.ckpt else None)
        self.model = self.predictor.model
        return self.predictor.predict(structures)

    def train(self, **kwargs) -> None:
        """
        Train a model with provided or overridden parameters.

        Args:
            **kwargs: Additional keyword arguments including training settings.
        """
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
    def _reset_ckpt_args(args: Union[Dict, SimpleNamespace]) -> Dict[str, Any]:
        """Reset arguments when loading a PyTorch model."""
        include = {'grid_dim', 'grid_spacing', 'shell_width', 'num_augmentations'}
        return {k: v for k, v in args.items() if k in include}

    def __getattr__(self, attr: str) -> None:
        """
        Overrides the default __getattr__ method to provide a custom error message.
        Raises an AttributeError if the requested attribute is not found.
        """
        name = self.__class__.__name__
        class_attributes = ', '.join(dir(self.__class__))  # Listing class-level attributes

        error_message = (
            f"'{name}' object has no attribute '{attr}'.\n"
            f"Valid class-level attributes are: {class_attributes}\n"
            f"For more details, refer to:\n{self.__doc__}"
        )

        raise AttributeError(error_message)

    @property
    def model_map(self):
        raise NotImplementedError('Provide model map!')
