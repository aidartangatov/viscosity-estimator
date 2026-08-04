from types import SimpleNamespace
from typing import Any, Dict, Union, Optional
from pathlib import Path
from quannet.tasks import load_model_from_checkpoint
from quannet.utils import LOGGER, load_yaml, search_file, TEST_STRUCTURES
from quannet.trainer import Trainer
from quannet.predictor import Predictor
from quannet.make_inputs import get_structure_paths
from quannet.preprocessor import Preprocessor

import numpy as np
import pandas as pd
import random
import torch.nn


class Model:
    """
    A high-level interface for handling training and prediction tasks with QuanNet models.

    Provides utilities for loading, building, training, and predicting QuanNet models from either pretrained
    checkpoints or new configurations. Additionally, it provides a method for preprocessing model inputs.

    Attributes
        predictor: QuanPredictor instance for making predictions.
        model: The neural network model instance.
        trainer: QuanTrainer instance for training the model.
        preprocessor: QuanPreprocessor instance for data preprocessing.
        ckpt: Loaded model checkpoint if any.
        ckpt_path: Path to the loaded model checkpoint.
        overrides: Dictionary of overridden configuration settings.
        metrics: Metrics related to the model's performance (if any).

    Methods:
        __call__: Alias for the predict method.
        preprocess: Preprocess input structures for the model.
        predict: Make predictions on the provided structures.
        train: Train the model using provided or default settings.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
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

    def _load_model(self, checkpoint: Union[str, Path]) -> None:
        """Load a pre-trained model from a checkpoint."""
        self.model, self.ckpt = load_model_from_checkpoint(checkpoint)
        self.ckpt_path = self.model.pt_path
        self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)

    def _build_model(self, model: Optional[torch.nn.Module] = None) -> None:
        """Build a new model."""
        if model is None:
            model = self.model_map['model']
        overrides = self.overrides.copy()
        self.model = model(overrides=overrides)

    def __call__(self, structures: Optional[Union[str, Path]] = None, **kwargs) -> pd.Series:
        """Make predictions on structures."""
        return self.predict(structures, **kwargs)

    def preprocess(self, structures: Optional[Union[str, Path]] = None, **kwargs) -> None:
        """
        Preprocess input structures for the model.

        Args:
            structures: Path to directory with .pdb structures to preprocess.
            **kwargs: Additional keyword arguments.
        """
        if structures is None:
            structures = TEST_STRUCTURES
            LOGGER.warning(f"'structures' is missing, using 'structures={structures}'.")
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        overrides['mode'] = kwargs.get('mode', 'preprocess')
        # Rotation augmentation pulls from `random.uniform` in the parent
        # process. Seed it so preprocess outputs are reproducible across runs.
        # Trainer.train seeds via `L.seed_everything` on its own path; predict
        # mode does not generate rotations.
        random.seed(42)
        np.random.seed(42)
        self.preprocessor = Preprocessor(overrides=overrides)
        structure_paths = get_structure_paths(structures)
        self.preprocessor.preprocess_inputs(structure_paths, return_arrays=False)

    def predict(self, structures: Optional[Union[str, Path]] = None, **kwargs) -> pd.Series:
        """
        Make predictions on provided structures.

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
        self.predictor = Predictor(overrides=overrides)
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
        self.trainer = Trainer(overrides=overrides)
        if not overrides.get('resume'):
            self.trainer.model = self.trainer.get_model(model=self.model, weights=self.model if self.ckpt else None)
            self.model = self.trainer.model
        self.trainer.train()
        self.model, _ = load_model_from_checkpoint(str(self.trainer.best))
        self.overrides = self.model.args

    # Args that should NOT carry over from a loaded checkpoint into a new run.
    # These are runtime/environment knobs (where to save, what device to use,
    # whether to resume, which dataset/structures to read). Everything *not*
    # in this set is treated as a property of the trained model — that way
    # adding a new physics/preprocessing param to `default.yaml` automatically
    # gets persisted with the checkpoint instead of silently defaulting on
    # reload. If a new *runtime* knob is added, add it here.
    _CKPT_DROP_ARGS = frozenset(
        {
            'mode',
            'project',
            'name',
            'exist_ok',
            'resume',
            'config',
            'accelerator',
            'devices',
            'every_n_train_steps',
            'deterministic',
            'dataset',
            'structures',
            'model',
            'path_csv_col',
            'target_csv_col',
            'docker_image',
            'precomputed_input',
            'remove_artefacts',
            'processes',
            'lr',
            'weight_decay',
            'batch_size',
            'max_epochs',
            'val_size',
            'max_bins_stratify',
        }
    )

    @staticmethod
    def _reset_ckpt_args(args: Union[Dict, SimpleNamespace]) -> Dict[str, Any]:
        """Strip runtime/environment args from a loaded checkpoint's args.

        See `_CKPT_DROP_ARGS` for the denylist and rationale.
        """
        return {k: v for k, v in args.items() if k not in Model._CKPT_DROP_ARGS}

    def __getattr__(self, attr: str) -> None:
        """Raises an AttributeError if the requested attribute is not found."""
        basic_error = f"'{self.__class__.__name__}' object has no attribute '{attr}'."
        doc_hint = f"\nFor valid attributes and methods, refer to the class documentation.\n{self.__doc__}"
        raise AttributeError(basic_error + doc_hint)

    @property
    def model_map(self):
        """Property placeholder. Requires implementation to map model names to classes."""
        raise NotImplementedError('Provide model map!')
