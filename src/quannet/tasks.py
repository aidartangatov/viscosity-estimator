from copy import deepcopy
from torch import nn, optim
from typing import Any, Dict, Tuple, Union
from pathlib import Path
from quannet import __version__
from datetime import datetime
from quannet.utils import LOGGER, DEFAULT_CONFIG, IterableNamespace, DEFAULT_CONFIG_DICT, DEFAULT_CONFIG_KEYS
from quannet.config import get_config

import torch
import lightning as L
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    Base class for QuanNet models

    Provides basic functionality for initializing and loading pre-trained QuanNet models.
    """

    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        super().__init__()
        self.args = get_config(config, overrides)

    @staticmethod
    def _intersect_dicts(da, db):
        """
        Returns a dictionary with intersecting keys having matching shapes.

        Args:
            da: First dictionary.
            db: Second dictionary.

        Returns:
            Dictionary containing keys present in both input dictionaries with matching shapes.
        """
        return {k: v for k, v in da.items() if k in db and v.shape == db[k].shape}

    def load(self, weights: Union[dict, torch.nn.Module], verbose: bool = False):
        """
        Load the weights into the model.

        Args:
            weights: The pre-trained weights to be loaded.
            verbose: Whether to log information about weight transfer. Defaults to False.
        """
        model = weights['model'] if isinstance(weights, dict) else weights
        checkpoint_state_dict = model.float().state_dict()
        # Check for intersection between parameters of the checkpoint and the current model,
        # and ensures the shapes of the weights match.
        checkpoint_state_dict = self._intersect_dicts(checkpoint_state_dict, self.state_dict())
        self.load_state_dict(checkpoint_state_dict, strict=False)
        if verbose:
            LOGGER.info(
                f'Transferred {len(checkpoint_state_dict)}/{len(model.state_dict())} items from pretrained weights'
            )


class LitModel(L.LightningModule):
    """
    Lightning module wrapper for QuanNet models.

    This class defines the training, validation, and prediction procedures for QuanNet models.
    """

    def __init__(self, model: torch.nn.Module, args: IterableNamespace):
        super().__init__()
        self.model = model
        self.args = args

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.huber_loss(y_hat, y)
        metrics = {'train_loss': loss}
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.huber_loss(y_hat, y)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        # Save only weights + the model's qualified class so the loader can
        # reinstantiate it. The previous implementation `deepcopy`'d the whole
        # nn.Module into the checkpoint, which (a) duplicated the state_dict
        # Lightning already writes, and (b) made checkpoints depend on the
        # exact pickling-compatible Python/PyTorch versions used at save time.
        checkpoint['model_state_dict'] = self.model.state_dict()
        checkpoint['model_class'] = f'{self.model.__class__.__module__}.{self.model.__class__.__qualname__}'
        checkpoint['train_args'] = vars(self.args)
        checkpoint['date'] = datetime.now().isoformat()
        checkpoint['version'] = __version__


def load_model_from_checkpoint(checkpoint: Union[str, Path]) -> Tuple[torch.nn.Module, Dict]:
    """
    Load a PyTorch model from a checkpoint file.

    Args:
        checkpoint: Path to the checkpoint file containing the model weights.

    Returns:
        A tuple containing the loaded model and the entire checkpoint dictionary.

    Raises:
        FileNotFoundError: If the provided 'weights' file path does not exist.
    """

    import importlib

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"'checkpoint' {checkpoint} does not exist")

    ckpt = torch.load(str(checkpoint), map_location='cpu')
    args = {**DEFAULT_CONFIG_DICT, **(ckpt.get('train_args', {}))}
    model_args = {k: v for k, v in args.items() if k in DEFAULT_CONFIG_KEYS}

    if 'model_state_dict' in ckpt and 'model_class' in ckpt:
        # New format: reinstantiate from class + state_dict
        module_path, class_name = ckpt['model_class'].rsplit('.', 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        model = cls(overrides=model_args)
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model' in ckpt and isinstance(ckpt['model'], torch.nn.Module):
        # Legacy format: a deepcopied nn.Module was saved under 'model'.
        model = ckpt['model']
    else:
        raise ValueError(
            f"Checkpoint {checkpoint} has neither 'model_state_dict'+'model_class' "
            "nor a legacy 'model' module."
        )

    model.args = model_args
    model.pt_path = str(checkpoint)

    return model, ckpt
