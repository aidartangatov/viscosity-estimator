from copy import deepcopy
from typing import Any, Dict, Tuple, Union, Optional
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn, optim
import lightning.pytorch as pl

from quannet import __version__
from quannet.utils import LOGGER, DEFAULT_CONFIG, DEFAULT_CONFIG_DICT, DEFAULT_CONFIG_KEYS, IterableNamespace
from quannet.config import get_config


class QuanModel(nn.Module):
    """
    Base class for all models
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
        checkpoint_state_dict = self.intersect_dicts(checkpoint_state_dict, self.state_dict())
        self.load_state_dict(checkpoint_state_dict, strict=False)
        if verbose:
            LOGGER.info(
                f'Transferred {len(checkpoint_state_dict)}/{len(self.model.state_dict())} items from pretrained weights'
            )


class LitModel(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, args: IterableNamespace):
        super().__init__()
        self.model = model
        self.args = args

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.huber_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.huber_loss(y_hat, y)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint['model'] = deepcopy(self.model)
        checkpoint['train_args'] = vars(self.args)
        checkpoint['date'] = datetime.now().isoformat()
        checkpoint['version'] = __version__


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
