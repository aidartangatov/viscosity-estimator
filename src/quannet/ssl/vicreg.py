"""VICReg self-supervised pretraining (v1) for the ResNet-3D ESP encoder.

VICReg (Bardes, Ponce & LeCun, 2022) avoids the collapse problem contrastive
methods usually solve with negatives or a momentum target network (BYOL): it
regularizes the two augmented views' embeddings directly with three terms -
invariance (make them equal), variance (keep each dimension's std across the
batch above a floor, preventing collapse to a constant), and covariance
(decorrelate feature dimensions, spreading information across all of them).
That means no target/momentum encoder and no large negative-sample batches
are needed, which matters here: the labeled fine-tuning set is tiny (56
antibodies), so a simple, low-hyperparameter pretext task is easier to get
right on a first pass than BYOL's dual-network setup.
"""
from typing import Dict, Tuple, Sequence

import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F


def _off_diagonal(mat: torch.Tensor) -> torch.Tensor:
    n, m = mat.shape
    assert n == m
    return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """VICReg loss between two projected embedding batches (N, D)."""
    repr_loss = F.mse_loss(z1, z2)

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1)) / 2 + torch.mean(F.relu(1 - std_z2)) / 2

    n, d = z1.shape
    cov_z1 = (z1.T @ z1) / (n - 1)
    cov_z2 = (z2.T @ z2) / (n - 1)
    cov_loss = _off_diagonal(cov_z1).pow(2).sum() / d + _off_diagonal(cov_z2).pow(2).sum() / d

    loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    return loss, {'repr_loss': repr_loss.detach(), 'std_loss': std_loss.detach(), 'cov_loss': cov_loss.detach()}


def _infer_encoder_dim(encoder: nn.Module, grid_dim: int = 96) -> int:
    """Probe the encoder's forward_features output width with a dummy input."""
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 1, grid_dim, grid_dim, grid_dim)
        dim = encoder.forward_features(dummy).shape[1]
    encoder.train(was_training)
    return dim


def make_projector(in_dim: int, hidden_dims: Sequence[int] = (128, 128)) -> nn.Sequential:
    dims = [in_dim, *hidden_dims]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class VICRegLitModule(L.LightningModule):
    """Lightning wrapper: encoder + projector head, trained with the VICReg loss.

    Only ``encoder`` (the ResNet-3D trunk) is meant to be reused downstream
    (via ``encoder.forward_features``); the projector is a pretext-task-only
    head, discarded after pretraining - same role as VICReg's original paper.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projector_dims: Sequence[int] = (128, 128),
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        grid_dim: int = 96,
    ):
        super().__init__()
        self.encoder = encoder
        enc_dim = _infer_encoder_dim(encoder, grid_dim)
        self.projector = make_projector(enc_dim, projector_dims)
        self.save_hyperparameters(ignore=['encoder'])

    def forward(self, x):
        return self.encoder.forward_features(x)

    def _step(self, batch, stage: str):
        x1, x2 = batch
        z1 = self.projector(self.encoder.forward_features(x1))
        z2 = self.projector(self.encoder.forward_features(x2))
        loss, parts = vicreg_loss(z1, z2, self.hparams.sim_coeff, self.hparams.std_coeff, self.hparams.cov_coeff)
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True, on_step=(stage == 'train'))
        for k, v in parts.items():
            self.log(f'{stage}_{k}', v, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
