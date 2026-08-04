"""3-D masked autoencoding (v2) pretext task for the ResNet-3D ESP encoder.

Random cubic blocks of the input ESP grid are zeroed out before the encoder
sees them (block masking, not per-voxel: a smoothly-varying electrostatic
field makes single masked voxels trivially interpolable from neighbors, so
masking has to remove whole neighborhoods to force the encoder to learn
structure rather than local smoothness). The encoder's pre-pool spatial
feature map (``forward_spatial_features``) is upsampled back to the input
grid by a small decoder, and the loss is MSE restricted to the masked voxels
- the classic MAE reconstruction objective, not a denoising one.

This is intentionally the lighter-weight of the two pretext tasks: it is
listed as "implemented or minimally tested" in the project plan, unlike
VICReg's "at least 1 complete run" requirement.
"""
from typing import Tuple

import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F


def block_mask(x: torch.Tensor, mask_ratio: float = 0.6, block_size: int = 12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero out random cubic blocks of a (B, 1, D, H, W) tensor.

    Returns (masked_x, mask) where mask is (B, 1, D, H, W) with 1s where
    voxels were masked (and should be scored in the reconstruction loss).
    """
    b, c, d, h, w = x.shape
    assert (
        d % block_size == 0 and h % block_size == 0 and w % block_size == 0
    ), f'grid dims {(d, h, w)} must be divisible by block_size={block_size}'
    nd, nh, nw = d // block_size, h // block_size, w // block_size
    n_blocks = nd * nh * nw
    n_masked = max(1, int(round(n_blocks * mask_ratio)))

    block_mask_flat = torch.zeros(b, n_blocks, device=x.device)
    for i in range(b):
        idx = torch.randperm(n_blocks, device=x.device)[:n_masked]
        block_mask_flat[i, idx] = 1.0

    mask = block_mask_flat.view(b, 1, nd, nh, nw)
    mask = F.interpolate(mask, scale_factor=block_size, mode='nearest')
    masked_x = x * (1 - mask)
    return masked_x, mask


class MAEDecoder(nn.Module):
    """Upsamples the encoder's spatial feature map back to the input grid."""

    def __init__(self, in_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, feat_map: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
        x = F.interpolate(feat_map, size=target_size, mode='trilinear', align_corners=False)
        return self.refine(x)


class MAELitModule(L.LightningModule):
    """Lightning wrapper: encoder trunk + lightweight reconstruction decoder.

    Only ``encoder`` is meant to be reused downstream; the decoder is
    pretext-task-only, discarded after pretraining.
    """

    def __init__(
        self,
        encoder: nn.Module,
        mask_ratio: float = 0.6,
        block_size: int = 12,
        decoder_hidden: int = 16,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.encoder = encoder
        enc_channels = self._infer_encoder_channels()
        self.decoder = MAEDecoder(enc_channels, decoder_hidden)
        self.save_hyperparameters(ignore=['encoder'])

    def _infer_encoder_channels(self, grid_dim: int = 96) -> int:
        was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, grid_dim, grid_dim, grid_dim)
            feat = self.encoder.forward_spatial_features(dummy)
        self.encoder.train(was_training)
        return feat.shape[1]

    def _step(self, batch, stage: str) -> torch.Tensor:
        x = batch
        masked_x, mask = block_mask(x, self.hparams.mask_ratio, self.hparams.block_size)
        feat_map = self.encoder.forward_spatial_features(masked_x)
        recon = self.decoder(feat_map, target_size=x.shape[-3:])
        loss = F.mse_loss(recon * mask, x * mask)
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True, on_step=(stage == 'train'))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
