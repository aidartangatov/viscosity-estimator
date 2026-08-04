from pathlib import Path

import sys
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quannet.ssl.mae import block_mask, MAELitModule  # noqa: E402
from quannet.models.resnet3d.model import ResNet3DModule  # noqa: E402


def test_block_mask_shapes_and_ratio():
    x = torch.randn(3, 1, 64, 64, 64)
    masked_x, mask = block_mask(x, mask_ratio=0.5, block_size=16)
    assert masked_x.shape == x.shape
    assert mask.shape == x.shape
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    # 64/16 = 4 blocks per axis -> 64 blocks total, ~50% masked
    frac_masked = mask[0].mean().item()
    assert 0.4 < frac_masked < 0.6


def test_block_mask_zeros_masked_region():
    x = torch.ones(1, 1, 32, 32, 32)
    masked_x, mask = block_mask(x, mask_ratio=0.5, block_size=8)
    assert torch.all(masked_x[mask.bool()] == 0)
    assert torch.all(masked_x[~mask.bool()] == 1)


def test_block_mask_rejects_non_divisible_block_size():
    x = torch.randn(1, 1, 33, 32, 32)
    with pytest.raises(AssertionError):
        block_mask(x, block_size=8)


def test_mae_lit_module_training_step_runs():
    encoder = ResNet3DModule()
    module = MAELitModule(encoder, mask_ratio=0.5, block_size=16)
    x = torch.randn(2, 1, 64, 64, 64)
    loss = module.training_step(x, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
