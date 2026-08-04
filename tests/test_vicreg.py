from pathlib import Path

import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quannet.ssl.vicreg import vicreg_loss, VICRegLitModule, _infer_encoder_dim  # noqa: E402
from quannet.models.resnet3d.model import ResNet3DModule  # noqa: E402


def test_vicreg_loss_is_near_zero_for_identical_constant_variance_input():
    torch.manual_seed(0)
    z = torch.randn(16, 8)
    loss, parts = vicreg_loss(z, z.clone())
    assert parts['repr_loss'].item() == 0.0
    assert torch.isfinite(loss)


def test_vicreg_loss_penalizes_collapsed_embeddings():
    z_collapsed = torch.zeros(16, 8)
    z_varied = torch.randn(16, 8)
    loss_collapsed, _ = vicreg_loss(z_collapsed, z_collapsed.clone())
    loss_varied, _ = vicreg_loss(z_varied, z_varied.clone())
    # a constant embedding has zero std everywhere -> maximal std_loss penalty
    assert loss_collapsed.item() > loss_varied.item()


def test_infer_encoder_dim_matches_forward_features():
    model = ResNet3DModule()
    dim = _infer_encoder_dim(model, grid_dim=64)
    with torch.no_grad():
        feats = model.forward_features(torch.randn(1, 1, 64, 64, 64))
    assert dim == feats.shape[1]


def test_vicreg_lit_module_training_step_runs(tmp_path):
    encoder = ResNet3DModule()
    module = VICRegLitModule(encoder, projector_dims=(16, 16), grid_dim=64)
    x1 = torch.randn(4, 1, 64, 64, 64)
    x2 = torch.randn(4, 1, 64, 64, 64)
    loss = module.training_step((x1, x2), 0)
    assert torch.isfinite(loss)
    loss.backward()
    # encoder params received gradients (pretraining actually updates the trunk)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
