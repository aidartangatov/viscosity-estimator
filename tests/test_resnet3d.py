from pathlib import Path

import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quannet.models import MODEL_ARCHS, get_model_class  # noqa: E402
from quannet.models.resnet3d.model import ResNet3DModule  # noqa: E402


def test_model_arch_resolution():
    from quannet.models.cnn3d import CNN3D
    from quannet.models.resnet3d import ResNet3D

    assert get_model_class('cnn3d') is CNN3D
    assert get_model_class('resnet3d') is ResNet3D
    assert set(MODEL_ARCHS) == {'cnn3d', 'resnet3d'}


def test_resnet3d_forward_shapes():
    model = ResNet3DModule().eval()
    x = torch.randn(2, 1, 96, 96, 96)
    with torch.no_grad():
        feats = model.forward_features(x)
        out = model(x)
    # encoder returns (B, C_last); head returns (B, 1)
    assert feats.shape == (2, 32)
    assert out.shape == (2, 1)


def test_forward_features_is_grid_size_agnostic():
    model = ResNet3DModule().eval()
    with torch.no_grad():
        f64 = model.forward_features(torch.randn(1, 1, 64, 64, 64))
        f96 = model.forward_features(torch.randn(1, 1, 96, 96, 96))
    assert f64.shape == f96.shape == (1, 32)
