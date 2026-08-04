from .cnn3d import CNN3D
from .resnet3d import ResNet3D

# Map a `model_arch` config value to its high-level wrapper class.
MODEL_ARCHS = {
    'cnn3d': CNN3D,
    'resnet3d': ResNet3D,
}


def get_model_class(arch: str = 'cnn3d'):
    """Resolve a `model_arch` string to its wrapper class (CNN3D / ResNet3D)."""
    if arch not in MODEL_ARCHS:
        raise ValueError(f"Unknown model_arch={arch!r}. Valid: {sorted(MODEL_ARCHS)}")
    return MODEL_ARCHS[arch]


__all__ = ('CNN3D', 'ResNet3D', 'MODEL_ARCHS', 'get_model_class')
