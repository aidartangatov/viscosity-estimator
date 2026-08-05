import importlib

# Map a `model_arch` config value to where its high-level wrapper class
# lives. Deliberately NOT imported eagerly: CNN3D pulls in the full
# supervised pipeline (quannet.model.Model -> Trainer -> Preprocessor ->
# make_inputs), which needs numba/biopython/pdb2pqr - dependencies the SSL
# training image doesn't install, since it only ever needs ResNet3D.
MODEL_ARCHS = {
    'cnn3d': ('quannet.models.cnn3d', 'CNN3D'),
    'resnet3d': ('quannet.models.resnet3d', 'ResNet3D'),
}


def get_model_class(arch: str = 'cnn3d'):
    """Resolve a `model_arch` string to its wrapper class (CNN3D / ResNet3D)."""
    if arch not in MODEL_ARCHS:
        raise ValueError(f"Unknown model_arch={arch!r}. Valid: {sorted(MODEL_ARCHS)}")
    module_name, class_name = MODEL_ARCHS[arch]
    return getattr(importlib.import_module(module_name), class_name)


def __getattr__(name):
    if name == 'CNN3D':
        return get_model_class('cnn3d')
    if name == 'ResNet3D':
        return get_model_class('resnet3d')
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ('CNN3D', 'ResNet3D', 'MODEL_ARCHS', 'get_model_class')
