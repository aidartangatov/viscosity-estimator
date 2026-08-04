__version__ = '0.0.1'


def __getattr__(name):
    if name == 'QuanNet':
        from quannet.models import CNN3D

        return CNN3D
    raise AttributeError(f"module 'quannet' has no attribute {name!r}")
