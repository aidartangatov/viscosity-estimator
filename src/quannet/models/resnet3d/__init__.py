from .model import ResNet3DModule

__all__ = ('ResNet3DModule', 'ResNet3D')


def __getattr__(name):
    # Lazy: quannet.model.Model pulls in Trainer -> Preprocessor ->
    # make_inputs, which needs numba/biopython/pdb2pqr - dependencies the SSL
    # training image doesn't install, since it only ever needs
    # ResNet3DModule directly, never the supervised-pipeline ResNet3D wrapper.
    if name == 'ResNet3D':
        from quannet.model import Model

        class ResNet3D(Model):
            @property
            def model_map(self):
                return {
                    'model': ResNet3DModule,
                }

        globals()['ResNet3D'] = ResNet3D  # cache: keeps repeated access identity-stable and skips __getattr__
        return ResNet3D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
