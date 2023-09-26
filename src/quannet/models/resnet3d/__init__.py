from .model import ResNet3DModule
from quannet.model import Model


class ResNet3D(Model):
    @property
    def model_map(self):
        return {
            'model': ResNet3DModule,
        }
