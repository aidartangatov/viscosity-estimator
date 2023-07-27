from quannet.model import Model
from .module import ResNet3DModule


class ResNet3D(Model):
    @property
    def model_map(self):
        return {
            'model': ResNet3DModule,
        }
