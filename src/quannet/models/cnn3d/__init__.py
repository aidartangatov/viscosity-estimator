from .model import CNN3DModule
from quannet.model import Model


class CNN3D(Model):
    @property
    def model_map(self):
        return {
            'model': CNN3DModule,
        }
