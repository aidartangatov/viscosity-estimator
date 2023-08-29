from quannet.model import Model
from .model import CNN3DModule


class CNN3D(Model):
    @property
    def model_map(self):
        return {
            'model': CNN3DModule,
        }
