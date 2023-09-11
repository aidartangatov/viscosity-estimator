from typing import Dict, List, Union, Optional
from pathlib import Path

import docker

from quannet.utils import SETTINGS


class QuanDockerClient:
    """
    Runs model input generation in a docker container with mounted source structures.
    """

    def __init__(
        self,
        source_dir: Union[str, Path],
        save_dir: Union[str, Path],
        image: Optional[str] = None,
        module: str = 'make_inputs',
    ):
        self.image = image
        self.source_dir = Path(source_dir)
        self.save_dir = Path(save_dir)
        self.module = module

    @property
    def command_base(self) -> List[str]:
        return ['python3', '-m', self.module]

    @property
    def container_datasets_dir(self) -> Path:
        return Path(SETTINGS['container_datasets_dir'], self.source_dir.name)

    @property
    def container_save_dir(self) -> Path:
        return Path(SETTINGS['container_runs_dir'], self.save_dir.name)

    @property
    def volumes_base(self) -> Dict[str, Dict[str, str]]:
        return {
            str(self.source_dir): {'bind': str(self.container_datasets_dir), 'mode': 'ro'},
            str(self.save_dir): {'bind': str(self.container_save_dir), 'mode': 'rw'},
        }

    @property
    def client(self) -> docker.client.DockerClient:
        client = docker.from_env()
        return client
