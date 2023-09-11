from typing import List, Union, Optional
from pathlib import Path

from docker.errors import DockerException
import numpy as np

from quannet.utils import LOGGER, InsufficientDataError, common_deepest_directory
from quannet.make_inputs import load_esp_arrays
from quannet.docker.client import QuanDockerClient


def run_make_inputs(
    structure_paths: Union[List[Union[str, Path]], str, Path],
    grid_dim: int = 96,
    grid_spacing: float = 0.75,
    shell_width: float = 2.0,
    num_augmentations: Optional[int] = None,
    processes: Optional[int] = None,
    artefacts_dir: Union[str, Path] = '.',
    remove_artefacts: bool = True,
    train_mode: bool = False,
    image: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Run 'make_inputs' function for all pdb structures from the specified directory
    inside a Docker container with all necessary dependencies installed.

    This function accepts the same arguments as `make_inputs` with the addition of docker `image` and excluding
    the `return_arrays` flag.

    Args:
        structure_paths: Either a single path or a list of paths pointing to molecular structure files.
        grid_dim: Dimension of the cubic grid for the ESP computation.
        grid_spacing: Spacing between grid points.
        shell_width: Width of the shell used in the ESP computation.
        num_augmentations: Specifies the number of augmented ESP grids to produce for each structure.
                           Ignored if `train_mode` is False.
        processes: Number of parallel processes for concurrent grid generation.
        artefacts_dir: Directory to store the computed ESP grids.
        remove_artefacts: If True, any pre-existing ESP grids in the `artefacts_dir` are removed.
        train_mode: When set to True, generate additional augmented ESP grids for each file.
        image: Docker image name to use.

    Returns:
        List[np.ndarray]: List of ESP arrays.
    """

    if train_mode and (num_augmentations is None or processes is None):
        raise ValueError("In 'train_mode', both 'num_augmentations' and 'processes' must be provided.")
    if isinstance(structure_paths, (str, Path)):
        structure_paths = [structure_paths]

    source_dir = common_deepest_directory(structure_paths).resolve()
    artefacts_dir = Path(artefacts_dir).resolve()
    save_dir = artefacts_dir.parent

    docker_client = QuanDockerClient(source_dir=source_dir, save_dir=save_dir, image=image, module='make_inputs')
    container_structure_paths = [
        docker_client.container_datasets_dir / Path(path).resolve().relative_to(source_dir) for path in structure_paths
    ]
    container_artefacts_dir_path = docker_client.container_save_dir / artefacts_dir.relative_to(save_dir)

    command = docker_client.command_base + [
        '--structure_paths',
        *map(str, container_structure_paths),
        '--grid_dim',
        str(grid_dim),
        '--grid_spacing',
        str(grid_spacing),
        '--shell_width',
        str(shell_width),
        '--artefacts_dir',
        str(container_artefacts_dir_path),
    ]
    if remove_artefacts:
        command.append('--remove_artefacts')
    if train_mode:
        command.extend(['--num_augmentations', str(num_augmentations), '--processes', str(processes)])

    volumes = docker_client.volumes_base.copy()

    try:
        container = docker_client.client.containers.run(
            docker_client.image, volumes=volumes, command=command, remove=True, detach=True
        )
        for line in container.logs(stream=True):
            LOGGER.warning(line.decode('utf-8'))
    except DockerException as ex:
        raise ValueError(ex)

    inputs = []
    for structure_artefacts_dir in artefacts_dir.iterdir():
        structure_arrays = load_esp_arrays(dir_path=structure_artefacts_dir)
        size = len(structure_arrays)
        if train_mode and size != num_augmentations:
            raise InsufficientDataError(f'Only {size} ESP arrays were found, while {num_augmentations} expected.')
        inputs.extend(load_esp_arrays(dir_path=structure_artefacts_dir))

    return inputs
