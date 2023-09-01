import os
import random
import tempfile
import subprocess
import multiprocessing
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Optional
from pathlib import Path
from itertools import islice

from Bio.PDB import PDBIO, PDBParser
import numpy as np

from quannet.utils import LOGGER, DEFAULT_CONFIG, DEFAULT_CONFIG_KEYS
from quannet.config import get_config

if TYPE_CHECKING:
    import Bio.PDB.Structure

APBS_PATH = os.environ['APBS_PATH']
PYTHON = os.environ['PYTHON']


class APBSWrapper:
    """
    A Python wrapper for the Adaptive Poisson-Boltzmann Solver (APBS) command-line tool.

    This class serves as an interface to APBS, simplifying the process of setting up, running, and post-processing
    electrostatic calculations. It can run PDB to PQR conversions, manage output and log files, prepare configuration
    files, and load results into memory.

    Attributes:
        input_path: The input file path containing the molecular structure.
        output_dir: The directory where all output files will be saved.
        artefacts_paths (dict): A mapping of various output and log files.
        grid_dim (int): The grid dimensions for the APBS calculations.
        grid_spacing (float): The spacing between grid points.
        inner_dielectric (float): The dielectric constant for the inner molecule.
        outer_dielectric (float): The dielectric constant for the surrounding solvent.
        keep_artefacts (bool): Whether to keep or remove output artefacts after calculations.
        delta (tuple): The grid spacing in the x, y, and z directions.
        network (dict): Holds the calculated grid values.
        min (list): Minimum grid value.
        has_run (bool): Flag indicating whether the APBS calculation has been executed.

    Methods:
        _run_pdb2pqr: Runs PDB to PQR conversion using pdb2pqr utility.
        _prepare_config_file: Prepares the configuration file for APBS run.
        run: Executes the APBS calculation.
        load_network: Loads the calculated grid into memory.
        check_probe: Checks if a point is within a certain threshold.
        remove_artefacts: Removes all output artefacts if 'keep_artefacts' is set to False.

    Note:
        It's advisable to call the 'run' method before accessing other functionalities.
    """

    line_preceding_network_coordinates = 'object 3 class array type double rank 0 items'

    def __init__(
        self,
        input_path,
        grid_dim,
        grid_spacing,
        output_dir,
        config_file_name='apbs_params.in',
        keep_artefacts=False,
        inner_dielectric=2.0,
        outer_dielectric=80,
        delta=(0.75, 0.75, 0.75),
    ):
        self.input_path = Path(input_path)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.artefacts_paths = {
            'pqr': self.output_dir / self.input_path.with_suffix('.pqr').name,
            'log': self.output_dir / self.input_path.with_suffix('.log').name,
            'config': self.output_dir / config_file_name,
            'mc': self.output_dir / 'io.mc',
            'accessibility': self.output_dir / f"accessibility_{self.input_path.with_suffix('.dx').name}",
            'potential': self.output_dir / f"potential_{self.input_path.with_suffix('.dx').name}",
            'charge': self.output_dir / f"charge_{self.input_path.with_suffix('.dx').name}",
        }

        self.grid_dim = grid_dim
        self.grid_spacing = grid_spacing
        self.inner_dielectric = inner_dielectric
        self.outer_dielectric = outer_dielectric

        self.keep_artefacts = keep_artefacts

        self.delta = delta
        self.network = {}
        self.min = None

        self.has_run = False

    def _run_pdb2pqr(self):
        """Convert PDB to PQR"""

        command = [
            PYTHON,
            '-m',
            'pdb2pqr',
            str(self.input_path),
            str(self.artefacts_paths['pqr']),
            '-ff=AMBER',
            '--whitespace',
            '--noopt',
            '--quiet',
        ]
        LOGGER.debug(f'Running PDB2PQR by command: {" ".join(command)}')
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _prepare_config_file(self, **kwargs):
        apbs_params = {
            'pdie': kwargs.get('pdie', 2.0),  # Specify the dielectric constant of the solute molecule
            'sdie': kwargs.get('sdie', 78.54),  # Specify the dielectric constant of the solvent molecule
            'srad': kwargs.get('srad', 1.4),  # Set up the radius of the solvent molecule
            'sdens': 10.0,  # Number of quadrature points per Å^2 to use in calculation surface terms
            'temp': kwargs.get('temp', 298.15),  # Specify of temperature of the system,
            'dime': ' '.join(map(str, [self.grid_dim + 1] * 3)),  # Set grid size by the all 3 dimensions
            'grid': ' '.join(map(str, self.delta)),
        }

        apbs_config_lines = (
            'read',
            f'mol pqr {str(self.artefacts_paths["pqr"])}',
            'end',
            '',
            'elec name viz',
            'mg-manual',
            'mol 1',
            'npbe',
            f'dime {apbs_params["dime"]}',
            f'grid {apbs_params["grid"]}',
            'gcent mol 1',
            'bcfl sdh',
            f'pdie {apbs_params["pdie"]}',
            f'sdie {apbs_params["sdie"]}',
            'chgm spl2',
            'srfm smol',
            f'srad {apbs_params["srad"]}',
            'swin 0.3',
            f'sdens {apbs_params["sdens"]}',
            f'temp {apbs_params["temp"]}',
            'calcenergy total',
            'calcforce no',
            f'write vdw dx {str(self.artefacts_paths["accessibility"].with_suffix(""))}',
            f'write pot dx {str(self.artefacts_paths["potential"].with_suffix(""))}',
            'end',
            '',
            'quit',
            '',
        )

        with self.artefacts_paths['config'].open('w') as f:
            f.write('\n'.join(apbs_config_lines))

    def run(self, **kwargs):
        self.has_run = True
        self._run_pdb2pqr()
        self._prepare_config_file(**kwargs)

        command = [APBS_PATH, str(self.artefacts_paths['config'])]
        with self.artefacts_paths['log'].open('w') as logf:
            try:
                subprocess.run(command, stdout=logf, check=True)
            except subprocess.CalledProcessError:
                raise RuntimeError(f'Error running APBS with command: {" ".join(command)}')

        LOGGER.debug(f'Running APBS by command: {" ".join(command)}')

        try:
            with self.artefacts_paths['log'].open('r') as f:
                for line in f:
                    if 'Grid center:' in line:
                        lst = line.split(':')[1].replace('(', '').replace(')', '').split(', ')
                        self.min = [float(x) for x in lst]
            self.min = [m - self.grid_dim * d / 2 for m, d in zip(self.min, self.delta)]
        except Exception as e:
            raise RuntimeError(f'Error reading the log file: {str(e)}')

    # Method obtains the grid and characteristics (charge, vdw potential, soluble, etc.) from the OpenDX output file
    def load_network(self, type_network='potential'):
        if not self.has_run:
            raise RuntimeError('The APBS calculation must be run before loading the network.')

        valid_network_types = ['accessibility', 'potential']

        if type_network not in valid_network_types:
            raise ValueError(f'Valid network types are: {", ".join(valid_network_types)}; you got {type_network}')

        with self.artefacts_paths[type_network].open('r') as f:
            for line in f:
                if line.startswith(self.line_preceding_network_coordinates):
                    break
            sliced_lines = islice((line for line in f), int(self.grid_dim / 3) * self.grid_dim * self.grid_dim)

            self.network[type_network] = np.array([line.split() for line in sliced_lines], dtype=np.float32).reshape(
                (self.grid_dim, self.grid_dim, self.grid_dim)
            )

    # TODO: check performance after understand why threshold is 2.0 while the network values are 1.0
    def check_probe(self, i, j, k, threshold=2.0):
        step = round(threshold / self.grid_spacing)
        for x in range(max(0, i - step), min(i + step, self.network['accessibility'].shape[0])):
            for y in range(max(0, j - step), min(j + step, self.network['accessibility'].shape[1])):
                for z in range(max(0, k - step), min(k + step, self.network['accessibility'].shape[2])):
                    if self.network['accessibility'][x, y, z] < threshold:
                        return True
        return False

    def remove_artefacts(self):
        for path in self.artefacts_paths.values():
            if path.exists():
                path.unlink()


def get_molecule(input_file: Union[str, Path]) -> 'Bio.PDB.Structure.Structure':
    """Retrieve a molecule's structure from a given input file using PDB parser.

    Args:
        input_file: Path to the file that contains the molecular structure data in PDB format.

    Returns:
        Bio.PDB.Structure.Structure: A structure object that represents the parsed molecule.
    """
    structure_name = Path(input_file).with_suffix('').name
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure(id=structure_name, file=input_file)
    return structure


def euler_rotate(
    structure: 'Bio.PDB.Structure.Structure', rotations: Tuple[float, float, float]
) -> 'Bio.PDB.Structure.Structure':
    """
    Apply Euler rotations to a molecular structure around the x, y, and z axes.

    Args:
        structure: The molecular structure to be rotated.
        rotations: A tuple containing the Euler angles (in radians) to rotate the structure around
                   the x, y, and z axes, respectively.

    Returns:
        Bio.PDB.Structure.Structure: The rotated molecular structure.

    Note:
        The function modifies the input structure in place and also returns it.
    """

    theta_x, theta_y, theta_z = rotations

    Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])

    # Combine rotations
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_coord = atom.get_coord()
                    atom_coord = np.dot(Rz, np.dot(Ry, np.dot(Rx, atom_coord)))
                    atom.set_coord(atom_coord)

    return structure


def save_molecule(structure: 'Bio.PDB.Structure.Structure', prefix: Optional[str] = None) -> Path:
    """
    Save the given molecular structure to a temporary PDB file.

    This function takes a Bio.PDB.Structure.Structure object as input and saves it to a temporary PDB file.
    The file is saved in the temporary directory of the operating system.

    Args:
        structure: The molecular structure to be saved.
        prefix: Optional string prefix for the temporary PDB file name.

    Returns:
        Path: The path to the created temporary PDB file.

    Note:
        The temporary PDB file will not be deleted automatically. Make sure to manage the temporary files as necessary.
    """
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)

    with tempfile.NamedTemporaryFile(suffix='.pdb', prefix=prefix, delete=False) as tmpfile:
        file_path = Path(tmpfile.name)
        LOGGER.debug(f'Created PDB file: {str(file_path)}')
        pdb_io.save(str(file_path))

    return file_path


def get_esp_array(
    structure,
    rotations: Optional[Tuple[float, float, float]] = None,
    grid_dim: int = 96,
    grid_spacing: float = 0.75,
    shell_width: float = 2.0,
    output_dir: Union[str, Path] = '.',
    remove_artefacts: bool = True,
) -> np.ndarray:
    """
    Generates an Electrostatic Potential (ESP) array based on the provided molecular structure.

    Given a molecular structure, this function optionally rotates the molecule using the specified angles,
    performs electrostatic calculations using the APBSWrapper class, and then creates an ESP array.

    Args:
        structure: The molecular structure for which to calculate the ESP.
        rotations: A tuple containing the angles in degrees to rotate the molecule around the X, Y, and Z axes.
        grid_dim: The grid dimensions for the electrostatic calculations. Defaults to 96.
        grid_spacing: The spacing between grid points for the electrostatic calculations. Defaults to 0.75.
        shell_width: The threshold for the electrostatic shell width. Defaults to 2.0.
        output_dir: The directory where all output files will be saved. Defaults to the current directory.
        remove_artefacts: Flag to indicate if output files should be removed after calculations. Defaults to True.

    Returns:
        An ESP array holding the calculated electrostatic potential.

    Notes:
        1. Make sure to call the 'run' method of APBSWrapper before accessing other functionalities.
        2. The electrostatic potential is set to zero for points outside the given shell_width threshold.
    """

    if rotations:
        structure = euler_rotate(structure, rotations)
        structure_name = f'{str(structure.get_id())}_{rotations[0]:06.2f}x{rotations[1]:06.2f}y{rotations[2]:06.2f}z_'
    else:
        structure_name = f'{str(structure.get_id())}_'
    structure_path = save_molecule(structure, prefix=structure_name)

    zap = APBSWrapper(
        input_path=structure_path,
        output_dir=output_dir,
        config_file_name=structure_path.with_suffix('.in').name,
        grid_dim=grid_dim,
        grid_spacing=grid_spacing,
        inner_dielectric=2.0,
        outer_dielectric=80,
    )

    zap.run()
    zap.load_network('potential')
    zap.load_network('accessibility')

    esp_array = np.zeros_like(zap.network['accessibility'])
    for i in range(zap.network['accessibility'].shape[0]):
        for j in range(zap.network['accessibility'].shape[1]):
            for k in range(zap.network['accessibility'].shape[2]):
                if zap.network['accessibility'][i, j, k] < 1.0 or zap.check_probe(i, j, k, shell_width):
                    esp_array[i, j, k] = zap.network['potential'][i, j, k]

    if remove_artefacts:
        zap.remove_artefacts()
    else:
        np_file_path = Path(output_dir, structure_name[:-1]).with_suffix('.npy')
        np.save(np_file_path, esp_array)

    return esp_array


def generate_esp_grids(
    structure_file: Union[str, Path],
    config: Union[Dict, SimpleNamespace] = DEFAULT_CONFIG,
    **kwargs,
) -> List[np.ndarray]:
    """Generate ESP grids from a given molecule file.

    Args:
        structure_file: The file path or object that contains the molecular structure data.
        config: Configuration settings for grid generation.
        **kwargs: Additional keyword arguments to override `config` settings or specify args for get_esp_array:
            remove_artefacts: Whether output files should be removed after calculations. Defaults to True.
            artefacts_dir: The directory where generated grids will be saved. Defaults to current directory.

    Raises:
        ValueError: If any required arguments are missing in the `config`.

    Returns:
        List of ESP arrays or a list of tuples containing ESP arrays and molecule objects, if `return_mol` is True.
    """

    config = get_config(config, {k: v for k, v in kwargs.items() if k in DEFAULT_CONFIG_KEYS})

    artefacts_dir = kwargs.get('artefacts_dir', '.')
    structure = get_molecule(structure_file)
    output_dir = Path(artefacts_dir, str(structure.get_id()))

    rotations_list = [
        (random.uniform(0, 180), random.uniform(0, 180), random.uniform(0, 180)) for _ in range(config.num_augmentations)
    ]

    processes = min(multiprocessing.cpu_count() - 1, config.processes)
    arguments_list = [
        (
            structure,
            rotations,
            config.grid_dim,
            config.grid_spacing,
            config.shell_width,
            output_dir,
            config.remove_artefacts,
        )
        for rotations in rotations_list
    ]
    with multiprocessing.Pool(processes=processes) as p:
        esp_array_output = p.starmap(get_esp_array, arguments_list)

    return esp_array_output


def load_esp_grids(artefacts_dir: Union[str, Path]) -> List[np.ndarray]:
    """
    Load Electrostatic Potential (ESP) grids from .npy files located in a specified directory.

    Parameters:
    - structure_file: Name of the structure file (minus the extension).
    - artefacts_dir: The directory where .npy files containing the ESP grids are stored.

    Returns:
    - A list of NumPy arrays, each representing an ESP grid.
    """

    artefacts_path = Path(artefacts_dir)
    esp_arrays_paths = list(artefacts_path.glob('*.npy'))

    if not esp_arrays_paths:
        LOGGER.warning('No matching ESP array files found!')
        return []

    esp_grids = [np.load(p) for p in esp_arrays_paths]

    return esp_grids
