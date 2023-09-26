from tqdm import tqdm
from numba import jit
from typing import List, Tuple, Union, Optional, TYPE_CHECKING
from Bio.PDB import PDBIO, PDBParser
from pathlib import Path
from itertools import islice

import os
import numpy as np
import random
import logging
import argparse
import subprocess
import multiprocessing

if TYPE_CHECKING:
    import Bio.PDB.Structure


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
        shell_width: Threshold for the electrostatic shell width.
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
        shell_width,
        output_dir,
        logger,
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
            'pdb': self.input_path,
            'pqr': self.output_dir / self.input_path.with_suffix('.pqr').name,
            'log': self.output_dir / self.input_path.with_suffix('.log').name,
            'config': self.output_dir / config_file_name,
            'mc': self.output_dir / 'io.mc',
            'accessibility': self.output_dir / f"accessibility_{self.input_path.with_suffix('.dx').name}",
            'potential': self.output_dir / f"potential_{self.input_path.with_suffix('.dx').name}",
            'charge': self.output_dir / f"charge_{self.input_path.with_suffix('.dx').name}",
        }

        self.logger = logger

        self.grid_dim = grid_dim
        self.grid_spacing = grid_spacing
        self.shell_width = shell_width
        self.inner_dielectric = inner_dielectric
        self.outer_dielectric = outer_dielectric

        self.keep_artefacts = keep_artefacts

        self.delta = delta
        self.network = {}
        self.min = None

        self.has_run = False

    @property
    def apbs(self):
        return os.environ['APBS']

    @property
    def python(self):
        return os.environ['PYTHON']

    def _run_pdb2pqr(self):
        """Convert PDB to PQR"""

        command = [
            self.python,
            '-m',
            'pdb2pqr',
            str(self.input_path),
            str(self.artefacts_paths['pqr']),
            '-ff=AMBER',
            '--whitespace',
            '--noopt',
            '--quiet',
        ]
        if self.logger:
            self.logger.debug(f'Running PDB2PQR by command: {" ".join(command)}')
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

        command = [self.apbs, str(self.artefacts_paths['config'])]
        with self.artefacts_paths['log'].open('w') as logf:
            try:
                subprocess.run(command, stdout=logf, check=True)
            except subprocess.CalledProcessError:
                raise RuntimeError(f'Error running APBS with command: {" ".join(command)}')

        if self.logger:
            self.logger.debug(f'Running APBS by command: {" ".join(command)}')

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

    def process_array(self):
        if not self.has_run:
            raise RuntimeError('The APBS calculation must be run before processing network arrays.')
        return _process_array(
            self.network['accessibility'], self.network['potential'], self.shell_width, self.grid_spacing
        )

    def remove_artefacts(self):
        for path in self.artefacts_paths.values():
            if path.exists():
                path.unlink()


@jit(nopython=True)
def _check_probe(i: int, j: int, k: int, accessibility: np.ndarray, threshold: float, grid_spacing: float) -> bool:
    """
    Check if a given point (i, j, k) is within the proximity of any inaccessible point.

    This function verifies if, within a defined radius around a point (i, j, k), there exists
    any point that is deemed inaccessible (defined by the value being less than the threshold).

    Parameters:
        i, j, k: The x, y, z indices of the point in the 3D grid.
        accessibility: A 3D array representing the accessibility of each point.
        threshold: A value below which a point in the accessibility array is considered inaccessible.
        grid_spacing: The spacing between points in the grid.

    Returns:
        True if an inaccessible point exists within the radius, False otherwise.
    """
    step = round(threshold / grid_spacing)
    for x in range(max(0, i - step), min(i + step, accessibility.shape[0])):
        for y in range(max(0, j - step), min(j + step, accessibility.shape[1])):
            for z in range(max(0, k - step), min(k + step, accessibility.shape[2])):
                if accessibility[x, y, z] < threshold:
                    return True
    return False


@jit(nopython=True)
def _process_array(
    accessibility: np.ndarray, potential: np.ndarray, shell_width: float, grid_spacing: float
) -> np.ndarray:
    """
    Process the accessibility and potential arrays to create a new array.

    This function evaluates each point in the accessibility array. If the value at the point is
    below a certain threshold or if the point is within the proximity of an inaccessible point,
    the corresponding value in the potential array is copied to the new array.

    Parameters:
        accessibility: A 3D numpy array representing the accessibility of each point.
        potential: A 3D array representing the potential at each point.
        shell_width: The threshold value defining the 'shell' around inaccessible regions.
        grid_spacing: The spacing between points in the grid.

    Returns:
        A new 3D array processed as per the rules defined above.
    """
    esp_array = np.zeros_like(accessibility)
    for i in range(accessibility.shape[0]):
        for j in range(accessibility.shape[1]):
            for k in range(accessibility.shape[2]):
                if accessibility[i, j, k] < 1.0 or _check_probe(i, j, k, accessibility, shell_width, grid_spacing):
                    esp_array[i, j, k] = potential[i, j, k]
    return esp_array


def load_molecule(input_file: Union[str, Path]) -> 'Bio.PDB.Structure.Structure':
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


def save_molecule(structure: 'Bio.PDB.Structure.Structure', file_path: Union[str, Path]) -> None:
    """
    Save the given molecular structure to a specified file_path.

    This function takes a Bio.PDB.Structure.Structure object as input and saves it to a PDB file at the specified path.

    Args:
        structure: The molecular structure to be saved.
        file_path: The location where the molecular structure should be saved as a PDB file.
    Raises:
        ValueError: If the provided file_path does not have a '.pdb' extension.
    """

    file_path = Path(file_path)
    if file_path.suffix.lower() != '.pdb':
        raise ValueError("The provided path must have a '.pdb' extension.")
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(str(file_path))


def get_esp_array(
    structure_path: Union[str, Path],
    rotations: Optional[Tuple[float, float, float]] = None,
    grid_dim: int = 96,
    grid_spacing: float = 0.75,
    shell_width: float = 2.0,
    output_dir: Union[str, Path] = '.',
    remove_artefacts: bool = True,
    write_esp_array: bool = True,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Generate an Electrostatic Potential (ESP) array for the given molecular structure.

    Given the path to a molecular structure file, this function:
    1. Loads the molecular structure from the provided file.
    2. If specified, rotates the molecule using the given angles.
    3. Saves the optionally rotated structure to a temporary PDB file within the designated output directory.
    4. Utilizes the APBSWrapper class to:
       a. Initialize the ESP calculations setup with the provided grid configurations and dielectric properties.
       b. Run the electrostatic calculations.
       c. Load the potential and accessibility network from the results.
    5. Constructs an ESP array, where points outside a given shell_width are set to zero.
    6. If set, removes calculation artefacts and saves the ESP array to a file.
    The result is an ESP array representing the electrostatic potential of the molecule.

    Args:
        structure_path: Path to the molecular structure file.
        rotations: Angles in degrees to rotate the molecule around the X, Y, and Z axes.
        grid_dim: Grid dimensions for the electrostatic calculations.
        grid_spacing: Spacing between grid points.
        shell_width: Threshold for the electrostatic shell width.
        output_dir: Directory for saving output files.
        remove_artefacts: Whether to remove intermediate files post-calculation.
        write_esp_array: Whether to write the ESP array to a file.
        logger: Logger instance for logs.

    Returns:
        An ESP array holding the calculated electrostatic potential.

    Notes:
        1. Make sure to call the 'run' method of APBSWrapper before accessing other functionalities.
        2. The electrostatic potential is set to zero for points outside the given shell_width threshold.
    """

    structure_path = Path(structure_path)
    structure = load_molecule(structure_path)
    structure_name = structure_path.with_suffix('').name
    if rotations:
        structure = euler_rotate(structure, rotations)
        structure_name = f'{structure_name}_{rotations[0]:06.2f}x{rotations[1]:06.2f}y{rotations[2]:06.2f}z'
    structure_path = Path(output_dir, structure_name + '.pdb')
    save_molecule(structure, structure_path)

    zap = APBSWrapper(
        input_path=structure_path,
        output_dir=output_dir,
        config_file_name=structure_path.with_suffix('.in').name,
        grid_dim=grid_dim,
        grid_spacing=grid_spacing,
        shell_width=shell_width,
        inner_dielectric=2.0,
        outer_dielectric=80,
        logger=logger,
    )

    zap.run()
    zap.load_network('potential')
    zap.load_network('accessibility')
    esp_array = zap.process_array()

    if remove_artefacts:
        zap.remove_artefacts()
    if write_esp_array:
        np_file_path = Path(output_dir, structure_name).with_suffix('.npy')
        np.save(np_file_path, esp_array)

    return esp_array


def get_esp_array_rotations(
    structure_path: Union[str, Path],
    grid_dim: int = 96,
    grid_spacing: float = 0.75,
    shell_width: float = 2.0,
    output_dir: Union[str, Path] = '.',
    remove_artefacts: bool = True,
    write_esp_array: bool = True,
    logger: Optional[logging.Logger] = None,
    num_augmentations: int = 10,
    processes: int = 4,
) -> List[np.ndarray]:
    """
    Generate multiple ESP grids for a given molecule file through various rotations.

    This function leverages multiprocessing to concurrently generate a series of ESP grids by applying
    different random rotations to the input molecule. Each grid is generated in a separate process,
    speeding up the process considerably when multiple CPU cores are available.

    For a detailed explanation on the grid generation process and other shared parameters, see
    the `generate_single_esp_grid` function.

    Args:
        num_augmentations: Number of random rotations (data augmentations) to generate.
        processes: Number of parallel processes to use for concurrent grid generation. This dictates
                   the number of cores the function can utilize. Defaults to 4, but it's constrained
                   by the available CPU cores.

    Returns:
        List[np.ndarray]: List of generated ESP arrays for each rotation.
    """

    rotations_list = [
        (random.uniform(0, 180), random.uniform(0, 180), random.uniform(0, 180)) for _ in range(num_augmentations)
    ]

    processes = min(multiprocessing.cpu_count() - 1, processes)
    arguments_list = [
        (
            structure_path,
            rotations,
            grid_dim,
            grid_spacing,
            shell_width,
            output_dir,
            remove_artefacts,
            write_esp_array,
            logger,
        )
        for rotations in rotations_list
    ]
    with multiprocessing.Pool(processes=processes) as p:
        esp_array_output = p.starmap(get_esp_array, arguments_list)

    return esp_array_output


def load_esp_arrays(dir_path: Union[str, Path], logger: Optional[logging.Logger] = None) -> List[np.ndarray]:
    """
    Load Electrostatic Potential (ESP) grids from .npy files located in a specified directory.

    Args:
        dir_path: The directory where .npy files containing the ESP grids are stored.
        logger: Logger instance for logs.

    Returns:
        A list of NumPy arrays, each representing an ESP grid.
    """

    dir_path = Path(dir_path)
    esp_arrays_paths = list(dir_path.glob('*.npy'))

    if not esp_arrays_paths:
        if logger:
            logger.warning('No matching ESP array files found!')
        return []

    esp_arrays = [np.load(p) for p in esp_arrays_paths]

    return esp_arrays


get_esp_array_rotations.__doc__ += '\n\n' + (get_esp_array.__doc__ or '')


def get_structure_paths(structures: Union[str, Path]) -> Union[Path, List[Path]]:
    """
    Retrieve all the paths to PDB files in the specified directory.

    Args:
        structures: The directory path where the PDB files are located or a single PDB file.

    Returns:
        A path or list of paths pointing to the PDB files found in the specified directory.

    Raises:
        FileNotFoundError: If no PDB files are found in the specified directory.
    """
    structures = Path(structures)
    if structures.is_dir():
        structure_paths = [p for p in structures.iterdir() if p.name.endswith('.pdb')]
        if not structure_paths:
            raise FileNotFoundError('No .pdb files found in the specified directory.')
        return structure_paths
    else:
        if structures.name.endswith('pdb'):
            return structures
        else:
            raise FileNotFoundError('Provided path is not a .pdb file.')


def make_inputs(
    structure_paths: Union[List[Union[str, Path]], str, Path],
    grid_dim: int = 96,
    grid_spacing: float = 0.75,
    shell_width: float = 2.0,
    num_augmentations: Optional[int] = None,
    processes: Optional[int] = None,
    artefacts_dir: Union[str, Path] = '.',
    remove_artefacts: bool = True,
    train_mode: bool = False,
    return_arrays: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Optional[List[np.ndarray]]:
    """
    Generate Electrostatic Potential (ESP) grids for provided molecular structures and save them to a specified
    directory.

    Given a list of paths or a single path to molecular structure files, this function processes each file to compute
    the corresponding ESP grids. The grids are saved in subdirectories within the provided `artefacts_dir`. If the
    `train_mode` is enabled, additional augmented ESP grids are generated for each file.

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
        return_arrays: If True, return a list of the computed ESP arrays.
        logger: Logger instance for logs.

    Returns:
        List of ESP arrays if `return_arrays` is set to True, otherwise None.

    Notes:
        - When 'train_mode' is disabled, 'num_augmentations' and 'processes' are ignored.
    """

    if not train_mode and (num_augmentations is not None or processes is not None):
        if logger:
            logger.warning(
                "'num_augmentations' and/or 'processes' param is not None, ",
                "however it is ignored when 'train_mode' is False.",
            )

    if isinstance(structure_paths, (Path, str)):
        structure_paths = [structure_paths]

    esp_arrays = []
    for structure_path in tqdm(structure_paths):
        structure_path = Path(structure_path)
        output_dir = Path(artefacts_dir, structure_path.with_suffix('').name)
        output_dir.mkdir(parents=False)

        shared_params = {
            'structure_path': structure_path,
            'grid_dim': grid_dim,
            'grid_spacing': grid_spacing,
            'shell_width': shell_width,
            'output_dir': output_dir,
            'remove_artefacts': remove_artefacts,
            'write_esp_array': True,
        }

        if train_mode:
            esp_array = get_esp_array_rotations(
                **shared_params,
                num_augmentations=num_augmentations,
                processes=processes,
            )
            if return_arrays:
                esp_arrays.extend(esp_array)
        else:
            esp_array = get_esp_array(**shared_params)
            if return_arrays:
                esp_arrays.append(esp_array)
    if return_arrays:
        return esp_arrays


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ESP array files')

    parser.add_argument(
        '--structure_paths',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to the molecular structure file(s) or directory containing them.',
    )
    parser.add_argument('--grid_dim', type=int, default=96, help='Dimension of the cubic grid for ESP computation.')
    parser.add_argument('--grid_spacing', type=float, default=0.75, help='Spacing between grid points.')
    parser.add_argument('--shell_width', type=float, default=2.0, help='Width of the shell used in the ESP computation.')
    parser.add_argument('--artefacts_dir', type=str, default='.', help='Directory where ESP grids should be saved.')
    parser.add_argument(
        '--remove_artefacts', action='store_true', help='If set, removes any pre-existing ESP grids in artefacts_dir.'
    )
    parser.add_argument(
        '--num_augmentations', type=int, help='Number of augmented ESP arrays to generate per structure file.'
    )
    parser.add_argument('--processes', type=int, help='Number of parallel processes to use.')

    args = parser.parse_args()

    train_mode = args.num_augmentations is not None

    make_inputs(
        structure_paths=args.structure_paths,
        grid_dim=args.grid_dim,
        grid_spacing=args.grid_spacing,
        shell_width=args.shell_width,
        num_augmentations=args.num_augmentations,
        processes=args.processes,
        artefacts_dir=args.artefacts_dir,
        remove_artefacts=args.remove_artefacts,
        train_mode=train_mode,
    )
