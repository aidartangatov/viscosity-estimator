import multiprocessing
import os
import random
import string
import subprocess
import tempfile
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, PDBParser
from sklearn.model_selection import train_test_split

from quannet.utils import LOGGER, ArrayLike

APBS_PATH = os.environ['APBS_PATH']
PYTHON = os.environ['PYTHON']

INPUT_MOL_KEY = 'input_mol'
ROT_X_KEY = 'rot_x'
ROT_Y_KEY = 'rot_y'
ROT_Z_KEY = 'rot_z'
GRID_SPACING_KEY = 'grid_spacing'
GRID_DIM_KEY = 'grid_dim'
SHELL_WIDTH_KEY = 'shell_width'
NUM_AUGMENTATIONS_KEY = 'num_augmentations'
DEFAULT_GRID_PARAMS = {
    GRID_DIM_KEY: 96,
    GRID_SPACING_KEY: 0.75,
    SHELL_WIDTH_KEY: 2.0,
    NUM_AUGMENTATIONS_KEY: 10,
}


class APBSWrapper:
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

        self.artefacts_path = {
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
            str(self.artefacts_path['pqr']),
            '-ff=AMBER',
            '--whitespace',
            '--noopt',
        ]
        print(f'Running PDB2PQR by command: {" ".join(command)}')
        subprocess.run(command)

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
            f'mol pqr {str(self.artefacts_path["pqr"])}',
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
            # f'write charge dx {str(self.artefacts_path["accessibility"].with_suffix(""))}',
            f'write vdw dx {str(self.artefacts_path["accessibility"].with_suffix(""))}',
            f'write pot dx {str(self.artefacts_path["potential"].with_suffix(""))}',
            'end',
            '',
            'quit',
            '',
        )

        with self.artefacts_path['config'].open('w') as f:
            f.write('\n'.join(apbs_config_lines))

    def run(self, **kwargs):
        self.has_run = True
        self._run_pdb2pqr()
        self._prepare_config_file(**kwargs)

        command = [APBS_PATH, str(self.artefacts_path['config'])]
        with self.artefacts_path['log'].open('w') as logf:
            try:
                subprocess.run(command, stdout=logf, check=True)
            except subprocess.CalledProcessError:
                raise RuntimeError(f'Error running APBS with command: {" ".join(command)}')

        print(f'Running APBS by command: {" ".join(command)}')

        try:
            with self.artefacts_path['log'].open('r') as f:
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

        with self.artefacts_path[type_network].open('r') as f:
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
        for path in self.artefacts_path.values():
            if path.exists():
                path.unlink()


def get_molecule(input_file):
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure('pdb', input_file)
    return structure


def euler_rotate(structure, theta_x, theta_y, theta_z):
    """
    Apply Euler rotations to a molecular structure around the x, y, and z axes.
    """
    # Define rotation matrices
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


def generate_random_string(length=10):
    """Generate random string of specified length."""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def save_molecule(structure):
    """Save the molecule to a temporary PDB file."""
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmpfile:
        file_path = Path(tmpfile.name)
        print(f'Created PDB file: {str(file_path)}')
        pdb_io.save(str(file_path))

    return file_path


def get_esp_array(params, output_dir, remove_artefacts=True, return_mol=False):
    """Generate ESP array from given parameters."""

    # Assuming euler_rotate and get_molecule are available in this scope
    structure = euler_rotate(params[INPUT_MOL_KEY], params[ROT_X_KEY], params[ROT_Y_KEY], params[ROT_Z_KEY])
    structure_path = save_molecule(structure)

    zap = APBSWrapper(
        input_path=structure_path,
        output_dir=output_dir,
        config_file_name=structure_path.with_suffix('.in').name,
        grid_dim=params[GRID_DIM_KEY],
        grid_spacing=params[GRID_SPACING_KEY],
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
                if zap.network['accessibility'][i, j, k] < 1.0 or zap.check_probe(i, k, k, params[SHELL_WIDTH_KEY]):
                    esp_array[i, j, k] = zap.network['potential'][i, j, k]

    if remove_artefacts:
        zap.remove_artefacts()

    if return_mol:
        return esp_array, structure
    return esp_array


def generate_esp_grids(structure_file, output_dir, return_mol=False, processes=4, **kwargs):
    """Generate ESP grids from the given molecule file."""

    # Assuming get_molecule is available in this scope
    structure = get_molecule(structure_file)

    grid_dim = kwargs.get(GRID_DIM_KEY, DEFAULT_GRID_PARAMS[GRID_DIM_KEY])
    grid_spacing = kwargs.get(GRID_SPACING_KEY, DEFAULT_GRID_PARAMS[GRID_SPACING_KEY])
    shell_width = kwargs.get(SHELL_WIDTH_KEY, DEFAULT_GRID_PARAMS[SHELL_WIDTH_KEY])
    num_augmentations = kwargs.get(NUM_AUGMENTATIONS_KEY, DEFAULT_GRID_PARAMS[NUM_AUGMENTATIONS_KEY])

    shared_params = {
        INPUT_MOL_KEY: structure,
        GRID_DIM_KEY: grid_dim,
        GRID_SPACING_KEY: grid_spacing,
        SHELL_WIDTH_KEY: shell_width,
    }

    params_list = [
        {
            ROT_X_KEY: random.uniform(0, 180),
            ROT_Y_KEY: random.uniform(0, 180),
            ROT_Z_KEY: random.uniform(0, 180),
            **shared_params,
        }
        for _ in range(num_augmentations)
    ]

    processes = min(multiprocessing.cpu_count() - 1, processes)
    with multiprocessing.Pool(processes=processes) as p:
        esp_array_output = p.map(partial(get_esp_array, return_mol=return_mol, output_dir=output_dir), params_list)

    if return_mol:
        return [(esp_array.reshape((1, *esp_array.shape)), mol) for esp_array, mol in esp_array_output]
    return esp_array_output


def find_stratification_bins(y: ArrayLike, max_bins_stratify: int = 5, min_samples_per_bin: int = 2):
    """
    Attempt to find a suitable number of bins to stratify the data into.

    Args:
        y: The array of labels to stratify by.
        max_bins_stratify: The maximum number of bins to try for stratification.
        min_samples_per_bin: Minimum number of samples required per bin.

    Returns:
        pandas.Series with the same length as `y` indicating the bin each sample belongs to, or
        None if no suitable stratification is possible.
    """
    for n_bins in range(max_bins_stratify, 1, -1):
        stratify = pd.Series(pd.qcut(y, q=n_bins, labels=False))
        if all(stratify.value_counts() >= min_samples_per_bin):
            return stratify
    return None


def split_indices(
    y: ArrayLike, val_size: float = 0.1, max_bins_stratify: Optional[int] = None
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Split the dataset into training and validation sets.

    Args:
        y: The array of labels.
        val_size: The proportion of the dataset to include in the test split.
        max_bins_stratify: Maximum number of bins to use for stratification.

    Returns:
        Indices for training set.
        Indices for validation set.
    """

    stratify = None
    if max_bins_stratify is not None:
        stratify = find_stratification_bins(y, max_bins_stratify=max_bins_stratify)

    if stratify is not None:
        unique_bins = len(stratify.unique())
        min_test_size = unique_bins / len(y)
        if val_size < min_test_size:
            LOGGER.warning(
                f'val_size {val_size} is too small for the number of classes {unique_bins}. Skipping stratification.'
            )
            stratify = None

    train_index, val_index = train_test_split(range(len(y)), test_size=val_size, stratify=stratify, random_state=42)

    return train_index, val_index
