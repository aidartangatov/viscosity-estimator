import logging
import logging.config
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Union

import numpy as np
import pandas as pd
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
# TODO: set right path, better do in settings
TEST_STRUCTURES = ROOT / 'structures'
DEFAULT_CONFIG_PATH = ROOT / 'config/default.yaml'
LOGGING_NAME = 'quannet'
ArrayLike = Union[list, pd.Series, np.ndarray]


def search_file(file: str, dir: Union[str, Path] = 'config') -> str:
    """
    Search the file (if necessary) by name or pattern and return its path.

    Args:
        file: The name, glob pattern, or path of the file.
        dir: The name or path of directory for recursive search

    Returns:
        The file path if it exists.
    """
    file_path = Path(file).resolve()

    if file_path.exists():
        return str(file_path)

    # Search for the file recursively within the dir directory under ROOT
    if isinstance(dir, str):
        matching_files = list(ROOT.joinpath(dir).rglob(file))
    # Search for the file recursively within the dir path
    elif isinstance(dir, Path):
        matching_files = list(dir.rglob(file))
    else:
        raise ValueError(f"'dir' is expected to be a str or pathlib.Path object, got {type(dir)}")

    if not matching_files:
        raise FileNotFoundError(f"'{file}' does not exist")
    elif len(matching_files) > 1:
        raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {matching_files}")

    return str(matching_files[0])


class SettingsManager(dict):
    """
    Manages settings
    """

    def __init__(self):
        import copy

        root = Path()
        for d in Path(__file__).parents:
            if (d / '.git').is_dir():
                root = d

        self.defaults = {
            'runs_dir': str(root / 'runs'),
            'datasets_dir': str(root / 'datasets'),
            'arefacts_dir_name': 'artefacts',
        }

        super().__init__(copy.deepcopy(self.defaults))


SETTINGS = SettingsManager()
DATASETS_DIR = Path(SETTINGS['datasets_dir'])


class InsufficientDataError(Exception):
    def __init__(self, message='Insufficient data based on the given threshold'):
        self.message = message
        super().__init__(self.message)


class DuplicatedDataError(Exception):
    def __init__(self, message='Duplicated .pdb files found'):
        self.message = message
        super().__init__(self.message)


def check_dataset(dataset: str, path_csv_col: str = 'path', target_csv_col: str = 'target', min_structures: int = 1):
    """
    Checks a dataset for inconsistencies in file paths.

    Args:
        dataset: The dataset directory or name.
        path_csv_col: The column name in the csv file that contains file paths.
        target_csv_col: The column name in the csv file that contains target values.
        min_structures: Minimum number of .pdb files with corresponding target values.

    Returns:
        A dict with keys 'paths' and 'targets' containing lists of valid pdb paths and corresponding target values.

    Raises:
        InsufficientDataError: If there are fewer than `threshold` files with corresponding target values.
    """
    dataset = Path(dataset)
    data_dir = (dataset if dataset.is_dir() else (DATASETS_DIR / dataset)).resolve()
    target_csv_path = search_file('**/*.csv', data_dir)
    target_df = pd.read_csv(target_csv_path, usecols=[path_csv_col, target_csv_col])

    structure_relpaths_from_csv = target_df[path_csv_col].tolist()
    structure_paths = [Path(data_dir / path) for path in structure_relpaths_from_csv]
    structure_relpaths_pathlib_compatible = [str(path.relative_to(data_dir)) for path in structure_paths]

    # Check for duplicates
    duplicates = [item for item, count in Counter(structure_relpaths_pathlib_compatible).items() if count > 1]
    if duplicates:
        message = '\n'.join(duplicates)
        messsage = f'Found duplicate paths in CSV:\n{message}'
        raise DuplicatedDataError(messsage)
    path2str = dict(zip(structure_relpaths_pathlib_compatible, structure_relpaths_from_csv))

    # Check if paths in CSV actually exist
    nonexistent_paths = [path for path in structure_relpaths_pathlib_compatible if not (data_dir / path).exists()]
    if nonexistent_paths:
        message = '\n'.join(nonexistent_paths)
        LOGGER.warning(f'Paths from CSV that do not exist:\n{message}')

    all_pdb_files = [str(path.relative_to(data_dir)) for path in data_dir.rglob('*.pdb')]

    # Check which .pdb files in the directory aren't mentioned in the CSV
    not_in_csv = set(all_pdb_files) - set(structure_relpaths_pathlib_compatible)
    if not_in_csv:
        message = '\n'.join(not_in_csv)
        LOGGER.warning(f'.pdb files not found in CSV:\n{message}')

    # Check which .pdb files are mentioned in the CSV but aren't in the directory
    not_in_dir = set(structure_relpaths_pathlib_compatible) - set(all_pdb_files)
    if not_in_dir:
        message = '\n'.join(not_in_dir)
        LOGGER.warning(f'.pdb paths in CSV but not in directory:\n{message}')

    # Count valid .pdb files with targets
    valid_pdb_with_targets = len(set(all_pdb_files).intersection(set(structure_relpaths_pathlib_compatible)))

    LOGGER.info(f'Found {valid_pdb_with_targets} .pdb files with corresponding target values.')

    if valid_pdb_with_targets < min_structures:
        raise InsufficientDataError(f'Found fewer than {min_structures} .pdb files with corresponding target values.')

    valid_rel_paths = list(set(all_pdb_files).intersection(set(structure_relpaths_pathlib_compatible)))
    valid_targets = target_df[target_df[path_csv_col].isin([path2str[p] for p in valid_rel_paths])][
        target_csv_col
    ].tolist()
    valid_paths = [Path(data_dir / path).resolve() for path in valid_rel_paths]

    return {'paths': valid_paths, 'targets': valid_targets}


def increment_path(path: Union[str, Path], exist_ok: bool = False, sep: str = '', mkdir: bool = False) -> Path:
    """
    Increment a file or directory path.

    If the path exists and `exist_ok` is not set to True, this function will append a number, preceded by `sep`,
    to the end of the path. If the path is a file, its file extension will be preserved. If the path is a directory,
    the number will be appended directly to its end.

    For example:
        runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    Args:
        path: The path to be incremented.
        exist_ok: If True, existing paths are returned as-is. Otherwise, they're incremented. Defaults to False.
        sep: Separator to use between the path and the incrementation number. Defaults to ''.
        mkdir: If True, create the path as a directory if it doesn't exist. Defaults to False.

    Returns:
        pathlib.Path: The incremented path, or the original path if `exist_ok` is True and the path exists.
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        root, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        n = 2
        while True:
            incremented_path = Path(f'{root}{sep}{n}{suffix}')
            if not incremented_path.exists():
                break
            n += 1
        path = incremented_path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)

    return path


def setup_logging(name=LOGGING_NAME, verbose=True):
    """Set up logging configuration"""
    level = logging.INFO if verbose else logging.ERROR
    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {name: {'format': '%(asctime)s %(levelname)-8s %(message)s'}},
            'handlers': {name: {'class': 'logging.StreamHandler', 'formatter': name, 'level': level}},
            'loggers': {name: {'level': level, 'handlers': [name], 'propagate': False}},
        }
    )


setup_logging(LOGGING_NAME, verbose=True)
LOGGER = logging.getLogger(LOGGING_NAME)


def load_yaml(file_path: Union[str, Path]) -> Dict:
    """
    Load a YAML file and return its contents as a dictionary.
    Args:
        file_path: The path to the YAML file to be loaded.

    Returns:
        A dictionary representation of the YAML file contents.
        If the file is empty or contains no valid YAML data, an empty dictionary is returned.
    """
    with Path(file_path).open('r') as f:
        s = f.read()
        data = yaml.safe_load(s) or {}
        return data


def print_yaml(yaml_file: Union[str, Path, Dict]):
    """
    Print the contents of a YAML file or a provided YAML dictionary.

    Args:
        yaml_file: The path to the YAML file, or a preloaded dictionary representation of the YAML contents.
    """
    yaml_dict = load_yaml(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    dump = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True)
    LOGGER.info(f"Printing '{yaml_file}'\n\n{dump}")


class IterableNamespace(SimpleNamespace):
    """
    An extended version of SimpleNamespace that supports iteration and some dictionary-like methods.
    """

    def __iter__(self):
        return iter(vars(self).items())

    def __str__(self):
        return '\n'.join(f'{k}={v}' for k, v in vars(self).items())

    def __getattr__(self, attr):
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'. "
            "This may be due to a modified 'default.yaml' file."
        )

    def get(self, key, default=None):
        return getattr(self, key, default)


DEFAULT_CONFIG_DICT = load_yaml(DEFAULT_CONFIG_PATH)
for k, v in DEFAULT_CONFIG_DICT.items():
    if isinstance(v, str) and v.lower() == 'none':
        DEFAULT_CONFIG_DICT[k] = None
DEFAULT_CONFIG_KEYS = DEFAULT_CONFIG_DICT.keys()
DEFAULT_CONFIG = IterableNamespace(**DEFAULT_CONFIG_DICT)
