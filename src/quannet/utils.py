import logging
import logging.config
from types import SimpleNamespace
from typing import Dict, List, Union
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
import torch

ArrayLike = Union[list, pd.Series, np.ndarray, torch.Tensor]

FILE = Path(__file__).resolve()
PACKAGE_ROOT = FILE.parents[0]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / 'config/default.yaml'
LOGGING_NAME = 'quannet'


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
            'models_dir': str(root / 'models'),
            'test_dataset_dir': str(root / 'datasets' / 'quannet_test'),
            'artefacts_dir_name': 'artefacts',
            'logs_dir_name': 'logs',
            'structures_dir_name': 'structures',
            'container_workdir': '/app',
            'container_datasets_dir': '/app/datasets',
            'container_runs_dir': '/app/runs',
        }

        super().__init__(copy.deepcopy(self.defaults))


SETTINGS = SettingsManager()
DATASETS_DIR = Path(SETTINGS['datasets_dir'])
MODELS_DIR = Path(SETTINGS['models_dir'])
RUNS_DIR = Path(SETTINGS['runs_dir'])
TEST_STRUCTURES = Path(SETTINGS['test_dataset_dir'], SETTINGS['structures_dir_name'])
QUANNET_MODEL = MODELS_DIR / 'quannet.pt'


class IterableNamespace(SimpleNamespace):
    """
    An extended version of SimpleNamespace that supports iteration and some dictionary-like methods.
    """

    def __iter__(self):
        """Allows iteration over attribute key-value pairs."""
        return iter(vars(self).items())

    def __str__(self):
        """Provides a string representation of all attributes."""
        return '\n'.join(f'{k}={v}' for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Raises an AttributeError if an attribute is missing."""
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'. "
            "This may be due to a modified 'default.yaml' file."
        )

    def get(self, key, default=None):
        """Fetch an attribute value with a default if the attribute doesn't exist."""
        return getattr(self, key, default)


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


DEFAULT_CONFIG_DICT = load_yaml(DEFAULT_CONFIG_PATH)
for k, v in DEFAULT_CONFIG_DICT.items():
    if isinstance(v, str) and v.lower() == 'none':
        DEFAULT_CONFIG_DICT[k] = None
DEFAULT_CONFIG_KEYS = DEFAULT_CONFIG_DICT.keys()
DEFAULT_CONFIG = IterableNamespace(**DEFAULT_CONFIG_DICT)


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

    # Search for the file recursively within the dir directory under PACKAGE_ROOT
    if isinstance(dir, str):
        matching_files = list(PACKAGE_ROOT.joinpath(dir).rglob(file))
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


class InsufficientDataError(Exception):
    def __init__(self, message='Insufficient data based on the given threshold'):
        self.message = message
        super().__init__(self.message)


class DuplicatedDataError(Exception):
    def __init__(self, message='Duplicated .pdb files found'):
        self.message = message
        super().__init__(self.message)


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
            'formatters': {name: {'format': '%(message)s'}},
            'handlers': {name: {'class': 'logging.StreamHandler', 'formatter': name, 'level': level}},
            'loggers': {name: {'level': level, 'handlers': [name], 'propagate': False}},
        }
    )


setup_logging(LOGGING_NAME, verbose=True)
LOGGER = logging.getLogger(LOGGING_NAME)


def extend_docstring_from(source_func):
    """
    Decorator to extend the docstring of the target function using the docstring of the source function.

    Args:
        source_func: The function whose docstring will be appended to the target function's docstring.

    Returns:
        function: The decorated function with the extended docstring.
    """

    def decorator(target_func):
        if target_func.__doc__:
            target_func.__doc__ += '\n' + (source_func.__doc__ or '')
        else:
            target_func.__doc__ = source_func.__doc__
        return target_func

    return decorator


def common_deepest_directory(paths: List[Union[str, Path]]) -> Path:
    """
    Determine the deepest common directory of a list of pathlib.Path objects.

    This function compares each part of the provided paths and identifies the
    deepest directory common to all paths.

    Args:
        paths: A list of pathlib.Path objects.

    Returns:
        Path: The common directory path.
    """

    # Convert paths to parts
    parts = [Path(p).parts for p in paths]

    # Identify common parts
    common_parts = []
    for part_group in zip(*parts):
        first_part = part_group[0]
        if all(part == first_part for part in part_group):
            common_parts.append(first_part)
        else:
            break

    # Construct common path from parts
    return Path(*common_parts)
