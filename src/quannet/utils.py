import logging
import logging.config
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Union

import numpy as np
import torch
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
TEST_STRUCTURES = ROOT / 'structures'
DEFAULT_CONFIG_PATH = ROOT / 'config/default.yaml'
LOGGING_NAME = 'quannet'


def setup_logging(name=LOGGING_NAME, verbose=True):
    """Set up logging configuration"""
    level = logging.INFO if verbose else logging.ERROR
    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {name: {'format': '%(asctime)s %(levelname)-8s %(name)-15s %(message)s'}},
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


def search_file(file: str, dir: str = 'config') -> str:
    """
    Search the file (if necessary) and return its path.

    Args:
        file: The name or path of the file.
        dir: The name of directory for recursive search

    Returns:
        The file path if it exists.
    """
    file_path = Path(file).resolve()

    if file_path.exists():
        return str(file_path)

    # Search for the file recursively within the 'config' directory under ROOT
    matching_files = list(ROOT.joinpath(dir).rglob(file))

    if not matching_files:
        raise FileNotFoundError(f"'{file}' does not exist")
    elif len(matching_files) > 1:
        raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {matching_files}")

    return str(matching_files[0])


def set_seed_and_determinism(seed: int = 42, deterministic: bool = False) -> None:
    """
    Sets seed for various random number generators and (optionally) enforces deterministic behavior in PyTorch.

    Args:
        seed: Seed for random number generators.
        deterministic: If True, enforce deterministic behavior in PyTorch.
    """

    # Set seed for Python, numpy, and torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)

    else:
        torch.use_deterministic_algorithms(False)

    torch.backends.cudnn.deterministic = deterministic
