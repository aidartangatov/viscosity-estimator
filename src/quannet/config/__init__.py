import re
import ast
import sys
from types import SimpleNamespace
from typing import Dict, List, Union
from difflib import get_close_matches
from pathlib import Path

from quannet.utils import (
    LOGGER,
    QUANNET_MODEL,
    DEFAULT_CONFIG,
    TEST_STRUCTURES,
    DEFAULT_CONFIG_DICT,
    DEFAULT_CONFIG_PATH,
    IterableNamespace,
    load_yaml,
    print_yaml,
    search_file,
)

MODES = 'train', 'val', 'predict'
CLI_HELP_MESSAGE = ''


def convert_string_to_type(value):
    """
    Convert a string to its most probable type, such as int, float, bool, list, or None.

    Args:
        value (str): Input string to be converted.

    Returns:
        The converted value or original string if no conversion was made.
    """
    lowered_value = value.lower()

    if lowered_value == 'none':
        return None
    elif lowered_value == 'true':
        return True
    elif lowered_value == 'false':
        return False

    # Try converting to int or float
    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    # Try evaluating as a list or tuple
    try:
        evaluated_value = ast.literal_eval(value)
        if isinstance(evaluated_value, (list, tuple)):
            return evaluated_value
    except (ValueError, SyntaxError):
        pass

    return value


def extract_key_value(pair):
    """
    Extract a key and value from a 'key=value' formatted string.

    Args:
        pair (str): A string containing a key-value pair, e.g., "key=value".

    Returns:
        tuple: A tuple containing the extracted key and value.

    Raises:
        ValueError: If the input string does not contain an equals sign or if the value is missing.
    """

    # Ensure the pair contains an equals sign.
    if '=' not in pair:
        raise ValueError(f"Input '{pair}' is not a valid key=value pair.")

    # Normalize by removing spaces around equals sign.
    normalized_pair = re.sub(r'\s*=\s*', '=', pair)

    # Split on the first '=' sign.
    key, value = normalized_pair.split('=', 1)

    # Ensure a value is present.
    if not value:
        raise ValueError(f"Missing value for key '{key}'.")

    return key, convert_string_to_type(value)


def assert_matching_keys(base: Dict, custom: Dict, e=None):
    """
    Checks for any mismatched keys between a custom configuration dict and a base configuration dict.
    If any mismatched keys are found, the function raises a SyntaxError with suggested keys from the base dict.

    Args:
        custom: dictionary of custom configuration options
        base: dictionary of base configuration options
        e: caught error
    """
    base_keys = set(base.keys())
    custom_keys = set(custom.keys())
    mismatched = custom_keys - base_keys

    if mismatched:
        error_msgs = []
        for x in mismatched:
            matches = get_close_matches(x, base_keys)
            match_string = ', '.join(f'{k}={base[k]}' if base.get(k) is not None else k for k in matches)
            match_str = f" Similar arguments are i.e. {match_string}." if matches else ""
            error_msgs.append(f"'{x}' is not a valid argument.{match_str}")

        raise SyntaxError("\n".join(error_msgs) + CLI_HELP_MESSAGE) from e


def config2dict(config: Union[str, Path, Dict, SimpleNamespace]) -> Dict:
    """
    Convert a configuration to a dictionary
    """
    if isinstance(config, (str, Path)):
        config = load_yaml(config)
    elif isinstance(config, SimpleNamespace):
        config = vars(config)
    return config


def get_config(
    config: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CONFIG_DICT, overrides: Union[str, Dict] = None
) -> IterableNamespace:
    """
    Load and merge configuration data from a file, dictionary, or a SimpleNamespace object.

    Args:
        config: Configuration data.
        overrides: Overrides in the form of a file name or a dictionary. Default is None.

    Returns:
        Training arguments namespace.
    """
    config = config2dict(config)

    if overrides:
        overrides = config2dict(overrides)
        assert_matching_keys(config, overrides)
        config = {**config, **overrides}

    return IterableNamespace(**config)


def merge_around_equals(args: List[str]) -> List[str]:
    """
    Merges arguments around isolated '=' args in a list of strings.

    The function considers the following cases:
    1. When the first argument ends with '=' and the second one doesn't start with it.
    2. When the first argument doesn't end with '=' but the second one starts with it.
    3. When an argument is just an equal sign and is surrounded by two other arguments.

    Args:
        args: A list of strings where each element is an argument.

    Returns:
        A list of strings where the arguments around isolated '=' are merged.
    """
    new_args = []
    i = 0
    while i < len(args):
        arg = args[i]
        if i < len(args) - 1:
            next_arg = args[i + 1]
            if arg == '=':  # merge ['arg', '=', 'val']
                new_args[-1] += f'={next_arg}'
                i += 2
                continue
            elif arg.endswith('=') and '=' not in next_arg:  # merge ['arg=', 'val']
                new_args.append(f'{arg}{next_arg}')
                i += 2
                continue
            elif next_arg.startswith('='):  # merge ['arg', '=val']
                new_args[-1] += next_arg
                i += 2
                continue
        new_args.append(arg)
        i += 1
    return new_args


def entrypoint(debug=''):
    args = (debug.split(' ') if debug else sys.argv)[1:]
    if not args:
        LOGGER.info(CLI_HELP_MESSAGE)
        return

    # Define special commands and their associated actions.
    special_commands = {'help': lambda: LOGGER.info(CLI_HELP_MESSAGE), 'config': lambda: print_yaml(DEFAULT_CONFIG_PATH)}
    full_args_dict = {**DEFAULT_CONFIG_DICT, **{k: None for k in MODES}, **special_commands}

    # Add shortcuts for special commands (e.g., 'h' for 'help' and 'helps' for 'help').
    abbreviations = {k[0]: v for k, v in special_commands.items()}
    pluralized = {k[:-1]: v for k, v in special_commands.items() if len(k) > 1 and k.endswith('s')}

    # Include prefixed versions for command-line conventions (e.g., '-help' and '--help').
    prefixed = {
        **{f'-{k}': v for k, v in special_commands.items()},
        **{f'--{k}': v for k, v in special_commands.items()},
    }

    special_commands = {**special_commands, **abbreviations, **pluralized, **prefixed}

    overrides = {}  # overrides of default parameters
    for arg in merge_around_equals(args):  # merge spaces around '=' sign
        if arg.startswith('--'):
            LOGGER.warning(f"'{arg}' does not require leading dashes '--', updating to '{arg[2:]}'.")
            arg = arg[2:]
        if arg.endswith(','):
            LOGGER.warning(f"'{arg}' does not require trailing comma ',', updating to '{arg[:-1]}'.")
            arg = arg[:-1]
        if '=' in arg:
            try:
                key, value = extract_key_value(arg)
                if key == 'config':  # custom.yaml passed
                    LOGGER.info(f'Overriding {DEFAULT_CONFIG_PATH} with {value}')
                    overrides = {k: v for k, v in load_yaml(search_file(value)).items() if k != 'config'}
                else:
                    overrides[key] = value
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                assert_matching_keys(full_args_dict, {arg: ''}, e)

        elif arg in MODES:
            overrides['mode'] = arg
        elif arg.lower() in special_commands:
            special_commands[arg.lower()]()
            return
        elif arg in DEFAULT_CONFIG_DICT and isinstance(DEFAULT_CONFIG_DICT[arg], bool):
            overrides[arg] = True  # automatically set True for default bool args
        elif arg in DEFAULT_CONFIG_DICT:
            raise SyntaxError(
                f"'{arg}' is a valid argument but is missing an '=' sign "
                f"to set its value, i.e. try '{arg}={DEFAULT_CONFIG_DICT[arg]}'\n{CLI_HELP_MESSAGE}"
            )
        else:
            assert_matching_keys(full_args_dict, {arg: ''})

    # Check keys
    assert_matching_keys(full_args_dict, overrides)

    # Mode
    mode = overrides.get('mode')
    if mode is None:
        mode = DEFAULT_CONFIG.mode or 'predict'
        LOGGER.warning(f"'mode' is missing. Valid modes are {MODES}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {MODES}.\n{CLI_HELP_MESSAGE}")

    # Mode and Model
    if mode == 'predict':
        if 'structures' not in overrides:
            overrides['structures'] = DEFAULT_CONFIG.structures or TEST_STRUCTURES
            LOGGER.warning(f"'structures' is missing. Using default 'structures={overrides['structures']}'.")
        if 'model' not in overrides:
            overrides['model'] = QUANNET_MODEL
            LOGGER.warning(f"'model' is missing. Using default 'model={overrides['model']}'.")
    elif mode in ('train', 'val') and 'dataset' not in overrides:
        LOGGER.exception(f"'target' is missing.\n{CLI_HELP_MESSAGE}")
        return
    from quannet import QuanNet

    model = QuanNet(overrides['model'])

    getattr(model, mode)(**overrides)


if __name__ == '__main__':
    entrypoint()
