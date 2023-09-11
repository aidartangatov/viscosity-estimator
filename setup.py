import re
from pathlib import Path

import pkg_resources as pkg
from setuptools import setup, find_packages

FILE = Path(__file__).resolve()
PARENT = FILE.parent


def parse_requirements(file_path):
    return [f'{x.name}{x.specifier}' for x in pkg.parse_requirements((file_path).read_text())]


def get_version():
    file = PARENT / 'src/quannet/__init__.py'
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]


REQUIREMENTS = parse_requirements(PARENT / 'requirements/inputs.txt') + parse_requirements(
    PARENT / 'requirements/model.txt'
)


setup(
    name='quannet',
    version=get_version(),
    python_requires='>=3.8',
    description='Estimation of Antibody Viscosity.',
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    entry_points={'console_scripts': ['quannet = quannet.config:entrypoint']},
)
