import re
from pathlib import Path

from setuptools import setup, find_packages

FILE = Path(__file__).resolve()
PARENT = FILE.parent


def parse_requirements(file_path):
    return [
        line.strip()
        for line in file_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]


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
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    package_data={'quannet': ['config/*.yaml']},
    install_requires=REQUIREMENTS,
    entry_points={'console_scripts': ['quannet = quannet.config:entrypoint']},
)
