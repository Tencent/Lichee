# -*- coding: utf-8 -*-
"""
setup script
"""

import os
import re
import sys

from setuptools import find_packages
from setuptools import setup

if sys.version_info < (3, 6):
    raise RuntimeError("lichee requires Python 3.6+")


def read_version():
    """
    read version from __init__.py
    :return: version
    """
    regexp = re.compile(r"^__version__\W*=\W*'([\d.abrc]+)'")
    init_py = os.path.join(os.path.dirname(__file__), "lichee", "__init__.py")
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        raise RuntimeError("Cannot find version in {}".format(init_py))


setup(
    name="lichee",
    version=read_version(),
    url="",
    project_urls={
        "Documentation": "",
        "Code": "",
        "Issue tracker": "",
    },
    maintainer="",
    description="",
    python_requires=">=3.6",
    install_requires=[
        "PyYAML>=3.10",
        "torch",
        "torchvision",
        "numpy",
        "yacs",
        "scipy",
        "scikit-learn",
        "protobuf",
        "cgroup_parser"
    ],
    extras_require={
        'dev': ['pytest', 'pyyaml']
    },
    package_data={'': ['*.yaml', '*.so', '*.txt', '*.pyx', '*.pxd']},
    packages=find_packages(include=("lichee", "lichee.*")),
)
