"""Conda utilities."""

import re
import sys
from pathlib import Path


def is_conda_package(package_name: str) -> bool:
    """Determines if a package was installed through conda.

    Parameters
    ----------
    package_name : str
        The name of the package.

    Returns
    -------
    bool: True if a conda package, False if not

    Notes
    -----
    Installed conda packages within a conda installation and environment can
    be identified as `<package-name>-<version>-<build-string>.json` files
    saved within a `conda-meta` folder within the given environment of
    interest.
    """
    conda_meta_dir = Path(sys.prefix) / 'conda-meta'
    return any(
        re.match(rf"{package_name}-[^-]+-[^-]+.json", p.name)
        for p in conda_meta_dir.glob(f"{package_name}-*-*.json")
    )
