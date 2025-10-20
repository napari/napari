import os
from enum import auto
from pathlib import Path

from napari.utils.compat import StrEnum


class Environment(StrEnum):
    pip = auto()
    uv = auto()
    conda = auto()
    pixi = auto()


def get_venv_path() -> Path | None:
    """Get the path to the current virtual environment, if any.

    Returns
    -------
    Path | None
        The path to the current virtual environment, or None if not in a venv.
    """
    venv = os.environ.get('VIRTUAL_ENV')
    return Path(venv) if venv else None


def get_conda_path() -> Path | None:
    """Get the path to the current conda environment, if any.

    Returns
    -------
    Path | None
        The path to the current conda environment, or None if not in conda.
    """
    conda = os.environ.get('CONDA_PREFIX')
    return Path(conda) if conda else None


def check_if_uv_env() -> bool:
    """Check if we are in a uv environment.

    Returns
    -------
    bool
        True if we are in a uv virtual environment, False otherwise.
    """
    venv_path = get_venv_path()
    if venv_path is None:
        raise ValueError('Not in a virtual environment')
    if not (venv_path / 'pyvenv.cfg').exists():
        return False
    return 'uv =' in (venv_path / 'pyvenv.cfg').read_text()


def check_if_pixi_env() -> bool:
    """Check if we are in a pixi virtual environment.

    Returns
    -------
    bool
        True if we are in a pixi virtual environment, False otherwise.
    """
    env_path = get_conda_path()
    if env_path is None:
        raise ValueError('Not in a conda environment')
    if not (env_path / 'conda-meta').exists():
        raise ValueError('Not in a conda environment')
    return (env_path / 'conda-meta' / 'pixi_env_prefix').exists()


def detect_environment() -> Environment:
    """Detect the current isolated environment."""
    if get_venv_path() is not None:
        if check_if_uv_env():
            return Environment.uv
        return Environment.pip
    if get_conda_path() is not None:
        if check_if_pixi_env():
            return Environment.pixi
        return Environment.conda
    return Environment.pip
