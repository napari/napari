"""Napari updates utilities."""

def _get_installed_versions():
    """"""
    # Check the current conda prefix for installed versions
    return []


def _check_pypi_updates():
    """"""
    # https://pypi.org/pypi/napari/json
    pass


def _check_conda_updates():
    pass
    # https://api.anaconda.org/package/conda-forge/napari
    # Filter versions


def check_updates(installer=None):
    """Check for updates."""
    # Ensure they are strings
    try:
        from napari._version import __version__
    except ImportError:
        __version__ = None

    data = {
        "current": __version__,
        "latest": "",
        "installer": "bundle",
    }
    return data
