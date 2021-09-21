"""
Launching logic for bundle packaging
"""
import os.path
import sys
from pathlib import Path
from typing import Optional

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # noqa


def running_as_bundled_app() -> bool:
    """Infer whether we are running as a briefcase bundle"""
    # https://github.com/beeware/briefcase/issues/412
    # https://github.com/beeware/briefcase/pull/425
    # note that a module may not have a __package__ attribute
    # From 0.4.12 we add a sentinel file next to the bundled sys.executable
    if (Path(sys.executable).parent / ".napari_is_bundled").exists():
        return True
    try:
        app_module = sys.modules['__main__'].__package__
    except AttributeError:
        return False
    try:
        metadata = importlib_metadata.metadata(app_module)
    except importlib_metadata.PackageNotFoundError:
        return False

    return 'Briefcase-Version' in metadata


def bundle_bin_dir() -> Optional[str]:
    """Return path to briefcase app_packages/bin if it exists."""
    bin = os.path.join(os.path.dirname(sys.exec_prefix), 'app_packages', 'bin')
    if os.path.isdir(bin):
        return bin


def set_conda_config():
    if sys.platform == "darwin":
        launcher = Path(sys.executable)
        contents = launcher.parent.parent
        conda = contents / "Resources" / "conda" / "envs" / "napari"
        site_packages = (
            conda
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )

    sys.path.insert(2, str(site_packages))
    os.environ["QT_PLUGIN_PATH"] = str(conda / "plugins")
