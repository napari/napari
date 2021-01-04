import sys
from functools import lru_cache
from glob import glob
from importlib.util import module_from_spec, spec_from_file_location
from os import environ, fspath
from os.path import abspath, dirname, expanduser, join
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, Tuple

from qtpy import API, QT_VERSION

from napari import __version__

from .build_icons import build_pyqt_resources


def _try_touch_file(target) -> Optional[Path]:
    """Test to see if we have permissions to create a file at ``target``.

    If the target already exists, it will not be touched.  If it does not
    exist, this function attempts to create it and delete it (i.e. testing
    permissions).  NOTE: all parent directories required to write the file will
    be created, but NOT deleted.

    If successful, the path is returned, if not, return None.

    Parameters
    ----------
    target : str
        Filepath to test

    Returns
    -------
    target : str or None
        Returns the target if it is writeable, returns None if it is not.
    """
    target = Path(target)
    if not target.exists():
        try:
            # create parent directories
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()  # create the file itself
        except Exception:
            return None
        target.unlink()  # delete it
    return target


def import_resources(
    version: str = '', overwrite: bool = False
) -> Tuple[str, Callable]:
    """Build and import our icons as Qt resources.

    This function attempts to write that file to one of three locations
    (in this order):

        1. The directory of *this* file (currently ``napari/resources``)
        2. The user ~/.config/napari directory
        3. A temporary file.

    If a temporary file must be used, resources will need to be rebuilt at each
    launch of napari (which takes ~300ms on a decent computer).

    Parameters
    ----------
    version : str, optional
        Version string, by default the resources will be written to a file that
        encodes the current napari version, as well as Qt backend and version:
        ``_qt_resources_{napari.__version__}_{API}_{QT_VERSION}.py``

    overwrite : bool, optional
        Whether to recompile and overwrite the resources.
        Resources will be rebuilt if any of the following are True:

            - the resources file does not already exist.
            - ``overwrite`` argument is True
            - the ``NAPARI_REBUILD_RESOURCES`` environmental variable is set

    Returns
    -------
    out_path : str
        Path to the python resource file. File is already imported under `napari._qt_resources name`.
        Copy this file to make the SVGs and other resources available in bundled application.

    Raises
    ------
    PermissionError
        If we cannot write to any of the requested locations.
    """
    # the resources filename holds the napari version, Qt API, and QT version
    version = version or f'{__version__}_{API}_{QT_VERSION}'
    filename = f'_qt_resources_{version}.py'

    # see if we can write to the current napari/resources directory
    target_file = _try_touch_file(join(abspath(dirname(__file__)), filename))
    # if not, try to write to ~/.config/napari
    if not target_file:
        target_file = expanduser(join('~', '.config', 'napari', filename))
        target_file = _try_touch_file(target_file)
    # if that still doesn't work, create a temporary directory.
    # all required files (themed SVG icons, res.qrc) will be temporarily built
    # in this directory, and cleaned up after the resources are imported
    if not target_file:
        # not using context manager because we need it for build_pyqt_resources
        # but tempdir will be cleaned automatically when the function ends
        tempdir = TemporaryDirectory()
        target_file = join(tempdir.name, filename)
    # if we can't even write a temporary file, we're probably screwed...
    if not target_file:
        raise PermissionError(
            "Could not write qt_resources to disk. Please report this with a "
            "description of your environment (pip freeze) at "
            "https://github.com/napari/napari/issues"
        )

    # build the res.qrc Qt resources file, and then from that autogenerate
    # the python resources file that needs to be imported.
    # If the file already exists and overwrite is False, it will not be
    # regenerated.
    overwrite = overwrite or bool(environ.get('NAPARI_REBUILD_RESOURCES'))
    respath = build_pyqt_resources(fspath(target_file), overwrite=overwrite)

    # import the python resources file and add to sys.modules
    # https://stackoverflow.com/a/67692/1631624
    spec = spec_from_file_location("napari._qt_resources", respath)
    module = module_from_spec(spec)
    # important to add to sys.modules! otherwise segfault when function ends.
    sys.modules[spec.name] = module
    # for some reason, executing this immediately is causing segfault in tests
    load = lambda: spec.loader.exec_module(module)
    return respath, load


@lru_cache(maxsize=4)
def get_stylesheet(extra: Optional[List[str]] = None) -> str:
    """Combine all qss files into single (cached) style string.

    Note, this string may still have {{ template_variables }} that need to be
    replaced using the :func:`napari.utils.theme.template` function.  (i.e. the
    output of this function serves as the input of ``template()``)

    Parameters
    ----------
    extra : list of str, optional
        Additional paths to QSS files to include in stylesheet, by default None

    Returns
    -------
    css : str
        The combined stylesheet.
    """
    resources_dir = abspath(dirname(__file__))
    stylesheet = ''
    for file in sorted(glob(join(resources_dir, 'styles/*.qss'))):
        with open(file) as f:
            stylesheet += f.read()
    if extra:
        for file in extra:
            with open(file) as f:
                stylesheet += f.read()
    return stylesheet


__all__ = ['build_pyqt_resources', 'get_stylesheet', 'import_resources']
