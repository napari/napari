from os import environ, fspath
from os.path import abspath, dirname, join, expanduser
from glob import glob
from typing import List, Optional
from .build_icons import build_pyqt_resources
from functools import lru_cache
from importlib.util import spec_from_file_location, module_from_spec
from tempfile import NamedTemporaryFile
import sys
from pathlib import Path


def _try_touch_file(target_file):
    target_file = Path(target_file)
    if not target_file.exists():
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.touch()
        except Exception:
            return None
        target_file.unlink()
    return target_file


def import_resources(version='', overwrite=False):
    filename = f'_qt_resources{"_" if version else ""}{version}.py'
    target_file = _try_touch_file(join(abspath(dirname(__file__)), filename))
    tempfile = None
    if not target_file:
        target_file = expanduser(join('~', '.config', 'napari', filename))
        target_file = _try_touch_file(target_file)
    if not target_file:
        tempfile = NamedTemporaryFile(suffix='.py')
        target_file = _try_touch_file(tempfile.name)
    if not target_file:
        raise PermissionError(
            "Could not write qt_resource to disk. Please report this with a "
            "description of your environment at "
            "https://github.com/napari/napari/issues"
        )
    overwrite = (
        overwrite
        or bool(environ.get('NAPARI_REBUILD_RESOURCES'))
        or tempfile is not None
    )
    respath = build_pyqt_resources(fspath(target_file), overwrite=overwrite)
    spec = spec_from_file_location("napari._qt_resources", respath)
    sys.modules[spec.name] = module_from_spec(spec)
    spec.loader.exec_module(sys.modules[spec.name])
    if tempfile:
        tempfile.close()


@lru_cache
def get_stylesheet(extra: Optional[List[str]] = None) -> str:
    """Combine all qss files into single (cached) style string.
    
    Note, this string may still have {{ template_variables }} that need to be
    replaced using the :func:`napari.utils.theme.template` function.  (i.e. the
    output of this function serves as the input of ``template()``)

    Parameters
    ----------
    extra : list of str, optional
        Additional qss files to include in stylesheet, by default None
    
    Returns
    -------
    css : str
        the combined stylesheet.
    """
    resources_dir = abspath(dirname(__file__))
    stylesheet = ''
    for file in sorted(glob(join(resources_dir, 'styles/*.qss'))):
        with open(file, 'r') as f:
            stylesheet += f.read()
    if extra:
        for file in extra:
            with open(file, 'r') as f:
                stylesheet += f.read()
    return stylesheet


__all__ = ['build_pyqt_resources', 'get_stylesheet']
