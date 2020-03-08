from os import environ
from os.path import abspath, dirname, join
from glob import glob
from typing import List, Optional
from .build_icons import build_pyqt_resources

overwrite = bool(environ.get('NAPARI_REBUILD_RESOURCES'))
build_pyqt_resources(overwrite=overwrite)
from . import _qt

_STYLESHEET = None  # cached stylesheet string, built once per session


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

    global _STYLESHEET
    if _STYLESHEET is None:
        _STYLESHEET = ''
        for file in sorted(glob(join(resources_dir, 'styles/*.qss'))):
            with open(file, 'r') as f:
                _STYLESHEET += f.read()
    out = _STYLESHEET
    if extra:
        for file in extra:
            with open(file, 'r') as f:
                out += f.read()
    return out


__all__ = ['build_pyqt_resources', 'get_stylesheet']
