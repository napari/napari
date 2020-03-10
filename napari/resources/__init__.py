from os import environ
from os.path import abspath, dirname, join
from glob import glob
from typing import List, Optional
from .build_icons import build_pyqt_resources
from functools import lru_cache

overwrite = bool(environ.get('NAPARI_REBUILD_RESOURCES'))
build_pyqt_resources(overwrite=overwrite)
from . import _qt


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
