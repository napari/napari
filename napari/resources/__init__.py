from os.path import abspath, dirname, join
from glob import glob
from . import qt
from typing import Optional, List

resources_dir = abspath(dirname(__file__))
_COMBINED = None


def combine_stylesheets(extra: Optional[List[str]] = []):
    global _COMBINED
    if _COMBINED is None:
        _COMBINED = ''
        for file in sorted(glob(join(resources_dir, 'styles/*.qss'))):
            with open(file, 'r') as f:
                _COMBINED += f.read()
    out = _COMBINED
    if extra:
        for file in extra:
            with open(file, 'r') as f:
                out += f.read()
    return out
