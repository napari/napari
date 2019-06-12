import os
from pathlib import Path
from qtpy import API_NAME

if API_NAME == 'PySide2':
    # Set plugin path appropriately if using PySide2
    import PySide2

    os.environ['QT_PLUGIN_PATH'] = str(
        Path(PySide2.__file__).parent / 'Qt' / 'plugins'
    )
elif API_NAME == 'PyQt5':
    # Set plugin path appropriately if using PyQt5
    import PyQt5

    os.environ['QT_PLUGIN_PATH'] = str(
        Path(PyQt5.__file__).parent / 'Qt' / 'plugins'
    )
else:
    raise (ValueError('qtpy backend must either be PyQt5 or PySide2'))

from .viewer import Viewer
from ._view import view
from ._version import get_versions


__version__ = get_versions()['version']
del get_versions
