import os
from pathlib import Path
from qtpy import API_NAME

if API_NAME == 'PySide2':
    # Set plugin path appropriately if using PySide2. This is a bug fix
    # for when both PyQt5 and Pyside2 are installed
    import PySide2

    os.environ['QT_PLUGIN_PATH'] = str(
        Path(PySide2.__file__).parent / 'Qt' / 'plugins'
    )

from .viewer import Viewer
from ._view import view
from ._version import get_versions


__version__ = get_versions()['version']
del get_versions
