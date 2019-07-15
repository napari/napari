import os
from distutils.version import StrictVersion
from pathlib import Path
from qtpy import API_NAME

if API_NAME == 'PySide2':
    # Set plugin path appropriately if using PySide2. This is a bug fix
    # for when both PyQt5 and Pyside2 are installed
    import PySide2

    os.environ['QT_PLUGIN_PATH'] = str(
        Path(PySide2.__file__).parent / 'Qt' / 'plugins'
    )

from qtpy import QtCore

if StrictVersion(QtCore.__version__) < StrictVersion('5.12.3'):
    raise ValueError(
        'QT library must be `>=5.12.3`, got ' + QtCore.__version__
    )

from .viewer import Viewer
from .view_function import view
from ._qt import gui_qt
from ._version import get_versions


__version__ = get_versions()['version']
del get_versions
