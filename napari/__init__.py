import os
from pathlib import Path

if 'NAPRAI_QT' not in os.environ:
    os.environ['NAPRAI_QT'] = 'pyside2'

if os.environ['NAPRAI_QT'] == 'pyside2':
    import PySide2

    os.environ['QT_API'] = 'pyside2'
    os.environ['QT_PLUGIN_PATH'] = str(
        Path(PySide2.__file__).parent / 'Qt' / 'plugins'
    )
elif os.environ['NAPRAI_QT'] == 'pyqt5':
    import PyQt5

    os.environ['QT_API'] = 'pyqt5'
    os.environ['QT_PLUGIN_PATH'] = str(
        Path(PyQt5.__file__).parent / 'Qt' / 'plugins'
    )
else:
    raise (ValueError('NAPRAI_QT environment variable not recognized'))

from .viewer import Viewer
from ._view import view
from ._version import get_versions


__version__ = get_versions()['version']
del get_versions
