import os
from pathlib import Path
import PySide2

os.environ['QT_API'] = 'pyside2'
os.environ['QT_PLUGIN_PATH'] = str(
    Path(PySide2.__file__).parent / 'Qt' / 'plugins'
)
from .viewer import Viewer
from ._view import view
from ._version import get_versions


__version__ = get_versions()['version']
del get_versions
