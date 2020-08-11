try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"

import os
from distutils.version import StrictVersion
from pathlib import Path

try:
    from qtpy import API_NAME
except Exception as e:
    if 'No Qt bindings could be found' in str(e):
        raise type(e)(
            "No Qt bindings could be found.\n\nnapari requires either PyQt5 or"
            " PySide2 to be installed in the environment.\nTo install the "
            'default backend (currently PyQt5), run "pip install napari[all]"'
            '\nYou may also use "pip install napari[pyside2]" for Pyside2, '
            'or "pip install napari[pyqt5]" for PyQt5'
        ) from e
    raise


if API_NAME == 'PySide2':
    # Set plugin path appropriately if using PySide2. This is a bug fix
    # for when both PyQt5 and Pyside2 are installed
    import PySide2

    os.environ['QT_PLUGIN_PATH'] = str(
        Path(PySide2.__file__).parent / 'Qt' / 'plugins'
    )

# When QT is not the specific version, we raise a warning:
from warnings import warn

from qtpy import QtCore

if StrictVersion(QtCore.__version__) < StrictVersion('5.12.3'):
    warn_message = f"""
    napari was tested with QT library `>=5.12.3`.
    The version installed is {QtCore.__version__}. Please report any issues with this
    specific QT version at https://github.com/Napari/napari/issues.
    """
    warn(message=warn_message)

import logging

from vispy import app

# set vispy application to the appropriate qt backend
app.use_app(API_NAME)
del app
# set vispy logger to show warning and errors only
vispy_logger = logging.getLogger('vispy')
vispy_logger.setLevel(logging.WARNING)

# Note that importing _viewer_key_bindings is needed as the Viewer gets
# decorated with keybindings during that process, but it is not directly needed
# by our users and so is deleted below
from . import _viewer_key_bindings  # noqa: F401
from ._qt import gui_qt
from .plugins.io import save_layers
from .utils import _magicgui, sys_info
from .view_layers import (
    view_image,
    view_labels,
    view_path,
    view_points,
    view_shapes,
    view_surface,
    view_vectors,
)
from .viewer import Viewer

# register napari object types with magicgui if it is installed
_magicgui.register_types_with_magicgui()


# this unused import is here to fix a very strange bug.
# there is some mysterious magical goodness in scipy stats that needs
# to be imported early.
# see: https://github.com/napari/napari/issues/925
# see: https://github.com/napari/napari/issues/1347
from scipy import stats  # noqa: F401

del _magicgui
del stats
del _viewer_key_bindings
