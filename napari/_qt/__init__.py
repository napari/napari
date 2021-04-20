import os
from distutils.version import StrictVersion
from pathlib import Path
from warnings import warn

from ..utils.translations import trans

try:
    from qtpy import API_NAME, QtCore
except Exception as e:
    if 'No Qt bindings could be found' in str(e):
        raise type(e)(
            trans._(
                "No Qt bindings could be found.\n\nnapari requires either PyQt5 or PySide2 to be installed in the environment.\nTo install the default backend (currently PyQt5), run \"pip install napari[all]\" \nYou may also use \"pip install napari[pyside2]\"for Pyside2, or \"pip install napari[pyqt5]\" for PyQt5",
                deferred=True,
            )
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
if StrictVersion(QtCore.__version__) < StrictVersion('5.12.3'):
    warn_message = trans._(
        "napari was tested with QT library `>=5.12.3`.\nThe version installed is {version}. Please report any issues with this specific QT version at https://github.com/Napari/napari/issues.",
        deferred=True,
        version=QtCore.__version__,
    )
    warn(message=warn_message)


from .qt_event_loop import get_app, gui_qt, quit_app, run
from .qt_main_window import Window
from .widgets.qt_range_slider import QHRangeSlider, QVRangeSlider
