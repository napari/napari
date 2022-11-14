import os
import sys
from pathlib import Path
from warnings import warn

from napari.utils.translations import trans

try:
    from qtpy import API_NAME, QT_VERSION, QtCore
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

if API_NAME == 'PySide6' and sys.version_info[:2] < (3, 10):
    from packaging import version

    if version.parse(QT_VERSION) > version.parse("6.3.1"):
        raise RuntimeError(
            trans._(
                "Napari is not expected to work with PySide6 >= 6.3.2 on Python < 3.10",
                deferred=True,
            )
        )


# When QT is not the specific version, we raise a warning:
if tuple(int(x) for x in QtCore.__version__.split('.')[:3]) < (5, 12, 3):
    import importlib.metadata

    try:
        dist_info_version = importlib.metadata.version(API_NAME)
        if dist_info_version != QtCore.__version__:
            warn_message = trans._(
                "\n\nIMPORTANT:\nYou are using QT version {version}, but version {dversion} was also found in your environment.\nThis usually happens when you 'conda install' something that also depends on PyQt\n*after* you have pip installed napari (such as jupyter notebook).\nYou will likely run into problems and should create a fresh environment.\nIf you want to install conda packages into the same environment as napari,\nplease add conda-forge to your channels: https://conda-forge.org\n",
                deferred=True,
                version=QtCore.__version__,
                dversion=dist_info_version,
            )
    except ModuleNotFoundError:
        warn_message = trans._(
            "\n\nnapari was tested with QT library `>=5.12.3`.\nThe version installed is {version}. Please report any issues with\nthis specific QT version at https://github.com/Napari/napari/issues.",
            deferred=True,
            version=QtCore.__version__,
        )
    warn(message=warn_message)


from napari._qt.qt_event_loop import get_app, gui_qt, quit_app, run
from napari._qt.qt_main_window import Window
