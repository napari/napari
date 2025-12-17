from warnings import warn

from napari.utils.translations import trans

try:
    from qtpy import API_NAME, QtCore
except Exception as e:
    if 'No Qt bindings could be found' in str(e):
        from importlib.metadata import version
        from inspect import cleandoc

        from napari.utils._env_detection import detect_environment

        raise ImportError(
            trans._(
                cleandoc(
                    """
                No Qt bindings could be found for napari=={version}.

                napari requires either PyQt5 (default), PyQt6 or PySide6 to be installed in the environment.

                With pip, you can install either with:
                  $ pip install -U 'napari[all]'  # default choice
                  $ pip install -U 'napari[pyqt5]'
                  $ pip install -U 'napari[pyqt6]'
                  $ pip install -U 'napari[pyside6]'

                With conda, you need to do:
                  $ conda install -c conda-forge pyqt
                  $ conda install -c conda-forge pyside6

                Our heuristics suggest you are using '{tool}' to manage your packages.
                """
                ),
                deferred=True,
                tool=detect_environment().value,
                version=version('napari'),
            )
        ) from e
    raise


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
            '\n\nnapari was tested with QT library `>=5.12.3`.\nThe version installed is {version}. Please report any issues with\nthis specific QT version at https://github.com/Napari/napari/issues.',
            deferred=True,
            version=QtCore.__version__,
        )
    warn(message=warn_message, stacklevel=1)


from napari._qt.qt_event_loop import get_qapp, quit_app, run
from napari._qt.qt_main_window import Window

__all__ = ['Window', 'get_qapp', 'quit_app', 'run']
