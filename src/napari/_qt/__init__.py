from warnings import warn

try:
    from qtpy import API_NAME, QtCore
except Exception as e:
    if 'No Qt bindings could be found' in str(e):
        from importlib.metadata import version
        from inspect import cleandoc

        from napari.utils._env_detection import detect_environment

        raise ImportError(
            cleandoc(
                f"""
                No Qt bindings could be found for napari=={version('napari')}.

                napari requires either PyQt5, PyQt6 (default) or PySide6 to be installed in the environment.

                With pip, you can install either with:
                    $ pip install -U 'napari[all]'  # default choice
                    $ pip install -U 'napari[pyqt5]'
                    $ pip install -U 'napari[pyqt6]'
                    $ pip install -U 'napari[pyside6]'

                With conda, you need to do:
                    $ conda install -c conda-forge pyqt6
                    $ conda install -c conda-forge pyside6

                Our heuristics suggest you are using '{detect_environment().value}' to manage your packages.
                """
            )
        ) from e
    raise


# When QT is not the specific version, we raise a warning:
if tuple(int(x) for x in QtCore.__version__.split('.')[:3]) < (5, 12, 3):
    import importlib.metadata

    try:
        dist_info_version = importlib.metadata.version(API_NAME)
        if dist_info_version != QtCore.__version__:
            warn_message = f"\n\nIMPORTANT:\nYou are using QT version {QtCore.__version__}, but version {dist_info_version} was also found in your environment.\nThis usually happens when you 'conda install' something that also depends on PyQt\n*after* you have pip installed napari (such as jupyter notebook).\nYou will likely run into problems and should create a fresh environment.\nIf you want to install conda packages into the same environment as napari,\nplease add conda-forge to your channels: https://conda-forge.org\n"
    except ModuleNotFoundError:
        warn_message = f'\n\nnapari was tested with QT library `>=5.12.3`.\nThe version installed is {QtCore.__version__}. Please report any issues with\nthis specific QT version at https://github.com/Napari/napari/issues.'
    warn(message=warn_message, stacklevel=1)


from napari._qt.qt_event_loop import get_qapp, quit_app, run
from napari._qt.qt_main_window import Window

__all__ = ['Window', 'get_qapp', 'quit_app', 'run']
