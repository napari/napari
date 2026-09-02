from warnings import warn

try:
    from qtpy import API_NAME, QtCore
except Exception as e:
    if 'No Qt bindings could be found' in str(e):
        import os
        import traceback
        from importlib import import_module
        from importlib.metadata import version
        from inspect import cleandoc

        from napari.utils._env_detection import (
            detect_environment,
            detect_installed_qt_bindings,
        )

        qt_api_enforce = os.environ.get('QT_API', '')

        if installed_bindings := detect_installed_qt_bindings():
            available_qt_bindins = ', '.join(
                f'{name}={version}'
                for name, version in installed_bindings.items()
            )
            if qt_api_enforce and qt_api_enforce not in installed_bindings:
                if len(installed_bindings) > 1:
                    qt_text = f'but {available_qt_bindins} are installed in your environment'
                else:
                    qt_text = f'but {available_qt_bindins} is installed in your environment'

                raise ImportError(
                    cleandoc(
                        f"""
                    The Qt bindings enforced by QT_API environment variable are not installed.
                    You have QT_API={qt_api_enforce} installed, {qt_text}.
                    """
                    )
                ) from e

            name_to_module = {
                'pyqt5': 'PyQt5',
                'pyqt6': 'PyQt6',
                'pyside6': 'PySide6',
            }

            fail_inf = {}

            for binding in name_to_module:
                if binding in installed_bindings:
                    try:
                        import_module(f'{name_to_module[binding]}.QtWidgets')
                    except:  # noqa: E722
                        fail_inf[binding] = traceback.format_exc()
                    else:
                        fail_inf[binding] = 'No error'

            error_summary = '\n\n'.join(
                f'{binding}: {exc}' for binding, exc in fail_inf.items()
            )

            raise ImportError(
                cleandoc(f"""
            Failed to import Qt bindings. We found following Qt bindings installed: {available_qt_bindins}.
            We have tried to import existing bindings and here are the errors:
            {error_summary}
            """)
            ) from e

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
