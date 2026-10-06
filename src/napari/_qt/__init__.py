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

                napari requires either PyQt6 (default) or PySide6 to be installed in the environment.

                With pip, you can install either with:
                    $ pip install -U 'napari[all]'  # default choice
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


MIN_QT_VERSION = (6, 7)


def _check_qt_version(qt_version: str, api_name: str) -> None:
    """Raise ImportError if the Qt version used by qtpy is not supported."""
    if tuple(int(x) for x in qt_version.split('.')[:2]) >= MIN_QT_VERSION:
        return

    min_version = '.'.join(map(str, MIN_QT_VERSION))
    qt5_hint = (
        (
            'If PyQt6 or PySide6 is also installed, select it with the QT_API '
            'environment variable (for example QT_API=pyqt6) '
            f'or uninstall {api_name}.\n\n'
        )
        if qt_version.startswith('5.')
        else ''
    )
    raise ImportError(
        f'napari requires Qt >= {min_version}, but {api_name} with '
        f'Qt {qt_version} was imported.\n\n'
        f'{qt5_hint}'
        'With pip, you can update with:\n'
        "    $ pip install -U 'napari[pyqt6]'\n"
        "    $ pip install -U 'napari[pyside6]'\n"
        'With conda, you can update with:\n'
        f'    $ conda install -c conda-forge "pyqt6>={min_version}"\n'
        f'    $ conda install -c conda-forge "pyside6>={min_version}"'
    )


_check_qt_version(QtCore.__version__, API_NAME)


from napari._qt.qt_event_loop import get_qapp, quit_app, run  # noqa: E402
from napari._qt.qt_main_window import Window  # noqa: E402

__all__ = ['Window', 'get_qapp', 'quit_app', 'run']
