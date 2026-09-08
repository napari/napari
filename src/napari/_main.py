import os
import shutil
import subprocess
import sys
import tempfile
import traceback


def _show_error_dialog_windows(message: str, title: str = 'Error') -> None:
    import ctypes

    # Display a message box with the error message
    # https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-messageboxw
    ctypes.windll.user32.MessageBoxW(0, message, title, 1)  # type: ignore[attr-defined]


def _show_error_dialog_mac(message: str, title: str = 'Error') -> None:
    # Escape quotes for AppleScript
    msg = message.replace('"', '\\"')
    ttl = title.replace('"', '\\"')
    script = (
        f'display alert "{ttl}" message "{msg}" as critical buttons {{"OK"}}'
    )
    subprocess.run(['osascript', '-e', script], check=False)


def _show_error_dialog_linux(message: str, title: str = 'Error') -> None:
    # Must have a GUI session
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        return

    msg = message.replace('\n', '\\n')
    ttl = title.replace('\n', ' ')

    if shutil.which('zenity'):
        subprocess.run(
            ['zenity', '--error', '--title', ttl, '--text', msg], check=False
        )
    elif shutil.which('kdialog'):
        subprocess.run(
            ['kdialog', '--error', msg, '--title', ttl], check=False
        )
    elif shutil.which('xmessage'):
        subprocess.run(
            ['xmessage', '-center', '-title', ttl, msg], check=False
        )


def show_startup_error_dialog(
    exc: Exception, title: str, message: str
) -> None:
    """Display a native OS error dialog before napari has started.

    Use this function to report exceptions that occur during
    startup. Works gracefully across Windows, macOS, and Linux when napari
    is not launched from the terminal.

    Note: After successful startup, napari's error handling and logging are used.
    """
    if sys.stdout is None:  # a guard when napari is run without a terminal
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', delete=False, prefix='napari-error-'
            ) as temp_file:
                traceback.print_exception(
                    type(exc), exc, exc.__traceback__, file=temp_file
                )
                message += f'\n\nSee {temp_file.name} for more details.'
        except Exception:  # noqa: BLE001
            message += f'\n\n{exc}'

        if sys.platform == 'win32':
            _show_error_dialog_windows(message, title)
        elif sys.platform == 'darwin':
            _show_error_dialog_mac(message, title)
        elif sys.platform == 'linux':
            _show_error_dialog_linux(message, title)


def main() -> None:
    try:
        from napari.__main__ import _build_viewer
        from napari._qt.qt_event_loop import run
    except Exception as e:
        show_startup_error_dialog(
            e, title='Import Error', message='napari failed to import.'
        )
        raise

    try:
        _viewer = _build_viewer()
    except Exception as e:
        show_startup_error_dialog(
            e,
            title='Startup Error',
            message='napari failed to create a viewer.',
        )
        raise

    run(gui_exceptions=True)
    sys.exit(0)


if __name__ == '__main__':
    main()
