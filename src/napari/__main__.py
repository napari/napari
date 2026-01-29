import os
import shutil
import subprocess
import sys


def show_error_dialog_windows(message: str, title: str = 'Error') -> None:
    import ctypes

    # Display a message box with the error message
    # https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-messageboxw
    ctypes.windll.user32.MessageBoxW(0, message, title, 1)


def show_error_dialog_mac(message: str, title: str = 'Error') -> None:
    # Escape quotes for AppleScript
    msg = message.replace('"', '\\"')
    ttl = title.replace('"', '\\"')
    script = (
        f'display alert "{ttl}" message "{msg}" as critical buttons {{"OK"}}'
    )
    subprocess.run(['osascript', '-e', script], check=False)


def show_error_dialog_linux(message: str, title: str = 'Error') -> None:
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


def message_exception(exc: Exception):
    title = 'Startup Error'
    message = f'An error occurred while starting napari.\n\nexc: {exc}'
    if sys.stdout is None:  # when run without terminal
        # if windows us ctypes to show a message box
        if sys.platform == 'win32':
            show_error_dialog_windows(message, title)
        elif sys.platform == 'darwin':
            show_error_dialog_mac(message, title)
        elif sys.platform == 'linux':
            show_error_dialog_linux(message, title)


if __name__ == '__main__':
    try:
        from napari._main import main
    except Exception as e:
        message_exception(e)
        raise
    else:
        sys.exit(main())
