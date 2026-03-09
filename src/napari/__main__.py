import hashlib
import os
import sys
from pathlib import Path

from appdirs import user_cache_dir


def _user_cache_dir():
    prefix_path = os.path.realpath(sys.prefix)
    environment_marker = f'{os.path.basename(prefix_path)}_{hashlib.sha1(prefix_path.encode()).hexdigest()}'
    return user_cache_dir('napari', False, environment_marker)


def _get_or_create_napari_symlink():
    """Get or create a persistent symlink for macOS menu naming."""
    cache_dir = _user_cache_dir()
    napari_link = os.path.join(cache_dir, 'napari_exec', 'napari')

    if os.path.exists(napari_link):
        if os.path.samefile(napari_link, sys.executable):
            return napari_link  # Already launched via correct symlink
        os.remove(napari_link)  # Remove stale symlink

    os.makedirs(os.path.dirname(napari_link), exist_ok=True)
    if not os.path.exists(napari_link):
        os.symlink(sys.executable, napari_link)

    return napari_link


def _maybe_rerun_with_macos_fixes():
    """
    Apply some fixes needed in macOS, which might involve
    running this script again using a different sys.executable.

    1) Quick fix for Big Sur Python 3.9 and Qt 5.
       No relaunch needed.
    2) Using `pythonw` instead of `python`.
       This can be used to ensure we're using a framework
       build of Python on macOS, which fixes frozen menubar issues
       in some macOS versions.
    3) Make sure the menu bar uses 'napari' as the display name.
       This requires relaunching the app from a symlink to the
       desired python executable, conveniently named 'napari'.
    """
    # This import mus be here to raise exception about PySide6 problem

    if (
        sys.platform != 'darwin'
        or 'pdb' in sys.modules
        or 'pydevd' in sys.modules
    ):
        return

    if '_NAPARI_RERUN_WITH_FIXES' in os.environ:
        # This function already ran, do not recurse!
        # We also restore sys.executable to its initial value,
        # if we used a symlink
        if exe := os.environ.pop('_NAPARI_SYMLINKED_EXECUTABLE', ''):
            sys.executable = exe
        return

    # In principle, we will relaunch to the same python we were using
    executable = sys.executable

    # Create the env copy now because the following changes
    # should not persist in the current process in case
    # we do not run the subprocess!
    env = os.environ.copy()

    # 3) Make sure the app name in the menu bar is 'napari', not 'python'
    NEEDS_SYMLINK_ = (
        # When napari is launched from the conda bundle shortcut
        # it already has the right 'napari' name in the app title
        # and __CFBundleIdentifier is set to 'com.napari._(<version>)'
        'napari' not in os.environ.get('__CFBUNDLEIDENTIFIER', '')
        # with a sys.executable named napari,
        # macOS should have picked the right name already
        or os.path.basename(executable) != 'napari'
    )
    if NEEDS_SYMLINK_:
        env['_NAPARI_SYMLINKED_EXECUTABLE'] = executable
        executable = _get_or_create_napari_symlink()

    # if at this point 'executable' is different from 'sys.executable', we
    # need to launch the subprocess to apply the fixes
    # it happens when use symlink
    if sys.executable != executable:
        env['_NAPARI_RERUN_WITH_FIXES'] = '1'
        if Path(sys.argv[0]).name == 'napari':
            # launched through entry point, we do that again to avoid
            # issues with working directory getting into sys.path (#5007)
            cmd = [executable, sys.argv[0]]
        else:  # we assume it must have been launched via '-m' syntax
            cmd = [executable, '-m', 'napari']

        # this fixes issues running from a venv/virtualenv based virtual
        # environment with certain python distributions (e.g. pyenv, asdf)
        env['PYTHONEXECUTABLE'] = sys.executable

        # Append original command line arguments.
        if len(sys.argv) > 1:
            cmd.extend(sys.argv[1:])
        os.execve(executable, cmd, env)


def main():

    _maybe_rerun_with_macos_fixes()

    from napari._main import main as main_

    return main_()


if __name__ == '__main__':
    sys.exit(main())
