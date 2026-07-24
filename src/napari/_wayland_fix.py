"""Wayland OpenGL startup workaround.

Must run before ``napari._vispy`` (and thus PyOpenGL) is imported, so the
function lives in its own module called from ``napari/__init__.py`` rather
than from ``napari._qt.qt_event_loop``. PyOpenGL caches its platform at
``OpenGL.platform`` import time (see ``OpenGL/platform/__init__.py``); once
the EGL backend is loaded, later writes to ``PYOPENGL_PLATFORM`` are ignored.
"""

import glob
import os
import sys


def _nvidia_driver_loaded() -> bool:
    """Return True if the proprietary Nvidia kernel module is loaded.

    Uses the existence of ``/proc/driver/nvidia/version`` as a proxy.
    """
    return os.path.exists('/proc/driver/nvidia/version')


def _qt_plugins_path() -> str | None:
    """Return Qt's plugin directory, or None if it cannot be determined.

    Goes through Qt's own plugin search path so it tracks whichever binding
    qtpy selected.
    """
    try:
        from qtpy.QtCore import QLibraryInfo

        try:
            return QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
        except AttributeError:  # PyQt5/PySide2 spelling
            return QLibraryInfo.location(QLibraryInfo.PluginsPath)  # type: ignore[attr-defined]
    except (ImportError, AttributeError, OSError):
        return None


def _native_wayland_plugin_available() -> bool:
    """Return True if Qt ships a native Wayland platform plugin.

    Without it, Qt silently falls back to XCB (X11 via XWayland) while PyOpenGL
    still defaults to its EGL backend, and the EGL/GLX context mismatch crashes
    napari on startup. Any failure is treated as "available" so the workaround
    is never forced on an unknown setup.
    """
    plugins = _qt_plugins_path()
    if plugins is None:
        return True
    return bool(glob.glob(os.path.join(plugins, 'platforms', '*wayland*')))


def _qt_from_conda() -> bool:
    """Return True if the active Qt binding uses Qt installed by conda.

    Conda's Qt packages keep their plugins under the environment prefix
    (outside ``site-packages``), while pip wheels bundle them inside the
    binding's own package directory.
    """
    if not os.path.isdir(os.path.join(sys.prefix, 'conda-meta')):
        return False
    plugins = _qt_plugins_path()
    return plugins is not None and 'site-packages' not in plugins


def _fix_wayland_opengl() -> None:
    """Set QT_QPA_PLATFORM=xcb and PYOPENGL_PLATFORM=glx on Linux+Wayland.

    On Linux+Wayland, napari crashes on startup with ``OpenGL.error.Error:
    Attempt to retrieve context when no valid context`` whenever Qt renders
    through X11/XWayland while PyOpenGL defaults to its EGL backend. Forcing
    XCB (X11 via XWayland) + GLX keeps both sides on the same context. Uses
    ``os.environ.setdefault`` so any user-set values are never overridden. See
    https://github.com/napari/napari/issues/8808.

    The workaround is gated to setups where it actually helps, so healthy
    native-Wayland sessions are left alone:

    - It applies when the proprietary Nvidia driver is loaded (its Wayland
      support is incomplete) or when no native Qt Wayland platform plugin is
      installed (Qt falls back to XCB, so PyOpenGL must match it with GLX).
    - ``DISPLAY`` is set only when an X server is reachable, i.e. XWayland is
      running. Without it, forcing XCB would just abort instead of falling back
      to the native Wayland session. If on top of that Qt comes from conda
      without its Wayland plugin, no platform plugin can load at all, so an
      actionable hint is printed instead of leaving only Qt's cryptic abort
      message (see https://github.com/napari/napari/issues/9166).
    """
    if sys.platform != 'linux':
        return
    wayland_active = bool(os.environ.get('WAYLAND_DISPLAY')) or (
        os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland'
    )
    if not wayland_active:
        return
    if not os.environ.get('DISPLAY'):
        if not _native_wayland_plugin_available() and _qt_from_conda():
            sys.stderr.write(
                'You use Qt from conda and need qt6-wayland installed to '
                'start napari: conda install -c conda-forge qt6-wayland\n'
            )
        return
    if not _nvidia_driver_loaded() and _native_wayland_plugin_available():
        return
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'glx')
