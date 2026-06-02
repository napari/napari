"""Wayland/Nvidia OpenGL startup workaround.

Must run before ``napari._vispy`` (and thus PyOpenGL) is imported, so the
function lives in its own module called from ``napari/__init__.py`` rather
than from ``napari._qt.qt_event_loop``. PyOpenGL caches its platform at
``OpenGL.platform`` import time (see ``OpenGL/platform/__init__.py``); once
the EGL backend is loaded, later writes to ``PYOPENGL_PLATFORM`` are ignored.
"""

import os
import sys


def _nvidia_driver_loaded() -> bool:
    """Return True if the proprietary Nvidia kernel module is loaded.

    Uses the existence of ``/proc/driver/nvidia/version`` as a proxy.
    """
    return os.path.exists('/proc/driver/nvidia/version')


def _fix_wayland_opengl() -> None:
    """Set QT_QPA_PLATFORM=xcb and PYOPENGL_PLATFORM=glx on Linux+Wayland+Nvidia.

    On Linux+Wayland with Nvidia proprietary drivers, the driver's incomplete
    Wayland support causes napari to crash on startup across Qt bindings.
    Forcing XCB (X11 via XWayland) + GLX is the recommended workaround. Uses
    ``os.environ.setdefault`` so any user-set values are never overridden.
    See https://github.com/napari/napari/issues/8808.

    The workaround is gated narrowly to avoid regressing setups it can't help:

    - ``/proc/driver/nvidia/version`` only exists when the proprietary Nvidia
      kernel module is loaded, so it limits the hack to Nvidia hardware.
    - ``DISPLAY`` is set only when an X server is reachable, i.e. XWayland
      is running. Without it, forcing XCB would just abort instead of falling
      back to the native Wayland session.
    """
    if sys.platform != 'linux':
        return
    wayland_active = bool(os.environ.get('WAYLAND_DISPLAY')) or (
        os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland'
    )
    if not wayland_active:
        return
    if not _nvidia_driver_loaded():
        return
    if not os.environ.get('DISPLAY'):
        return
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'glx')
