"""Wayland/Nvidia OpenGL startup workaround.

Must run before ``napari._vispy`` (and thus PyOpenGL) is imported, so the
function lives in its own module called from ``napari/__init__.py`` rather
than from ``napari._qt.qt_event_loop``. PyOpenGL caches its platform at
``OpenGL.platform`` import time (see ``OpenGL/platform/__init__.py``); once
the EGL backend is loaded, later writes to ``PYOPENGL_PLATFORM`` are ignored.
"""

import os
import sys


def _fix_wayland_opengl() -> None:
    """Set QT_QPA_PLATFORM=xcb and PYOPENGL_PLATFORM=glx on Linux+Wayland.

    On Linux+Wayland with Nvidia hardware, the Nvidia driver's incomplete
    Wayland support causes napari to crash on startup across Qt bindings.
    Forcing XCB (X11 via XWayland) + GLX is the recommended workaround. Uses
    ``os.environ.setdefault`` so any user-set values are never overridden.
    See https://github.com/napari/napari/issues/8808.
    """
    if sys.platform != 'linux':
        return
    wayland_active = bool(os.environ.get('WAYLAND_DISPLAY')) or (
        os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland'
    )
    if not wayland_active:
        return
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'glx')
