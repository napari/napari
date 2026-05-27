import os
import sys

import pytest

from napari._wayland_fix import _fix_wayland_opengl


@pytest.mark.parametrize(
    ('platform', 'env'),
    [
        ('darwin', {}),
        ('linux', {'XDG_SESSION_TYPE': 'x11'}),
    ],
)
def test_fix_wayland_opengl_no_op(monkeypatch, platform, env):
    """Does not set env vars when not on Linux+Wayland."""
    monkeypatch.setattr(sys, 'platform', platform)
    for key in {'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE'} - env.keys():
        monkeypatch.delenv(key, raising=False)  # clear vars not explicitly set
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    monkeypatch.delenv('QT_QPA_PLATFORM', raising=False)
    monkeypatch.delenv('PYOPENGL_PLATFORM', raising=False)
    _fix_wayland_opengl()
    assert 'QT_QPA_PLATFORM' not in os.environ
    assert 'PYOPENGL_PLATFORM' not in os.environ


@pytest.mark.parametrize(
    'wayland_env',
    [
        {'WAYLAND_DISPLAY': 'wayland-0'},
        {'XDG_SESSION_TYPE': 'wayland'},
    ],
)
def test_fix_wayland_opengl_sets_vars(monkeypatch, wayland_env):
    """Sets QT_QPA_PLATFORM=xcb and PYOPENGL_PLATFORM=glx on Linux+Wayland."""
    monkeypatch.setattr(sys, 'platform', 'linux')
    for key in {'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE'} - wayland_env.keys():
        monkeypatch.delenv(key, raising=False)  # clear vars not explicitly set
    for key, val in wayland_env.items():
        monkeypatch.setenv(key, val)
    monkeypatch.delenv('QT_QPA_PLATFORM', raising=False)
    monkeypatch.delenv('PYOPENGL_PLATFORM', raising=False)
    _fix_wayland_opengl()
    assert os.environ['QT_QPA_PLATFORM'] == 'xcb'
    assert os.environ['PYOPENGL_PLATFORM'] == 'glx'


def test_fix_wayland_opengl_does_not_override(monkeypatch):
    """Does not override env vars already set by the user."""
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setenv('WAYLAND_DISPLAY', 'wayland-0')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'wayland')
    monkeypatch.setenv('PYOPENGL_PLATFORM', 'egl')
    _fix_wayland_opengl()
    assert os.environ['QT_QPA_PLATFORM'] == 'wayland'
    assert os.environ['PYOPENGL_PLATFORM'] == 'egl'
