import os
import sys

import pytest

from napari._wayland_fix import _fix_wayland_opengl

# Should only exist on Linux with the proprietary Nvidia driver loaded
NVIDIA_PATH = '/proc/driver/nvidia/version'

# Env that satisfies every gate: Linux + Wayland + Nvidia + reachable X server.
WAYLAND_NVIDIA_ENV = {'WAYLAND_DISPLAY': 'wayland-0', 'DISPLAY': ':0'}


def _apply_env(monkeypatch, env):
    """Set exactly the given env vars, clearing the ones we care about first."""
    managed = {'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE', 'DISPLAY'}
    for key in managed - env.keys():
        monkeypatch.delenv(key, raising=False)  # clear vars not explicitly set
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    monkeypatch.delenv('QT_QPA_PLATFORM', raising=False)
    monkeypatch.delenv('PYOPENGL_PLATFORM', raising=False)


def _patch_nvidia(monkeypatch, present):
    """Make ``os.path.exists`` report the Nvidia driver file as present (or not)."""
    real_exists = os.path.exists
    monkeypatch.setattr(
        os.path,
        'exists',
        lambda p: present if p == NVIDIA_PATH else real_exists(p),
    )


@pytest.mark.parametrize(
    ('platform', 'env', 'nvidia'),
    [
        # Not Linux.
        ('darwin', WAYLAND_NVIDIA_ENV, True),
        # Linux but not Wayland.
        ('linux', {'XDG_SESSION_TYPE': 'x11', 'DISPLAY': ':0'}, True),
        # Linux + Wayland but no Nvidia proprietary driver.
        ('linux', WAYLAND_NVIDIA_ENV, False),
        # Linux + Wayland + Nvidia but no reachable X server (no XWayland).
        ('linux', {'WAYLAND_DISPLAY': 'wayland-0'}, True),
    ],
)
def test_fix_wayland_opengl_no_op(monkeypatch, platform, env, nvidia):
    """Does not set env vars unless Linux+Wayland+Nvidia+X server all hold."""
    monkeypatch.setattr(sys, 'platform', platform)
    _patch_nvidia(monkeypatch, nvidia)
    _apply_env(monkeypatch, env)
    _fix_wayland_opengl()
    assert 'QT_QPA_PLATFORM' not in os.environ
    assert 'PYOPENGL_PLATFORM' not in os.environ


@pytest.mark.parametrize(
    'wayland_env',
    [
        {'WAYLAND_DISPLAY': 'wayland-0', 'DISPLAY': ':0'},
        {'XDG_SESSION_TYPE': 'wayland', 'DISPLAY': ':0'},
    ],
)
def test_fix_wayland_opengl_sets_vars(monkeypatch, wayland_env):
    """Sets xcb+glx on Linux+Wayland+Nvidia with a reachable X server."""
    monkeypatch.setattr(sys, 'platform', 'linux')
    _patch_nvidia(monkeypatch, True)
    _apply_env(monkeypatch, wayland_env)
    _fix_wayland_opengl()
    assert os.environ['QT_QPA_PLATFORM'] == 'xcb'
    assert os.environ['PYOPENGL_PLATFORM'] == 'glx'


def test_fix_wayland_opengl_does_not_override(monkeypatch):
    """Does not override env vars already set by the user."""
    monkeypatch.setattr(sys, 'platform', 'linux')
    _patch_nvidia(monkeypatch, True)
    _apply_env(monkeypatch, WAYLAND_NVIDIA_ENV)
    monkeypatch.setenv('QT_QPA_PLATFORM', 'wayland')
    monkeypatch.setenv('PYOPENGL_PLATFORM', 'egl')
    _fix_wayland_opengl()
    assert os.environ['QT_QPA_PLATFORM'] == 'wayland'
    assert os.environ['PYOPENGL_PLATFORM'] == 'egl'
