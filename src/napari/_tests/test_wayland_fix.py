import os
import sys

import pytest

from napari import _wayland_fix
from napari._wayland_fix import _fix_wayland_opengl

# Env that satisfies the platform gates: Linux + Wayland + reachable X server.
WAYLAND_ENV = {'WAYLAND_DISPLAY': 'wayland-0', 'DISPLAY': ':0'}


def _apply_env(monkeypatch, env):
    """Set exactly the given env vars, clearing the ones we care about first."""
    managed = {'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE', 'DISPLAY'}
    for key in managed - env.keys():
        monkeypatch.delenv(key, raising=False)  # clear vars not explicitly set
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    monkeypatch.delenv('QT_QPA_PLATFORM', raising=False)
    monkeypatch.delenv('PYOPENGL_PLATFORM', raising=False)


def _patch_gpu(monkeypatch, nvidia, wayland_plugin, qt_from_conda=False):
    """Stub the Nvidia-driver, Wayland-plugin and conda-Qt probes."""
    monkeypatch.setattr(_wayland_fix, '_nvidia_driver_loaded', lambda: nvidia)
    monkeypatch.setattr(
        _wayland_fix,
        '_native_wayland_plugin_available',
        lambda: wayland_plugin,
    )
    monkeypatch.setattr(_wayland_fix, '_qt_from_conda', lambda: qt_from_conda)


@pytest.mark.parametrize(
    ('platform', 'env', 'nvidia', 'wayland_plugin'),
    [
        # Not Linux.
        ('darwin', WAYLAND_ENV, True, False),
        # Linux but not Wayland.
        ('linux', {'XDG_SESSION_TYPE': 'x11', 'DISPLAY': ':0'}, True, False),
        # Linux + Wayland but no reachable X server (no XWayland).
        ('linux', {'WAYLAND_DISPLAY': 'wayland-0'}, True, False),
        # Healthy native Wayland: no Nvidia and the plugin is present.
        ('linux', WAYLAND_ENV, False, True),
    ],
)
def test_fix_wayland_opengl_no_op(
    monkeypatch, platform, env, nvidia, wayland_plugin
):
    """Leaves env untouched unless the workaround can actually help."""
    monkeypatch.setattr(sys, 'platform', platform)
    _patch_gpu(monkeypatch, nvidia, wayland_plugin)
    _apply_env(monkeypatch, env)
    _fix_wayland_opengl()
    assert 'QT_QPA_PLATFORM' not in os.environ
    assert 'PYOPENGL_PLATFORM' not in os.environ


@pytest.mark.parametrize(
    ('nvidia', 'wayland_plugin'),
    [
        (True, True),  # Nvidia: native Wayland is broken even with the plugin
        (True, False),  # Nvidia, no plugin
        (False, False),  # integrated GPU, no plugin -> Qt falls back to XCB
    ],
)
def test_fix_wayland_opengl_sets_vars(monkeypatch, nvidia, wayland_plugin):
    """Sets xcb+glx on Nvidia or when no native Wayland plugin is present."""
    monkeypatch.setattr(sys, 'platform', 'linux')
    _patch_gpu(monkeypatch, nvidia, wayland_plugin)
    _apply_env(monkeypatch, WAYLAND_ENV)
    _fix_wayland_opengl()
    assert os.environ['QT_QPA_PLATFORM'] == 'xcb'
    assert os.environ['PYOPENGL_PLATFORM'] == 'glx'


@pytest.mark.parametrize(
    ('wayland_plugin', 'qt_from_conda', 'expect_hint'),
    [
        (False, True, True),  # conda Qt without qt6-wayland: hint
        (False, False, False),  # pip Qt bundles the plugin differently
        (True, True, False),  # plugin present: native Wayland works
    ],
)
def test_fix_wayland_opengl_no_display_hint(
    monkeypatch, capsys, wayland_plugin, qt_from_conda, expect_hint
):
    """Prints an install hint when Qt cannot load any platform plugin."""
    monkeypatch.setattr(sys, 'platform', 'linux')
    _patch_gpu(monkeypatch, False, wayland_plugin, qt_from_conda)
    _apply_env(monkeypatch, {'WAYLAND_DISPLAY': 'wayland-0'})
    _fix_wayland_opengl()
    err = capsys.readouterr().err
    assert ('qt6-wayland' in err) == expect_hint
    assert 'QT_QPA_PLATFORM' not in os.environ
    assert 'PYOPENGL_PLATFORM' not in os.environ


def test_fix_wayland_opengl_does_not_override(monkeypatch):
    """Does not override env vars already set by the user."""
    monkeypatch.setattr(sys, 'platform', 'linux')
    _patch_gpu(monkeypatch, nvidia=False, wayland_plugin=False)
    _apply_env(monkeypatch, WAYLAND_ENV)
    monkeypatch.setenv('QT_QPA_PLATFORM', 'wayland')
    monkeypatch.setenv('PYOPENGL_PLATFORM', 'egl')
    _fix_wayland_opengl()
    assert os.environ['QT_QPA_PLATFORM'] == 'wayland'
    assert os.environ['PYOPENGL_PLATFORM'] == 'egl'
