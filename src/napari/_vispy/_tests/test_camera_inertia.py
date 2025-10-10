"""Tests for camera inertia."""

import numpy as np
import pytest

from napari._vispy.camera_inertia import InertiaConfig


def test_inertia_default_config():
    """Test default InertiaConfig values."""
    config = InertiaConfig()

    assert config.pan_friction == 5.0
    assert config.pan_damping == 0.6
    assert config.pan_max_speed == 200.0
    assert config.pan_min_speed == 4.0
    assert config.pan_stop_speed == 2.5

    assert config.rotate_friction == 7.0
    assert config.rotate_damping == 0.4
    assert config.rotate_max_speed == 120.0
    assert config.rotate_min_speed == 1.5
    assert config.rotate_stop_speed == 1.0


def test_inertia_config_validation():
    """Test InertiaConfig parameter validation."""
    # Valid config should not raise
    InertiaConfig(pan_friction=10.0, pan_damping=0.5)

    # Invalid friction should raise
    with pytest.raises(ValueError, match='pan_friction must be non-negative'):
        InertiaConfig(pan_friction=-1.0)

    # Invalid damping should raise
    with pytest.raises(
        ValueError, match='pan_damping must be between 0 and 1'
    ):
        InertiaConfig(pan_damping=1.5)


def test_camera_inertia_enabled_default(make_napari_viewer):
    """Test that camera inertia is enabled by default."""
    viewer = make_napari_viewer()

    assert viewer.camera.inertia is True


def test_camera_inertia_toggle(make_napari_viewer):
    """Test toggling camera inertia on and off."""
    viewer = make_napari_viewer()

    # Disable inertia
    viewer.camera.inertia = False
    assert viewer.camera.inertia is False

    # Re-enable inertia
    viewer.camera.inertia = True
    assert viewer.camera.inertia is True


def test_camera_inertia_syncs_with_canvas(make_napari_viewer):
    """Test that camera.inertia syncs with VispyCanvas._inertia.enabled."""
    viewer = make_napari_viewer()
    canvas_inertia = viewer.window._qt_viewer.canvas._inertia

    # Check default
    assert viewer.camera.inertia is True
    assert canvas_inertia.enabled is True

    # Disable via camera
    viewer.camera.inertia = False
    assert canvas_inertia.enabled is False

    # Re-enable via camera
    viewer.camera.inertia = True
    assert canvas_inertia.enabled is True


def test_inertia_stops_on_press(make_napari_viewer):
    """Test that pressing the mouse stops any ongoing inertia."""
    viewer = make_napari_viewer()
    inertia = viewer.window._qt_viewer.canvas._inertia

    # Simulate some velocity
    inertia._pan_velocity = np.array([10.0, 0.0, 0.0])
    inertia._timer.start(16)

    assert inertia._timer.isActive()

    # Press should stop animation
    inertia.on_press()

    assert not inertia._timer.isActive()
    assert inertia._pan_velocity is None


def test_inertia_2d_vs_3d(make_napari_viewer):
    """Test that rotation inertia only works in 3D mode."""
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((10, 10, 10)))
    inertia = viewer.window._qt_viewer.canvas._inertia

    # In 2D mode
    assert viewer.dims.ndisplay == 2
    inertia.on_drag()
    assert inertia._last_angles is None  # Should not track angles in 2D

    # Switch to 3D mode
    viewer.dims.ndisplay = 3
    inertia.on_drag()
    assert inertia._last_angles is not None  # Should track angles in 3D
