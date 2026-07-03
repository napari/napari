"""Tests for QtHistogramControl widget."""

from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget

from napari._qt.layer_controls.qt_image_controls import QtImageControls
from napari._qt.layer_controls.widgets.qt_histogram_control import (
    QtHistogramControl,
)
from napari.layers import Image


def _create_control(qtbot, layer=None):
    """Helper: create a QtHistogramControl with qtbot-tracked parent."""
    if layer is None:
        layer = Image(np.random.rand(10, 10))
    parent = QWidget()
    qtbot.addWidget(parent)
    # Keep parent reference alive to prevent C++ object teardown
    control = QtHistogramControl(parent, layer)
    control._parent_ref = parent
    return control


def test_ensure_content_creates_children(qtbot):
    """ensure_content() should create histogram_content, histogram_widget, settings_widget."""
    control = _create_control(qtbot)

    # Before ensure_content
    assert control.histogram_content is None
    assert control.histogram_widget is None
    assert control.settings_widget is None

    # After ensure_content
    control.ensure_content()
    assert control.histogram_content is not None
    assert control.histogram_widget is not None
    assert control.settings_widget is not None


def test_ensure_content_idempotent(qtbot):
    """Calling ensure_content() twice should not create duplicate widgets."""
    control = _create_control(qtbot)
    control.ensure_content()
    first_content = control.histogram_content

    control.ensure_content()
    assert control.histogram_content is first_content


def test_content_widget_starts_hidden(qtbot):
    """The content_widget should be hidden by default."""
    control = _create_control(qtbot)
    assert control.content_widget.isHidden()


def test_get_widget_controls_returns_empty(qtbot):
    """get_widget_controls should return an empty list."""
    control = _create_control(qtbot)
    assert control.get_widget_controls() == []


def test_disconnect_widget_controls_calls_cleanup(qtbot):
    """disconnect_widget_controls should call cleanup on histogram_content."""
    control = _create_control(qtbot)
    control.ensure_content()
    assert control.histogram_content is not None

    # Should not crash
    control.disconnect_widget_controls()

    # Second call should be safe (cleanup idempotency)
    control.disconnect_widget_controls()


def test_disconnect_without_ensure_content_is_safe(qtbot):
    """disconnect_widget_controls should be safe if ensure_content was never called."""
    control = _create_control(qtbot)
    # histogram_content is None
    control.disconnect_widget_controls()


@pytest.mark.parametrize(
    'layer_kwargs',
    [
        {},
        {'rgb': True},
        {'multiscale': False},
    ],
)
def test_histogram_control_works_with_various_image_configs(
    qtbot, layer_kwargs
):
    """QtHistogramControl should work with different Image configurations."""
    data = (
        np.random.rand(8, 8, 3)
        if layer_kwargs.get('rgb')
        else np.random.rand(8, 8)
    )
    layer = Image(data, **layer_kwargs)
    control = _create_control(qtbot, layer)
    control.ensure_content()
    assert control.histogram_content is not None
    assert control.histogram_widget is not None


def test_histogram_control_show_hide_toggle_integration(qtbot):
    """Integration test: toggle button should show/hide the content_widget."""
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    control = qtctrl._histogram_control
    assert control is not None
    assert control.content_widget.isHidden()

    button = qtctrl._contrast_limits_control.histogram_button
    assert button is not None

    # Left click to show
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert not control.content_widget.isHidden()
    assert layer.histogram.enabled

    # Left click to hide
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert control.content_widget.isHidden()
    assert not layer.histogram.enabled
