"""Tests for QtHistogramControl widget."""

from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtWidgets import QWidget

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
