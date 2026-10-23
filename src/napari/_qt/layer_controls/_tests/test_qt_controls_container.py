import numpy as np
import pytest

from napari._qt.layer_controls.dynamic.qt_dynamic_layer_controls import (
    QtDynamicLayerControls,
)
from napari._qt.layer_controls.qt_layer_controls_container import (
    QtLayerControlsContainer,
)
from napari._qt.layer_controls.qt_points_controls import QtPointsControls
from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari.settings import get_settings
from napari.utils.theme import get_theme


@pytest.mark.parametrize('dynamic', [True, False])
def test_qt_container_creation(dynamic, qtbot, viewer_model):
    get_settings().experimental.dynamic_layer_controls = dynamic
    cont = QtLayerControlsContainer(viewer_model)
    qtbot.addWidget(cont)
    viewer_model.add_points()
    if dynamic:
        assert isinstance(cont.currentWidget(), QtDynamicLayerControls)
    else:
        assert isinstance(cont.currentWidget(), QtPointsControls)


@pytest.mark.parametrize('dynamic', [True, False])
def test_qt_container_theme_change(dynamic, qtbot, viewer_model):
    settings = get_settings()
    settings.experimental.dynamic_layer_controls = dynamic

    cont = QtLayerControlsContainer(viewer_model)
    qtbot.addWidget(cont)
    viewer_model.add_image(np.arange(25).reshape(5, 5))

    widgets = cont.findChildren(QtHistogramWidget)
    for widget in widgets:
        widget._ensure_histogram_content()

    settings.appearance.theme = 'light'
    light_theme = get_theme('light')
    light_bg = np.array(light_theme.canvas.as_rgb_tuple()) / 255
    dark_theme = get_theme('dark')
    dark_bg = np.array(dark_theme.canvas.as_rgb_tuple()) / 255

    for widget in widgets:
        bgcolor = widget.histogram_canvas.canvas.bgcolor.rgba[:3]
        np.testing.assert_allclose(bgcolor, light_bg)

    settings.appearance.theme = 'dark'
    for widget in widgets:
        bgcolor = widget.histogram_canvas.canvas.bgcolor.rgba[:3]
        np.testing.assert_allclose(bgcolor, dark_bg)

    # viewer theme overrides settings
    viewer_model.theme = 'light'
    for widget in widgets:
        bgcolor = widget.histogram_canvas.canvas.bgcolor.rgba[:3]
        np.testing.assert_allclose(bgcolor, light_bg)
