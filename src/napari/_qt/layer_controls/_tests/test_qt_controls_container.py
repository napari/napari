import numpy as np
import pytest

from napari._qt.layer_controls.dynamic.qt_dynamic_layer_controls import (
    QtDynamicLayerControls,
)
from napari._qt.layer_controls.qt_layer_controls_container import (
    QtLayerControlsContainer,
)
from napari._qt.layer_controls.qt_points_controls import QtPointsControls
from napari.settings import get_settings


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
    cont = QtLayerControlsContainer(viewer_model)
    qtbot.addWidget(cont)
    viewer_model.add_image(np.arange(25).reshape(5, 5))
    get_settings().appearance.theme = 'light'
    # only affects histogram; too nasty to actualy check for, but
    # at least this runs the lines to ensure nothign crashes
