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
def test_qt_container_is_thick_change(dynamic, qtbot, viewer_model):
    """Changing the dims margins should update the controls' is_thick state."""
    get_settings().experimental.dynamic_layer_controls = dynamic
    cont = QtLayerControlsContainer(viewer_model)
    qtbot.addWidget(cont)
    layer = viewer_model.add_image(np.zeros((5, 5, 5)))

    assert viewer_model.dims.is_thick is False

    viewer_model.dims.margin_left = (1, 0, 0)
    assert viewer_model.dims.is_thick is True
    panel = cont.panel if dynamic else cont.widgets[layer]
    assert panel.is_thick is True
    if dynamic:
        projection_control = next(
            ctrl
            for ctrl in panel._controls
            if hasattr(ctrl, 'projection_combobox')
        )
    else:
        projection_control = panel._projection_mode_control
    assert not projection_control.projection_combobox.isHidden()

    viewer_model.dims.margin_left = (0, 0, 0)
    assert viewer_model.dims.is_thick is False
    assert panel.is_thick is False
    assert projection_control.projection_combobox.isHidden()


@pytest.mark.parametrize('dynamic', [True, False])
def test_qt_container_theme_change(dynamic, qtbot, viewer_model):
    get_settings().experimental.dynamic_layer_controls = dynamic
    cont = QtLayerControlsContainer(viewer_model)
    qtbot.addWidget(cont)
    viewer_model.add_image(np.arange(25).reshape(5, 5))
    viewer_model.theme = 'light'
    # only affects histogram; too nasty to actualy check for, but
    # at least this runs the lines to ensure nothign crashes
