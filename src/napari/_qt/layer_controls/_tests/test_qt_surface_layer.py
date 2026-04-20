import numpy as np

from napari._qt.layer_controls.qt_surface_controls import QtSurfaceControls
from napari.layers import Surface
from napari.layers.surface._surface_constants import SHADING_TRANSLATION

data = np.array([[0, 0], [0, 20], [10, 0], [10, 10]])
faces = np.array([[0, 1, 2], [1, 2, 3]])
values = np.linspace(0, 1, len(data))
_SURFACE = (data, faces, values)


def test_shading_combobox(qtbot):
    layer = Surface(_SURFACE)
    qtctrl = QtSurfaceControls(layer)
    qtbot.addWidget(qtctrl)
    assert (
        qtctrl._shading_combobox_control.shading_combobox.currentText()
        == layer.shading
    )

    for display, shading in SHADING_TRANSLATION.items():
        qtctrl._shading_combobox_control.shading_combobox.setCurrentText(
            display
        )
        assert layer.shading == shading

    for display, shading in SHADING_TRANSLATION.items():
        layer.shading = shading
        assert (
            qtctrl._shading_combobox_control.shading_combobox.currentText()
            == display
        )


def test_intensity_controls_disabled_with_vertex_colors(qtbot):
    vertex_colors = np.full((len(data), 3), 0.5)
    layer = Surface(_SURFACE, vertex_colors=vertex_colors)
    qtctrl = QtSurfaceControls(layer)
    qtbot.addWidget(qtctrl)

    assert all(
        not widget.isEnabled()
        for ctrl in qtctrl._intensity_controls
        for _, widget in ctrl.get_widget_controls()
    )
    assert qtctrl._shading_combobox_control.shading_combobox.isEnabled()


def test_intensity_controls_toggle_with_vertex_colors(qtbot):
    layer = Surface(_SURFACE)
    qtctrl = QtSurfaceControls(layer)
    qtbot.addWidget(qtctrl)

    assert all(
        widget.isEnabled()
        for ctrl in qtctrl._intensity_controls
        for _, widget in ctrl.get_widget_controls()
    )

    layer.vertex_colors = np.full((len(data), 3), 0.5)
    assert all(
        not widget.isEnabled()
        for ctrl in qtctrl._intensity_controls
        for _, widget in ctrl.get_widget_controls()
    )

    layer.vertex_colors = None
    assert all(
        widget.isEnabled()
        for ctrl in qtctrl._intensity_controls
        for _, widget in ctrl.get_widget_controls()
    )
