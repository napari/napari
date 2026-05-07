import numpy as np

from napari._qt.layer_controls.qt_surface_controls import QtSurfaceControls
from napari.layers import Surface

data = np.array([[0, 0], [0, 20], [10, 0], [10, 10]])
faces = np.array([[0, 1, 2], [1, 2, 3]])
values = np.linspace(0, 1, len(data))
_SURFACE = (data, faces, values)


def _assert_controls_enabled(controls, enabled):
    for control in controls:
        for label, widget in control.get_widget_controls():
            assert label.isEnabled() is enabled
            assert widget.isEnabled() is enabled


def _scalar_coloring_controls(qtctrl):
    return (
        qtctrl._contrast_limits_control,
        qtctrl._gamma_slider_control,
        qtctrl._colormap_control,
    )


def test_intensity_controls_disabled_with_vertex_colors(qtbot):
    vertex_colors = np.full((len(data), 3), 0.5)
    layer = Surface(_SURFACE, vertex_colors=vertex_colors)
    qtctrl = QtSurfaceControls(layer)
    qtbot.addWidget(qtctrl)

    _assert_controls_enabled(_scalar_coloring_controls(qtctrl), False)
    _assert_controls_enabled(
        (qtctrl._shading_combobox_control,),
        True,
    )


def test_intensity_controls_toggle_with_vertex_colors(qtbot):
    layer = Surface(_SURFACE)
    qtctrl = QtSurfaceControls(layer)
    qtbot.addWidget(qtctrl)

    _assert_controls_enabled(_scalar_coloring_controls(qtctrl), True)

    layer.vertex_colors = np.full((len(data), 3), 0.5)
    _assert_controls_enabled(_scalar_coloring_controls(qtctrl), False)

    layer.vertex_colors = None
    _assert_controls_enabled(_scalar_coloring_controls(qtctrl), True)
