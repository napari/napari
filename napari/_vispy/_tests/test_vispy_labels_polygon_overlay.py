import numpy as np

from napari._vispy.overlays.labels_polygon import VispyLabelsPolygonOverlay
from napari.components.overlays import LabelsPolygonOverlay


def test_vispy_brush_circle_overlay(make_napari_viewer):
    viewer = make_napari_viewer()

    labels_polygon = LabelsPolygonOverlay()

    data = np.zeros((50, 50), dtype=int)
    layer = viewer.add_labels(data, opacity=0.5)

    vispy_labels_polygon = VispyLabelsPolygonOverlay(
        layer=layer, overlay=labels_polygon
    )

    assert vispy_labels_polygon._polygon.color.alpha == 0.5

    labels_polygon.points = []
    assert not vispy_labels_polygon._line.visible
    assert not vispy_labels_polygon._polygon.visible

    labels_polygon.points = [(0, 0), (1, 1)]
    assert vispy_labels_polygon._line.visible
    assert not vispy_labels_polygon._polygon.visible
    assert np.allclose(
        vispy_labels_polygon._line.color[:3], layer._selected_color[:3]
    )

    labels_polygon.points = [(0, 0), (1, 1), (0, 3)]
    assert not vispy_labels_polygon._line.visible
    assert vispy_labels_polygon._polygon.visible

    layer.selected_label = layer._background_label
    assert vispy_labels_polygon._polygon.color.is_blank
