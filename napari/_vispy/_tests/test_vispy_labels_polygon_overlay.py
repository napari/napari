import numpy as np

from napari._vispy.overlays.labels_polygon import VispyLabelsPolygonOverlay
from napari.components.overlays import LabelsPolygonOverlay
from napari.layers.labels._labels_key_bindings import complete_polygon
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
)


def test_vispy_labels_polygon_overlay(make_napari_viewer):
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


def test_labels_drawing_with_polygons(MouseEvent, make_napari_viewer):
    """Test polygon painting."""
    np.random.seed(0)

    data = np.zeros((3, 15, 15), dtype=np.int32)
    viewer = make_napari_viewer()
    layer = viewer.add_labels(data)

    layer.mode = 'polygon'
    layer.selected_label = 1

    # Place some random points and then cancel them all
    for _ in range(5):
        position = (0,) + tuple(np.random.randint(20, size=2))
        event = MouseEvent(
            type='mouse_press',
            button=1,
            position=position,
            dims_displayed=[1, 2],
        )
        mouse_press_callbacks(layer, event)

    # Cancel all the points
    for _ in range(5):
        event = MouseEvent(
            type='mouse_press',
            button=2,
            position=(0, 0, 0),
            dims_displayed=(1, 2),
        )
        mouse_press_callbacks(layer, event)

    assert np.array_equiv(data[0, :], 0)

    # Draw a rectangle (the latest two points will be cancelled)
    points = [
        (1, 1, 1),
        (1, 1, 10),
        (1, 10, 10),
        (1, 10, 1),
        (1, 12, 0),
        (1, 0, 0),
    ]
    for position in points:
        event = MouseEvent(
            type='mouse_move',
            button=None,
            position=(1,) + tuple(np.random.randint(20, size=2)),
            dims_displayed=(1, 2),
        )
        mouse_move_callbacks(layer, event)

        event = MouseEvent(
            type='mouse_press',
            button=1,
            position=position,
            dims_displayed=[1, 2],
        )
        mouse_press_callbacks(layer, event)

    # Cancel the latest two points
    for _ in range(2):
        event = MouseEvent(
            type='mouse_press',
            button=2,
            position=(1, 5, 1),
            dims_displayed=(1, 2),
        )
        mouse_press_callbacks(layer, event)

    # Finish drawing
    complete_polygon(layer)

    assert np.array_equiv(data[[0, 2], :], 0)
    assert np.array_equiv(data[1, 1:11, 1:11], 1)
    assert np.array_equiv(data[1, 0, :], 0)
    assert np.array_equiv(data[1, :, 0], 0)
    assert np.array_equiv(data[1, 11:, :], 0)
    assert np.array_equiv(data[1, :, 11:], 0)

    # Try to finish with an incomplete polygon
    for position in [(0, 1, 1)]:
        event = MouseEvent(
            type='mouse_press',
            button=1,
            position=position,
            dims_displayed=(1, 2),
        )
        mouse_press_callbacks(layer, event)

    # Finish drawing
    complete_polygon(layer)
    assert np.array_equiv(data[0, :], 0)
