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

    layer.selected_label = layer.colormap.background_value
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


def test_labels_polygon_with_downsampling(
    MouseEvent, make_napari_viewer, monkeypatch
):
    """Test that polygon overlay visual positions are correct with downsampling.

    This test verifies that when a Labels layer is downsampled (exceeding
    GL_MAX_TEXTURE_SIZE), the polygon overlay visual correctly transforms
    coordinates from data space to texture space for proper display.
    """
    # Patch get_max_texture_sizes to a small value
    monkeypatch.setattr(
        'napari._vispy.layers.base.get_max_texture_sizes',
        lambda: (256, 256),
    )

    viewer = make_napari_viewer()

    # Create a labels layer that will be downsampled
    shape = (600, 500)
    data = np.zeros(shape, dtype=np.int32)
    layer = viewer.add_labels(data, multiscale=False)

    expected_downsample = np.array([3, 2])
    np.testing.assert_array_equal(
        layer._transforms['tile2data'].scale, expected_downsample
    )

    layer.mode = 'polygon'
    layer.selected_label = 1

    polygon_overlay = layer._overlays['polygon']
    from napari._vispy.overlays.labels_polygon import VispyLabelsPolygonOverlay

    vispy_polygon_overlay = None
    for (
        overlay_visual
    ) in viewer.window._qt_viewer.canvas._layer_overlay_to_visual.get(
        layer, {}
    ).values():
        if isinstance(overlay_visual, VispyLabelsPolygonOverlay):
            vispy_polygon_overlay = overlay_visual
            break

    assert vispy_polygon_overlay is not None, (
        'Could not find polygon overlay visual'
    )

    # Define points in data coordinates (512x512 space)
    # These coordinates are what mouse events would provide
    data_points = [
        [200.5, 200.5],  # data coordinates
        [200.5, 300.5],
        [300.5, 300.5],
    ]

    # Set overlay points (simulating mouse clicks adding vertices)
    polygon_overlay.points = data_points

    # Get the visual positions that were set
    # The overlay's _on_points_change should have been called
    # Vispy Markers store position data in _data['a_position'] attribute
    visual_positions = vispy_polygon_overlay._nodes._data['a_position'][:, :2]

    # Expected visual positions should be in texture space
    # With 2x downsampling: texture_coord = data_coord / 2
    # Note: dims are reversed for vispy (y, x instead of x, y)
    expected_texture_positions = (
        np.array(
            [
                [200.5, 200.5],  # reversed: (y, x)
                [300.5, 200.5],
                [300.5, 300.5],
            ]
        )
        / 2
    )  # Apply downsampling

    # The visual positions should match the texture coordinates
    # (with coordinates properly transformed by tile2data.inverse)
    np.testing.assert_array_almost_equal(
        visual_positions,
        expected_texture_positions,
        decimal=1,
        err_msg=(
            'Polygon overlay visual positions should be in texture space '
            '(data coordinates divided by downsample factor) when downsampling is active'
        ),
    )
