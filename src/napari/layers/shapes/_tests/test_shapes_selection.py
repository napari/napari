import numpy as np

from napari.components import ViewerModel


def test_preserve_selection_toggling_3d():
    viewer = ViewerModel()
    # Create a shapes layer with a rectangle
    shapes = [np.array([[10, 10], [10, 20], [20, 20], [20, 10]])]
    layer = viewer.add_shapes(shapes)

    # Select the shape
    layer.selected_data = {0}
    assert layer.selected_data == {0}

    # Toggle 3D mode
    viewer.dims.ndisplay = 3
    # In some cases we might need to wait for events or call _set_view_slice
    # but ViewerModel usually handles this.

    # Selection should be preserved
    assert layer.selected_data == {0}


def test_preserve_selection_scrolling():
    viewer = ViewerModel()
    # Create a shapes layer with rectangles on different slices
    shapes = [
        np.array([[0, 10, 10], [0, 10, 20], [0, 20, 20], [0, 20, 10]]),
        np.array([[2, 10, 10], [2, 10, 20], [2, 20, 20], [2, 20, 10]]),
    ]
    layer = viewer.add_shapes(shapes)

    # Select the shape on slice 0
    layer.selected_data = {0}
    assert layer.selected_data == {0}

    # Scroll to next slice
    viewer.dims.set_point(0, 1)

    # Selection should be preserved
    assert layer.selected_data == {0}


def test_preserve_selection_order_change():
    viewer = ViewerModel()
    # Create a shapes layer
    shapes = [np.array([[0, 10, 10], [0, 10, 20], [0, 20, 20], [0, 20, 10]])]
    layer = viewer.add_shapes(shapes)

    # Select the shape
    layer.selected_data = {0}
    assert layer.selected_data == {0}

    # Change display order
    viewer.dims.order = (0, 1, 2)

    # Selection should be preserved
    assert layer.selected_data == {0}


def test_hover_highlight_cleared_on_scrolling():
    viewer = ViewerModel()
    # Add a dummy 3D image to set the dimensions properly
    viewer.add_image(np.zeros((5, 30, 30)))
    # Create a shapes layer with a rectangle only on slice 0
    shapes = [np.array([[0, 10, 10], [0, 10, 20], [0, 20, 20], [0, 20, 10]])]
    layer = viewer.add_shapes(shapes)

    # Set mode to SELECT
    layer.mode = 'select'

    # Mock hover over shape 0 on slice 0
    layer._value = (0, None)

    # Scroll to slice 1
    viewer.dims.set_point(0, 1)

    # _value should be reset
    assert layer._value == (None, None)

    # Outline highlight should NOT be visible on slice 1
    outline_vertices, _ = layer._outline_shapes()
    assert outline_vertices is None or outline_vertices.shape[0] == 0


def test_interaction_box_tracks_current_slice_multiselect():
    """Selecting shapes on adjacent slices must not leave a stale highlight.

    Regression test: when multiple shapes are selected across adjacent
    slices, the interaction box (``_selected_box``) used for handle
    hit-testing must follow the shape that is in view on the current slice.
    Previously it lagged one slice behind because ``_set_view_slice`` rebuilt
    it inside a batched update, before ``_indices_view`` reflected the new
    slice, leaving the previous slice's box hoverable as a phantom highlight.
    """
    viewer = ViewerModel()
    viewer.add_image(np.zeros((3, 50, 50)))

    # shape 0 lives on slice 0 (~15, 15), shape 1 on slice 1 (~35, 35)
    shapes = [
        np.array([[0, 10, 10], [0, 10, 20], [0, 20, 20], [0, 20, 10]]),
        np.array([[1, 30, 30], [1, 30, 40], [1, 40, 40], [1, 40, 30]]),
    ]
    layer = viewer.add_shapes(shapes)
    layer.mode = 'select'

    viewer.dims.set_point(0, 0)
    layer.selected_data = {0, 1}

    def box_center():
        return layer._selected_box[8]

    def value_at(point):
        return layer.get_value(
            np.array(point),
            view_direction=None,
            dims_displayed=[1, 2],
            world=False,
        )

    # On slice 0 the box is around shape 0.
    np.testing.assert_allclose(box_center(), [15, 15])

    viewer.dims.set_point(0, 1)
    # On slice 1 the box must follow shape 1, not stay on shape 0.
    np.testing.assert_allclose(box_center(), [35, 35])
    # Hovering where shape 0's box used to be (now empty) must not produce a
    # phantom handle hit.
    assert value_at((1, 10, 10)) == (None, None)
    assert value_at((1, 20, 20)) == (None, None)
    # A genuine handle hover must be attributed to the in-view shape (1), not
    # to the first shape in the (cross-slice) selection (0).
    hovered_shape, hovered_vertex = value_at((1, 30, 30))
    assert hovered_shape == 1
    assert hovered_vertex is not None

    viewer.dims.set_point(0, 0)
    # Scrolling back: the box follows shape 0 again.
    np.testing.assert_allclose(box_center(), [15, 15])
    assert value_at((0, 30, 30)) == (None, None)
