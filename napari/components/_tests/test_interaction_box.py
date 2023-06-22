import numpy as np

from napari.components.overlays.interaction_box import SelectionBoxOverlay
from napari.layers.base._base_constants import InteractionBoxHandle
from napari.layers.points import Points
from napari.layers.utils.interaction_box import (
    generate_interaction_box_vertices,
    generate_transform_box_from_layer,
    get_nearby_handle,
)


def test_transform_box_vertices_from_bounds():
    expected = np.array(
        [
            [0, 0],
            [10, 0],
            [0, 10],
            [10, 10],
            [0, 5],
            [5, 0],
            [5, 10],
            [10, 5],
            [-1, 5],
        ]
    )

    top_left = 0, 0
    bottom_right = 10, 10
    # works in vispy coordinates, so x and y are swapped
    vertices = generate_interaction_box_vertices(
        top_left, bottom_right, handles=False
    )
    np.testing.assert_allclose(vertices, expected[:4, ::-1])
    vertices = generate_interaction_box_vertices(
        top_left, bottom_right, handles=True
    )
    np.testing.assert_allclose(vertices, expected[:, ::-1])


def test_transform_box_from_layer():
    pts = np.array([[0, 0], [10, 10]])
    translate = [-2, 3]
    scale = [4, 5]
    # size of 2 means wider bounding box by 1 in every direction
    pt_size = 2
    layer = Points(pts, translate=translate, scale=scale, size=pt_size)
    vertices = generate_transform_box_from_layer(layer, dims_displayed=(0, 1))
    # scale/translate should not affect vertices, cause they're in data space
    expected = np.array(
        [
            [-1, -1],
            [11, -1],
            [-1, 11],
            [11, 11],
            [-1, 5],
            [5, -1],
            [5, 11],
            [11, 5],
            [-2.2, 5],
        ]
    )
    np.testing.assert_allclose(vertices, expected)


def test_transform_box_get_nearby_handle():
    # square box from (0, 0) to (10, 10)
    vertices = np.array(
        [
            [0, 0],
            [10, 0],
            [0, 10],
            [10, 10],
            [0, 5],
            [5, 0],
            [5, 10],
            [10, 5],
            [-1, 5],
        ]
    )
    near_top_left = [0.04, -0.05]
    top_left = get_nearby_handle(near_top_left, vertices)
    assert top_left == InteractionBoxHandle.TOP_LEFT
    near_rotation = [-1.05, 4.95]
    rotation = get_nearby_handle(near_rotation, vertices)
    assert rotation == InteractionBoxHandle.ROTATION
    middle = [5, 5]
    inside = get_nearby_handle(middle, vertices)
    assert inside == InteractionBoxHandle.INSIDE
    outside = [12, -1]
    none = get_nearby_handle(outside, vertices)
    assert none is None


def test_selection_box_from_points():
    points = np.array(
        [
            [0, 5],
            [-3, 0],
            [0, 7],
        ]
    )
    selection_box = SelectionBoxOverlay()
    selection_box.update_from_points(points)
    assert selection_box.bounds == ((-3, 0), (0, 7))
