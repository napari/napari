import numpy as np
import pytest
from vispy import keys

from napari.components._interaction_box_constants import Box
from napari.utils.transforms import Affine


def check_corners_of_axis_aligned_interaction_box(
    box, *, top_left_corner=[0, 0], bottom_right_corner=None
):
    if not np.allclose(box[Box.TOP_LEFT], np.array(top_left_corner) - 0.5):
        pytest.fail(
            f"Top-left corner incorrect {box[Box.TOP_LEFT]} vs {np.array(top_left_corner) - 0.5}"
        )
    if not np.allclose(
        box[Box.BOTTOM_RIGHT], np.array(bottom_right_corner) - 0.5
    ):
        pytest.fail(
            f"Bottom-right corner incorrect {box[Box.BOTTOM_RIGHT]} vs {np.array(bottom_right_corner) - 0.5}"
        )


def test_interaction_box_display(make_napari_viewer):
    viewer = make_napari_viewer()

    data = np.random.random((2, 6, 30, 40))
    layer = viewer.add_image(data)

    layer.mode = 'transform'

    assert viewer.overlays.interaction_box.show
    check_corners_of_axis_aligned_interaction_box(
        viewer.overlays.interaction_box._box,
        top_left_corner=[0, 0],
        bottom_right_corner=[30, 40],
    )


def test_disable_with_3d(make_napari_viewer):
    viewer = make_napari_viewer()

    data = np.random.random((2, 6, 30, 40))
    layer = viewer.add_image(data)

    layer.mode = 'transform'
    viewer.dims.ndisplay = 3
    assert layer.mode == 'pan_zoom'
    with pytest.warns(UserWarning):
        layer.mode = 'transform'
    assert layer.mode == 'pan_zoom'


def test_disable_on_layer_cange(make_napari_viewer):
    viewer = make_napari_viewer()

    data = np.random.random((2, 6, 30, 40))
    layer = viewer.add_image(data)

    layer.mode = 'transform'
    viewer.add_image(data)
    assert viewer.overlays.interaction_box.show is False
    viewer.layers.selection.active = layer
    assert viewer.overlays.interaction_box.show is True


def test_interaction_box_dim_change(make_napari_viewer):

    viewer = make_napari_viewer()

    data = np.random.random((2, 6, 30, 40))
    layer = viewer.add_image(data)

    layer.mode = 'transform'

    viewer.dims._roll()

    check_corners_of_axis_aligned_interaction_box(
        viewer.overlays.interaction_box._box,
        top_left_corner=[0, 0],
        bottom_right_corner=[6, 30],
    )

    viewer.dims.transpose()

    check_corners_of_axis_aligned_interaction_box(
        viewer.overlays.interaction_box._box,
        top_left_corner=[0, 0],
        bottom_right_corner=[30, 6],
    )


def test_vertex_highlight(make_napari_viewer):
    viewer = make_napari_viewer()

    data = np.random.random((2, 6, 30, 40))
    layer = viewer.add_image(data)

    layer.mode = 'transform'

    viewer.overlays.interaction_box.selected_vertex = 9

    np.testing.assert_almost_equal(
        viewer.window._qt_viewer.interaction_box_visual.round_marker_node._data[
            'a_bg_color'
        ][
            0
        ][
            :-1
        ],
        viewer.window._qt_viewer.interaction_box_visual._highlight_color,
    )


def test_panzoom_on_space(make_napari_viewer):
    viewer = make_napari_viewer()
    view = viewer.window._qt_viewer
    data = np.random.random((2, 6, 30, 40))
    layer = viewer.add_image(data)

    layer.mode = 'transform'
    view.canvas.events.key_press(key=keys.Key('Space'))
    assert layer.mode == 'pan_zoom'
    assert viewer.overlays.interaction_box.show is False


def test_transform_coupling(make_napari_viewer):
    viewer = make_napari_viewer()

    data = np.random.random((2, 6, 30, 40))
    layer = viewer.add_image(data)
    layer.mode = 'transform'

    layer.affine = Affine(scale=[0.5, 0.5, 0.5, 0.5])
    np.testing.assert_almost_equal(
        viewer.overlays.interaction_box.transform.scale, [0.5, 0.5]
    )

    viewer.overlays.interaction_box.transform_drag = Affine(scale=[2.0, 2.0])
    np.testing.assert_almost_equal(layer.affine.scale, [0.5, 0.5, 2.0, 2.0])


def test_interaction_box_changes_with_layer_transform(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((28, 28)))
    layer.mode = 'transform'
    initial_selection_box = np.copy(
        viewer.overlays.interaction_box.transform.scale
    )
    layer.scale = [5, 5]
    np.testing.assert_almost_equal(
        initial_selection_box * 5,
        viewer.overlays.interaction_box.transform.scale,
    )
