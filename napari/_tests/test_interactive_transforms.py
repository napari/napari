import numpy as np
import pytest
from vispy import keys

from napari.utils.transforms import Affine


def test_interaction_box_display(make_napari_viewer):
    viewer = make_napari_viewer()

    data = np.random.random((2, 6, 30, 40))
    layer = viewer.add_image(data)

    layer.mode = 'transform'

    assert viewer.overlays.interaction_box.show
    np.testing.assert_almost_equal(
        viewer.overlays.interaction_box._box,
        [
            [-0.5, -0.5],
            [14.5, -0.5],
            [29.5, -0.5],
            [29.5, 19.5],
            [29.5, 39.5],
            [14.5, 39.5],
            [-0.5, 39.5],
            [-0.5, 19.5],
            [14.5, 19.5],
        ],
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

    np.testing.assert_almost_equal(
        viewer.overlays.interaction_box._box,
        [
            [-0.5, -0.5],
            [2.5, -0.5],
            [5.5, -0.5],
            [5.5, 14.5],
            [5.5, 29.5],
            [2.5, 29.5],
            [-0.5, 29.5],
            [-0.5, 14.5],
            [2.5, 14.5],
        ],
    )

    viewer.dims._transpose()
    np.testing.assert_almost_equal(
        viewer.overlays.interaction_box._box,
        [
            [-0.5, -0.5],
            [14.5, -0.5],
            [29.5, -0.5],
            [29.5, 2.5],
            [29.5, 5.5],
            [14.5, 5.5],
            [-0.5, 5.5],
            [-0.5, 2.5],
            [14.5, 2.5],
        ],
    )


def test_vertex_highlight(make_napari_viewer):
    viewer = make_napari_viewer()

    data = np.random.random((2, 6, 30, 40))
    layer = viewer.add_image(data)

    layer.mode = 'transform'

    viewer.overlays.interaction_box.selected_vertex = 9

    np.testing.assert_almost_equal(
        viewer.window.qt_viewer.interaction_box_visual.round_marker_node._data[
            'a_fg_color'
        ][0][:-1],
        viewer.window.qt_viewer.interaction_box_visual._highlight_color,
    )


def test_panzoom_on_space(make_napari_viewer):
    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer
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
