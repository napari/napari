from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from napari_plugin_engine import HookImplementation

from napari._tests.utils import layer_test_data
from napari.components.viewer_model import ViewerModel
from napari.layers._source import Source

img = np.random.rand(10, 10)
layer_data = [(lay[1], {}, lay[0].__name__.lower()) for lay in layer_test_data]


def _impl(path):
    """just a dummy Hookimpl object to return from mocks"""


_testimpl = HookImplementation(_impl, plugin_name='testimpl')


@pytest.mark.parametrize("layer_datum", layer_data)
def test_add_layers_with_plugins(layer_datum):
    """Test that add_layers_with_plugins adds the expected layer types."""
    with patch(
        "napari.plugins.io.read_data_with_plugins",
        MagicMock(return_value=([layer_datum], _testimpl)),
    ):
        v = ViewerModel()
        v._add_layers_with_plugins(['mock_path'], stack=False)
        layertypes = [layer._type_string for layer in v.layers]
        assert layertypes == [layer_datum[2]]

        expected_source = Source(path='mock_path', reader_plugin='testimpl')
        assert all(lay.source == expected_source for lay in v.layers)


@patch(
    "napari.plugins.io.read_data_with_plugins",
    MagicMock(return_value=([], _testimpl)),
)
def test_plugin_returns_nothing():
    """Test that a plugin returning nothing adds nothing to the Viewer."""
    v = ViewerModel()
    v._add_layers_with_plugins(['mock_path'], stack=False)
    assert not v.layers


@patch(
    "napari.plugins.io.read_data_with_plugins",
    MagicMock(return_value=([(img,)], _testimpl)),
)
def test_viewer_open():
    """Test that a plugin to returning an image adds stuff to the viewer."""
    viewer = ViewerModel()
    assert len(viewer.layers) == 0
    viewer.open('mock_path.tif')
    assert len(viewer.layers) == 1
    # The name should be taken from the path name, stripped of extension
    assert viewer.layers[0].name == 'mock_path'

    # stack=True also works... and very long names are truncated
    viewer.open('mock_path.tif', stack=True)
    assert len(viewer.layers) == 2
    assert viewer.layers[1].name.startswith('mock_path')

    expected_source = Source(path='mock_path.tif', reader_plugin='testimpl')
    assert all(lay.source == expected_source for lay in viewer.layers)

    viewer.open([], stack=[], plugin=None)
    assert len(viewer.layers) == 2


def test_viewer_open_no_plugin(tmp_path):
    viewer = ViewerModel()
    fname = tmp_path / 'gibberish.gbrsh'
    fname.touch()
    with pytest.raises(ValueError, match=".*gibberish.gbrsh.*"):
        # will default to builtins
        viewer.open(fname)


plugin_returns = [
    ([(img, {'name': 'foo'})], {'name': 'bar'}),
    ([(img, {'blending': 'additive'}), (img,)], {'blending': 'translucent'}),
]


@pytest.mark.parametrize("layer_data, kwargs", plugin_returns)
def test_add_layers_with_plugins_and_kwargs(layer_data, kwargs):
    """Test that _add_layers_with_plugins kwargs override plugin kwargs.

    see also: napari.components._test.test_prune_kwargs
    """
    with patch(
        "napari.plugins.io.read_data_with_plugins",
        MagicMock(return_value=(layer_data, _testimpl)),
    ):
        v = ViewerModel()
        v._add_layers_with_plugins(['mock_path'], kwargs=kwargs, stack=False)
        expected_source = Source(path='mock_path', reader_plugin='testimpl')
        for layer in v.layers:
            for key, val in kwargs.items():
                assert getattr(layer, key) == val
                # if plugins don't provide "name", it falls back to path name
                if 'name' not in kwargs:
                    assert layer.name.startswith('mock_path')
            assert layer.source == expected_source


def test_add_points_layer_with_different_range_updates_all_slices():
    """See https://github.com/napari/napari/pull/4819"""
    viewer = ViewerModel()

    # Adding the first point should show the point
    initial_point = viewer.add_points([[10, 5, 5]])
    np.testing.assert_array_equal(initial_point._indices_view, [0])
    assert viewer.dims.point == (10, 5, 5)

    # Adding an earlier point should keep the dim slider at the position
    # and therefore should not change the viewport.
    earlier_point = viewer.add_points([[8, 1, 1]])
    np.testing.assert_array_equal(initial_point._indices_view, [0])
    np.testing.assert_array_equal(earlier_point._indices_view, [])
    assert viewer.dims.point == (10, 5, 5)

    # Adding a point on the same slice as the initial point should keep the
    # dim slider at the position and should additionally show the added point
    # in the viewport.
    same_slice_as_initial_point = viewer.add_points([[10, 1, 1]])
    np.testing.assert_array_equal(initial_point._indices_view, [0])
    np.testing.assert_array_equal(earlier_point._indices_view, [])
    np.testing.assert_array_equal(
        same_slice_as_initial_point._indices_view, [0]
    )
    assert viewer.dims.point == (10, 5, 5)

    # Adding a later point should keep the dim slider at the position
    # and therefore should not change the viewport.
    later_point = viewer.add_points([[14, 1, 1]])
    np.testing.assert_array_equal(initial_point._indices_view, [0])
    np.testing.assert_array_equal(earlier_point._indices_view, [])
    np.testing.assert_array_equal(
        same_slice_as_initial_point._indices_view, [0]
    )
    np.testing.assert_array_equal(later_point._indices_view, [])
    assert viewer.dims.point == (10, 5, 5)

    # Removing the earlier point should keep the dim slider at the position
    # and therefore should not change the viewport.
    viewer.layers.remove(earlier_point)
    np.testing.assert_array_equal(initial_point._indices_view, [0])
    np.testing.assert_array_equal(
        same_slice_as_initial_point._indices_view, [0]
    )
    np.testing.assert_array_equal(later_point._indices_view, [])
    assert viewer.dims.point == (10, 5, 5)

    # Removing the point on the same slice as the initial point should keep
    # the dim slider at the position and should additionally remove the added
    # point from the viewport.
    viewer.layers.remove(same_slice_as_initial_point)
    np.testing.assert_array_equal(initial_point._indices_view, [0])
    np.testing.assert_array_equal(later_point._indices_view, [])
    assert viewer.dims.point == (10, 5, 5)

    # Removing the initial point should move the dim slider to the later
    # position and update the viewport.
    viewer.layers.remove(initial_point)
    np.testing.assert_array_equal(later_point._indices_view, [0])
    assert viewer.dims.point == (14, 1, 1)

    # Adding an earlier point should keep the dim slider at the position
    # and therefore should not change the viewport.
    earlier_point2 = viewer.add_points([[8, 0, 0]])
    np.testing.assert_array_equal(initial_point._indices_view, [0])
    np.testing.assert_array_equal(earlier_point._indices_view, [])
    assert viewer.dims.point == (14, 1, 1)

    # Removing the second earlier point should move the dim slider to the
    # later position and update the viewport.
    viewer.layers.remove(later_point)
    np.testing.assert_array_equal(earlier_point2._indices_view, [0])
    assert viewer.dims.point == (8, 0, 0)

    # Removing all points should reset the viewport.
    viewer.layers.remove(earlier_point2)
    assert viewer.dims.point == (0, 0)


@pytest.mark.xfail(reason="https://github.com/napari/napari/issues/6198")
def test_last_point_is_visible_in_viewport():
    viewer = ViewerModel()

    # Removing the last point while viewing it should cause
    # us to view the first point due to the layer's new extent.
    points = viewer.add_points([[0, 1, 1], [1, 2, 2]])
    viewer.dims.set_point(0, 1)
    assert viewer.dims.point[0] == 1
    np.testing.assert_array_equal(points._indices_view, [1])

    points.data = [[0, 1, 1]]

    assert viewer.dims.point[0] == 0
    np.testing.assert_array_equal(points._indices_view, [0])
    viewer.layers.remove(points)

    # Removing the first point while viewing it should cause us
    # to view the last point due to the layer's new extent.
    points = viewer.add_points([[0, 1, 1], [1, 2, 2]])
    viewer.dims.set_point(0, 0)
    assert viewer.dims.point[0] == 0
    np.testing.assert_array_equal(points._indices_view, [0])

    points.data = [[1, 2, 2]]

    assert viewer.dims.point[0] == 1
    np.testing.assert_array_equal(points._indices_view, [0])


@pytest.mark.xfail(reason="https://github.com/napari/napari/issues/6199")
def test_dimension_change_is_visible_in_viewport():
    viewer = ViewerModel()

    # Adding a 4d point leads to a visible 4d point with dims.point
    # having the same values.
    point_4d = viewer.add_points([[0] * 4])
    assert viewer.dims.point == tuple([0] * 4)
    np.testing.assert_array_equal(point_4d._indices_view, [0])

    # Adding a 5d point with different 4d coordinates does not change the viewport.
    # Only the first (actual 5th) dimension of the dims.point should change.
    point_5d = viewer.add_points([[2] * 5])
    assert viewer.dims.point == tuple([2] + [0] * 4)
    np.testing.assert_array_equal(point_4d._indices_view, [0])
    np.testing.assert_array_equal(point_5d._indices_view, [])

    # Removing the 4d point leads to an update of the viewport and dims.
    viewer.layers.remove(point_4d)
    assert viewer.dims.point == tuple([2] * 5)
    np.testing.assert_array_equal(point_5d._indices_view, [0])

    # Adding another 4d point does not lead to an update of the viewport
    # because the current dims.point is still in the unified extent.
    point_4d = viewer.add_points([[0] * 4])
    assert viewer.dims.point == tuple([2] * 5)
    np.testing.assert_array_equal(point_4d._indices_view, [])
    np.testing.assert_array_equal(point_5d._indices_view, [0])

    # Removing the 5d point leads to an update of the viewport and dims.
    viewer.layers.remove(point_5d)
    assert viewer.dims.point == tuple([0] * 4)
    np.testing.assert_array_equal(point_4d._indices_view, [0])
