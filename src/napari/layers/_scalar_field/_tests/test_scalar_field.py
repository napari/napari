import numpy as np
import pytest

from napari.components import Dims
from napari.layers import Image, Labels
from napari.layers._scalar_field.scalar_field import ScalarFieldBase
from napari.utils._test_utils import (
    validate_all_params_in_docstring,
    validate_docstring_parent_class_consistency,
    validate_kwargs_sorted,
)


def test_docstring():
    validate_all_params_in_docstring(ScalarFieldBase)
    validate_kwargs_sorted(ScalarFieldBase)
    validate_docstring_parent_class_consistency(
        ScalarFieldBase, skip=('data', 'ndim', 'multiscale')
    )


def test_multiscale_thumbnail_level_prematerialized():
    """Thumbnail level materializes as numpy via the level materializer."""
    rng = np.random.default_rng(0)
    data = [rng.random((128, 128)), rng.random((64, 64))]
    layer = Image(data, multiscale=True)
    result = layer._level_materializer(layer._thumbnail_level)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, data[layer._thumbnail_level])


def test_3d_multiscale_thumbnail_not_prematerialized():
    """3D multiscale images should not create a level materializer: volumes can be too large."""
    data = [np.random.random((16, 128, 128)), np.random.random((8, 64, 64))]
    layer = Image(data, multiscale=True)
    assert layer._level_materializer is None


def test_single_scale_no_prematerialization():
    """Single-scale images should not materialize and cache."""
    layer = Image(np.random.random((32, 32)))
    assert layer._level_materializer is None


def test_multiscale_slice_request_receives_preselected_sources():
    """Slice requests should receive the preselected sources."""
    data = [np.random.random((64, 64)), np.random.random((32, 32))]
    layer = Image(data, multiscale=True)
    dims = Dims(
        ndim=2,
        ndisplay=2,
        range=tuple((0, s - 1, 1) for s in data[0].shape),
    )
    request = layer._slicing_state._make_slice_request(dims)
    assert layer.data_level == layer._thumbnail_level
    # slice request should re-use thumbnail source when possible
    assert request.data_at_data_level is request.data_at_thumbnail_level
    np.testing.assert_array_equal(
        request.data_at_thumbnail_level, data[layer._thumbnail_level]
    )


def test_multiscale_slice_request_keeps_non_thumbnail_image_source():
    """Non-thumbnail views should keep their own level."""
    data = [np.random.random((64, 64)), np.random.random((32, 32))]
    layer = Image(data, multiscale=True)
    layer.data_level = 0
    dims = Dims(
        ndim=2,
        ndisplay=2,
        range=tuple((0, s - 1, 1) for s in data[0].shape),
    )
    request = layer._slicing_state._make_slice_request(dims)
    assert request.data_at_data_level is data[0]
    np.testing.assert_array_equal(
        request.data_at_thumbnail_level, data[layer._thumbnail_level]
    )


def test_thumbnail_level_data_refreshed_on_data_replacement():
    """Replacing layer.data must bind a fresh materializer for the new data
    so slices never read from the old dataset."""
    data1 = [np.zeros((64, 64)), np.zeros((32, 32))]
    data2 = [np.ones((90, 90)), np.ones((45, 45)), np.ones((22, 22))]

    layer = Image(data1, multiscale=True)
    assert layer._thumbnail_level == 1
    np.testing.assert_array_equal(
        layer._level_materializer(layer._thumbnail_level), data1[1]
    )

    layer.data = data2
    assert layer._thumbnail_level == 2  # new last level
    np.testing.assert_array_equal(
        layer._level_materializer(layer._thumbnail_level), data2[2]
    )


def test_thumbnail_level_data_uses_new_data_ndim_2d_to_3d():
    """Replacing 2D data with 3D should set materializer to None — 3D volumes
    are left as lazy data-array references."""
    from napari.layers._multiscale_data import MultiScaleData

    data2d = [np.zeros((64, 64)), np.zeros((32, 32))]
    data3d = [np.ones((16, 64, 64)), np.ones((8, 32, 32))]

    layer = Image(data2d, multiscale=True)
    assert layer._level_materializer is not None  # 2D: materializer present

    layer._data = MultiScaleData(data3d)
    layer._reset_thumbnail_level_data()

    assert layer._level_materializer is None  # 3D: no materializer


def test_thumbnail_level_data_uses_new_data_ndim_3d_to_2d():
    """Replacing 3D data with 2D should produce a fresh materializer for
    the new 2D data."""
    from napari.layers._multiscale_data import MultiScaleData

    data3d = [np.zeros((16, 64, 64)), np.zeros((8, 32, 32))]
    data2d = [np.ones((64, 64)), np.ones((32, 32))]

    layer = Image(data3d, multiscale=True)
    old_materializer = layer._level_materializer

    layer._data = MultiScaleData(data2d)
    layer._reset_thumbnail_level_data()
    assert layer._level_materializer is not old_materializer
    np.testing.assert_array_equal(
        layer._level_materializer(layer._thumbnail_level),
        data2d[layer._thumbnail_level],
    )


def test_rgb_2d_multiscale_prematerialized():
    """RGB (H, W, 3) multiscale images materialise correctly via the closure."""
    data = [np.ones((128, 128, 3)), np.ones((64, 64, 3))]
    layer = Image(data, multiscale=True, rgb=True)

    assert layer.ndim == 2  # spatial dims only
    tld = layer._level_materializer(layer._thumbnail_level)
    assert isinstance(tld, np.ndarray)
    assert tld.shape == (64, 64, 3)
    np.testing.assert_array_equal(tld, data[1])


# ---------------------------------------------------------------------------
# locked_data_level tests
# ---------------------------------------------------------------------------

_MULTISCALE_SHAPES_2D = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
_MULTISCALE_SHAPES_3D = [(8, 40, 20), (4, 20, 10), (2, 10, 5)]


def _make_multiscale_2d():
    """2D multiscale data where each level has a distinct constant value."""
    return [
        np.full(s, fill_value=(i + 1) * 10, dtype=np.uint8)
        for i, s in enumerate(_MULTISCALE_SHAPES_2D)
    ]


def _make_multiscale_3d():
    """3D multiscale data where each level has a distinct constant value."""
    return [
        np.full(s, fill_value=(i + 1) * 10, dtype=np.uint8)
        for i, s in enumerate(_MULTISCALE_SHAPES_3D)
    ]


@pytest.mark.parametrize('Layer', [Image, Labels])
class TestLockedDataLevel:
    """Tests for the locked_data_level property on ScalarFieldBase."""

    def test_locked_data_level_property(self, Layer):
        """Default is None; setting valid/invalid values works correctly."""
        data = _make_multiscale_3d()
        layer = Layer(data, multiscale=True)

        assert layer.locked_data_level is None

        layer.locked_data_level = 1
        assert layer.locked_data_level == 1

        layer.locked_data_level = None
        assert layer.locked_data_level is None

        with pytest.raises(ValueError, match='locked_data_level'):
            layer.locked_data_level = len(data)
        with pytest.raises(ValueError, match='locked_data_level'):
            layer.locked_data_level = -1

    def test_locked_data_level_reset_on_data_change(self, Layer):
        """Setting .data resets locked_data_level to None (auto)."""
        data = _make_multiscale_3d()
        layer = Layer(data, multiscale=True)

        # Lock to the last level
        layer.locked_data_level = len(data) - 1
        assert layer.locked_data_level == len(data) - 1

        # Replace with data that has fewer levels
        fewer_levels = data[:2]
        layer.data = fewer_levels

        # The old locked level would be out of range; it must be reset
        assert layer.locked_data_level is None


@pytest.mark.parametrize('add_method', ['add_image', 'add_labels'])
class TestLockedDataLevelViewer:
    """Viewer-based tests for locked_data_level.

    These use the real viewer draw cycle instead of manually calling
    _update_draw / _slice_dims, so they exercise the feature the same
    way a user would.  Parametrized over Image and Labels layers.
    """

    @staticmethod
    def _trigger_draw(viewer):
        """Set a canvas size and trigger the draw cycle."""
        viewer.window._qt_viewer.canvas.size = (800, 600)
        viewer.window._qt_viewer.canvas.on_draw(None)

    @staticmethod
    def _add_layer(viewer, data, add_method):
        return getattr(viewer, add_method)(data, multiscale=True)

    def test_lock_overrides_auto_2d(self, make_napari_viewer, add_method):
        """Locking to level 0 keeps full resolution even when the
        viewport is wide enough that auto would pick a coarser level."""
        viewer = make_napari_viewer()
        data = _make_multiscale_2d()
        layer = self._add_layer(viewer, data, add_method)

        self._trigger_draw(viewer)
        auto_level = layer.data_level
        assert auto_level > 0, 'sanity: auto should pick a coarser level'

        layer.locked_data_level = 0
        self._trigger_draw(viewer)

        assert layer.data_level == 0

    def test_lock_overrides_auto_3d(self, make_napari_viewer, add_method):
        """In 3D, auto defaults to the coarsest level; locking to 0
        should override that."""
        viewer = make_napari_viewer()
        data = _make_multiscale_3d()
        layer = self._add_layer(viewer, data, add_method)

        viewer.dims.ndisplay = 3
        self._trigger_draw(viewer)
        assert layer.data_level == len(data) - 1, (
            'sanity: 3D auto should pick coarsest'
        )

        layer.locked_data_level = 0
        self._trigger_draw(viewer)

        assert layer.data_level == 0

    def test_unlock_restores_auto_2d(self, make_napari_viewer, add_method):
        """Setting locked_data_level back to None restores automatic
        level selection in 2D."""
        viewer = make_napari_viewer()
        data = _make_multiscale_2d()
        layer = self._add_layer(viewer, data, add_method)

        self._trigger_draw(viewer)
        auto_level = layer.data_level

        layer.locked_data_level = 0
        self._trigger_draw(viewer)
        assert layer.data_level == 0

        layer.locked_data_level = None
        self._trigger_draw(viewer)
        assert layer.data_level == auto_level

    def test_unlock_restores_auto_3d(self, make_napari_viewer, add_method):
        """Setting locked_data_level back to None restores coarsest-level
        default in 3D."""
        viewer = make_napari_viewer()
        data = _make_multiscale_3d()
        layer = self._add_layer(viewer, data, add_method)

        viewer.dims.ndisplay = 3
        layer.locked_data_level = 0
        self._trigger_draw(viewer)
        assert layer.data_level == 0

        layer.locked_data_level = None
        self._trigger_draw(viewer)
        assert layer.data_level == len(data) - 1

    def test_ndisplay_transitions(self, make_napari_viewer, add_method):
        """Lock and auto behaviour across 2D/3D switches."""
        viewer = make_napari_viewer()
        data = _make_multiscale_3d()
        layer = self._add_layer(viewer, data, add_method)
        coarsest = len(data) - 1

        # -- auto 2D->3D: should pick coarsest
        self._trigger_draw(viewer)
        viewer.dims.ndisplay = 3
        self._trigger_draw(viewer)
        assert layer.data_level == coarsest

        # -- auto 3D->2D: should leave coarsest-level default behind
        viewer.dims.ndisplay = 2
        self._trigger_draw(viewer)
        assert layer.data_level != coarsest

        # -- locked 2D->3D: lock should persist
        layer.locked_data_level = 0
        self._trigger_draw(viewer)
        assert layer.data_level == 0

        viewer.dims.ndisplay = 3
        self._trigger_draw(viewer)
        assert layer.data_level == 0

        # -- locked 3D->2D: lock should persist
        viewer.dims.ndisplay = 2
        self._trigger_draw(viewer)
        assert layer.data_level == 0

    def test_locked_level_loads_correct_data_3d(
        self, make_napari_viewer, add_method
    ):
        """Verify the actual pixel values match the locked level."""
        viewer = make_napari_viewer()
        data = _make_multiscale_3d()
        layer = self._add_layer(viewer, data, add_method)

        viewer.dims.ndisplay = 3
        self._trigger_draw(viewer)

        for level_idx in range(len(data)):
            expected_value = (level_idx + 1) * 10
            layer.locked_data_level = level_idx
            self._trigger_draw(viewer)
            view = layer._data_view
            assert np.all(view == expected_value), (
                f'Level {level_idx}: expected all {expected_value}, '
                f'got unique values {np.unique(view)}'
            )
