from concurrent.futures import ThreadPoolExecutor
from threading import current_thread, main_thread

import numpy as np
import pytest
import xarray as xr

from napari.components import Dims
from napari.layers import Image, Labels
from napari.layers._scalar_field.scalar_field import ScalarFieldBase
from napari.settings import get_settings
from napari.utils._test_utils import (
    validate_all_params_in_docstring,
    validate_docstring_parent_class_consistency,
    validate_kwargs_sorted,
)


@pytest.fixture
def _enable_async(_fresh_settings, monkeypatch):
    monkeypatch.setattr(get_settings().experimental, 'async_', True)


def test_docstring():
    validate_all_params_in_docstring(ScalarFieldBase)
    validate_kwargs_sorted(ScalarFieldBase)
    validate_docstring_parent_class_consistency(
        ScalarFieldBase, skip=('data', 'ndim', 'multiscale')
    )


def test_single_scale_request_keeps_data_after_replacement():
    data = np.zeros((32, 32))
    layer = Image(data)
    request = layer._slicing_state._make_slice_request(Dims(ndim=2))

    layer.data = np.ones_like(data)

    response = request()
    np.testing.assert_array_equal(response.image.raw, data)
    assert response.thumbnail is response.image


class _RecordingArray:
    """Eager store that records indices and the threads of all reads.

    Like zarr or dask, indexing returns a copy, not a view of the store.
    """

    def __init__(self, array):
        self._array = array
        self.keys = []
        self.read_threads = []
        # LayerDataProtocol requires these as concrete attributes
        self.shape, self.dtype = array.shape, array.dtype
        self.ndim, self.size = array.ndim, array.size

    def __getitem__(self, key):
        self.keys.append(key)
        self.read_threads.append(current_thread())
        return np.array(self._array[key])

    def __array__(self, dtype=None, copy=None):
        self.read_threads.append(current_thread())
        array = np.asarray(self._array, dtype=dtype)
        return array.copy() if copy else array


def test_multiscale_slice_reads_only_the_visible_tile():
    """An eager store must read the visible tile with a single index.

    Indexing the displayed and non-displayed dims separately made eager
    stores materialize every plane before the crop was applied (#9515).
    """
    levels = [
        _RecordingArray(np.zeros((2, 3, 32 // 2**i, 32 // 2**i), np.uint8))
        for i in range(2)
    ]
    layer = Image(levels, multiscale=True)
    # set the private attribute: the public setter refreshes, which slices
    layer._data_level = 0
    corner_pixels = np.zeros((2, 4), dtype=int)
    corner_pixels[0, 2:] = 8
    corner_pixels[1, 2:] = 15
    layer.corner_pixels = corner_pixels

    dims = Dims(
        ndim=4,
        ndisplay=2,
        range=tuple((0, s - 1, 1) for s in levels[0].shape),
    )
    dims.set_point(0, 1)
    dims.set_point(1, 2)

    levels[0].keys.clear()  # drop reads made while constructing the layer
    response = layer._slicing_state._make_slice_request(dims)()

    assert response.image.view.shape == (8, 8)
    # one read: the two sliced dims and the tile crop in a single index
    assert levels[0].keys == [
        (slice(1, 2), slice(2, 3), slice(8, 16, 1), slice(8, 16, 1))
    ]


@pytest.mark.usefixtures('_enable_async')
@pytest.mark.parametrize('prefix', [(), (3,)], ids=['2d', 'nd'])
def test_thumbnail_read_deferred_to_slice_execution(prefix):
    data = [
        _RecordingArray(np.zeros(prefix + (64, 64), dtype=np.uint8)),
        _RecordingArray(np.ones(prefix + (32, 32), dtype=np.uint8)),
    ]
    layer = Image(data, multiscale=True)
    dims = Dims(ndim=layer.ndim)
    request = layer._slicing_state._make_slice_request(dims)

    assert not data[-1].read_threads

    with ThreadPoolExecutor(max_workers=1) as executor:
        response = executor.submit(request).result(timeout=5)
        next_request = layer._slicing_state._make_slice_request(dims)
        executor.submit(next_request).result(timeout=5)

    assert len(data[-1].read_threads) == 1
    assert data[-1].read_threads[0] is not main_thread()
    assert not data[0].read_threads
    np.testing.assert_array_equal(
        response.image.raw, data[-1]._array[(0,) * len(prefix)]
    )


@pytest.mark.usefixtures('_enable_async')
@pytest.mark.parametrize(
    ('shape', 'rgb', 'ndisplay', 'order', 'projection_mode', 'expected'),
    [
        ((32, 40), False, 2, (0, 1), 'none', lambda a: a),
        ((32, 40, 3), True, 2, (0, 1), 'none', lambda a: a),
        ((5, 32, 40), False, 2, (0, 2, 1), 'none', lambda a: a[2].T),
        # point 2 with margins of 1 projects planes 1 to 3
        ((5, 32, 40), False, 2, (0, 1, 2), 'max', lambda a: a[1:4].max(0)),
        ((5, 32, 40), False, 3, (0, 1, 2), 'none', lambda a: a),
    ],
    ids=['2d', 'rgb', 'nd-transposed', 'nd-thick-max', 'volume'],
)
@pytest.mark.parametrize('data_level', [0, 1])
def test_full_extent_image_reused_as_thumbnail(
    shape, rgb, ndisplay, order, projection_mode, expected, data_level
):
    """A tile covering its whole level is read once and is the thumbnail."""
    # only the last two spatial dims are downsampled, so the levels share
    # the dims point
    spatial = len(shape) - rgb
    fine_shape = (
        shape[: spatial - 2]
        + tuple(2 * s for s in shape[spatial - 2 : spatial])
        + shape[spatial:]
    )
    rng = np.random.default_rng(0)
    data = [
        _RecordingArray(rng.integers(0, 256, s, dtype=np.uint8))
        for s in (fine_shape, shape)
    ]
    layer = Image(
        data, multiscale=True, rgb=rgb, projection_mode=projection_mode
    )
    # Locking selects the level in 3D display too.
    layer._locked_data_level = data_level
    ndim = len(order)
    dims = Dims(
        ndim=ndim,
        ndisplay=ndisplay,
        order=order,
        range=tuple((0, s - 1, 1) for s in fine_shape[:ndim]),
    )
    if ndim == 3:
        dims.set_point(0, 2)
        dims.margin_left = dims.margin_right = (1, 0, 0)
    displayed = list(dims.displayed)
    layer.corner_pixels = np.zeros((2, layer.ndim), dtype=int)
    # 3D bounds intentionally remain zero: volume display ignores them.
    if ndisplay == 2:
        layer.corner_pixels[1, displayed] = (
            layer.level_shapes[data_level, displayed] - 1
        )
    response = layer._slicing_state._make_slice_request(dims)()

    source, other = data[data_level], data[1 - data_level]
    assert response.thumbnail is response.image
    np.testing.assert_array_equal(response.image.raw, expected(source._array))
    assert len(source.read_threads) == 1
    assert not other.read_threads


@pytest.mark.parametrize(
    'order', [(0, 1), (0, 2, 1)], ids=['2d', 'nd-transposed']
)
@pytest.mark.parametrize('data_level', [0, 1])
def test_cropped_multiscale_image_keeps_full_thumbnail(order, data_level):
    ndim = len(order)
    prefix = (3,) * (ndim - 2)
    plane = (1,) * (ndim - 2)
    data = [
        np.arange(np.prod(prefix + s), dtype=np.uint16).reshape(prefix + s)
        for s in ((64, 80), (32, 40))
    ]
    layer = Image(data, multiscale=True)
    layer._data_level = data_level
    layer.corner_pixels = np.zeros((2, ndim), dtype=int)
    layer.corner_pixels[0, -2:] = (4, 6)
    layer.corner_pixels[1, -2:] = (11, 19)
    dims = Dims(ndim=ndim, order=order, point=plane + (0, 0))
    response = layer._slicing_state._make_slice_request(dims)()
    expected_image = data[data_level][plane][4:12, 6:20]
    expected_thumbnail = data[-1][plane]
    if order == (0, 2, 1):  # the displayed dims are transposed
        expected_image = expected_image.T
        expected_thumbnail = expected_thumbnail.T

    assert response.thumbnail is not response.image
    np.testing.assert_array_equal(response.image.raw, expected_image)
    np.testing.assert_array_equal(response.thumbnail.raw, expected_thumbnail)


@pytest.mark.parametrize('ndim', [2, 3])
def test_multiscale_request_keeps_data_after_replacement(ndim):
    prefix = (3,) if ndim == 3 else ()
    data1 = [np.zeros(prefix + (64, 64)), np.ones(prefix + (32, 32))]
    data2 = [np.zeros(prefix + (64, 64)), np.full(prefix + (32, 32), 2)]
    layer = Image(data1, multiscale=True)
    dims = Dims(ndim=ndim)
    old_request = layer._slicing_state._make_slice_request(dims)

    layer.data = data2
    new_request = layer._slicing_state._make_slice_request(dims)

    expected_old = data1[-1][0] if ndim == 3 else data1[-1]
    expected_new = data2[-1][0] if ndim == 3 else data2[-1]
    np.testing.assert_array_equal(old_request().image.raw, expected_old)
    np.testing.assert_array_equal(new_request().image.raw, expected_new)


def test_thumbnail_level_refreshed_on_data_replacement():
    """Replacing layer.data with a different number of levels moves the
    thumbnail to the new coarsest level."""
    data1 = [np.zeros((64, 64)), np.zeros((32, 32))]
    data2 = [np.ones((90, 90)), np.ones((45, 45)), np.ones((22, 22))]

    layer = Image(data1, multiscale=True)
    assert layer._thumbnail_level == 1

    layer.data = data2
    assert layer._thumbnail_level == 2  # new last level
    request = layer._slicing_state._make_slice_request(Dims(ndim=2))
    np.testing.assert_array_equal(request._thumbnail_plane(), data2[2])


def test_3d_display_does_not_cache_volume():
    """Volumes are never cached: the thumbnail cache only holds 2D planes."""
    levels = [
        np.zeros((8, 64, 64), np.uint8),
        _RecordingArray(np.ones((4, 32, 32), np.uint8)),
    ]
    layer = Image(levels, multiscale=True)
    dims = Dims(ndim=3, ndisplay=3)
    levels[1].read_threads.clear()  # drop reads made while constructing

    for _ in range(2):
        response = layer._slicing_state._make_slice_request(dims)()
        assert response.thumbnail is response.image

    assert len(levels[1].read_threads) == 2


@pytest.mark.parametrize('prefix', [(), (1,)], ids=['2d', 'singleton'])
def test_3d_display_caches_single_plane(prefix):
    """Planes shown in 3D (2D data, singleton dims) are cached, so slider
    moves of other layers don't re-read them."""
    levels = [
        np.zeros(prefix + (64, 64), np.uint8),
        _RecordingArray(np.ones(prefix + (32, 32), np.uint8)),
    ]
    layer = Image(levels, multiscale=True)
    dims = Dims(ndim=3, ndisplay=3)
    # drop the cache and reads from construction, which slices in 2D
    layer._reset_thumbnail_level_data()
    levels[1].read_threads.clear()

    for _ in range(2):
        response = layer._slicing_state._make_slice_request(dims)()
        np.testing.assert_array_equal(response.image.raw, levels[1]._array)

    assert len(levels[1].read_threads) == 1


@pytest.mark.parametrize(
    'prefix', [(), (1, 1, 1), (3,)], ids=['2d', 'singletons', 'nd']
)
def test_cropped_thumbnail_level_read_once(prefix):
    """Panning a cropped tile must not re-read the thumbnail level."""
    levels = [
        _RecordingArray(np.ones(prefix + (128, 128), np.uint8)),
        _RecordingArray(np.ones(prefix + (64, 64), np.uint8)),
    ]
    layer = Image(levels, multiscale=True)
    layer._data_level = 0
    ndim = layer.ndim
    dims = Dims(
        ndim=ndim,
        ndisplay=2,
        range=tuple((0, s - 1, 1) for s in levels[0].shape),
    )

    for start in (0, 16, 32):
        corner_pixels = np.zeros((2, ndim), dtype=int)
        corner_pixels[0, -2:] = start
        corner_pixels[1, -2:] = start + 31
        layer.corner_pixels = corner_pixels
        response = layer._slicing_state._make_slice_request(dims)()
        assert response.thumbnail is not response.image
        assert response.thumbnail.view.shape == (64, 64)

    # one read in total, none per pan
    assert len(levels[1].read_threads) == 1


# 'none' keys on ints for the not-displayed dims, 'max' on slices
@pytest.mark.parametrize('projection_mode', ['none', 'max'])
def test_thumbnail_plane_reread_on_dims_change(projection_mode):
    """Moving the dims point reads the new thumbnail plane."""
    levels = [
        np.arange(3 * 64 * 64, dtype=np.uint16).reshape(3, 64, 64),
        _RecordingArray(
            np.arange(3 * 32 * 32, dtype=np.uint16).reshape(3, 32, 32)
        ),
    ]
    layer = Image(levels, multiscale=True, projection_mode=projection_mode)
    # a cropped level-0 tile, so the thumbnail is read separately
    layer._data_level = 0
    corner_pixels = np.zeros((2, 3), dtype=int)
    corner_pixels[1, 1:] = 31
    layer.corner_pixels = corner_pixels
    dims = Dims(ndim=3, range=((0, 2, 1), (0, 63, 1), (0, 63, 1)))

    thumbnails = []
    for point in (0, 0, 2):
        dims.set_point(0, point)
        response = layer._slicing_state._make_slice_request(dims)()
        thumbnails.append(response.thumbnail.raw)

    assert len(levels[1].read_threads) == 2
    np.testing.assert_array_equal(thumbnails[0], levels[1]._array[0])
    np.testing.assert_array_equal(thumbnails[2], levels[1]._array[2])


@pytest.mark.parametrize('Layer', [Image, Labels])
def test_data_setter_updates_transforms(Layer):
    """Replacing data with different ndim should expand transforms."""
    data_2d = np.zeros((10, 10), dtype=np.uint8)
    layer = Layer(data_2d, scale=(2, 3))
    assert layer.ndim == 2
    assert len(layer.scale) == 2

    data_3d = np.zeros((5, 10, 10), dtype=np.uint8)
    layer.data = data_3d
    assert layer.ndim == 3
    assert len(layer.scale) == 3


@pytest.mark.parametrize('Layer', [Image, Labels])
def test_invisible_layer_keeps_placeholder_rank_in_3d(Layer):
    """When ndisplay flips while the layer is invisible, the cached
    placeholder must still match ndisplay's rank, otherwise vispy crashes
    with "Volume visual needs a 3D array." when it later reads
    layer._data_view.
    """
    layer = Layer(np.zeros((4, 5, 6), dtype=np.uint8), visible=False)
    layer._slice_dims(Dims(ndim=3, ndisplay=3))
    assert layer._data_view.ndim == 3
    layer.visible = True
    layer.visible = False
    assert layer._data_view.ndim == 3


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

        # Out-of-range values are silently ignored (e.g. for linked layers
        # where one layer may have fewer levels than another).
        layer.locked_data_level = 1
        layer.locked_data_level = len(data)
        assert layer.locked_data_level == 1  # unchanged

        layer.locked_data_level = -1
        assert layer.locked_data_level == 1  # unchanged

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


def test_locked_data_level_constructor():
    """locked_data_level can be set via the Image constructor."""
    data = _make_multiscale_3d()
    layer = Image(data, multiscale=True, locked_data_level=2)
    assert layer.locked_data_level == 2

    # Default (None) should also be safe
    layer2 = Image(data, multiscale=True)
    assert layer2.locked_data_level is None


def _draw_layer(layer, shape_threshold=(800, 600)):
    """Call ``_update_draw`` with a viewport that sees the full data extent."""
    displayed = layer._slice_input.displayed
    corner_pixels = layer._extent_data[:, displayed]
    layer._update_draw(
        scale_factor=1,
        corner_pixels_displayed=corner_pixels,
        shape_threshold=shape_threshold,
    )


@pytest.mark.parametrize('Layer', [Image, Labels])
class TestLockedDataLevelDraw:
    """Tests for locked_data_level interacting with the draw cycle.

    These call ``_update_draw`` and ``_slice_dims`` directly instead of
    going through a full Qt viewer, so they run in headless CI
    environments without Qt bindings.
    """

    def test_lock_overrides_auto_2d(self, Layer):
        """Locking to level 0 keeps full resolution even when the
        viewport is wide enough that auto would pick a coarser level."""
        data = _make_multiscale_2d()
        layer = Layer(data, multiscale=True)

        _draw_layer(layer)
        auto_level = layer.data_level
        assert auto_level > 0, 'sanity: auto should pick a coarser level'

        layer.locked_data_level = 0
        _draw_layer(layer)

        assert layer.data_level == 0

    def test_lock_overrides_auto_3d(self, Layer):
        """In 3D, auto defaults to the coarsest level; locking to 0
        should override that."""
        data = _make_multiscale_3d()
        layer = Layer(data, multiscale=True)

        layer._slice_dims(Dims(ndim=3, ndisplay=3))
        _draw_layer(layer)
        assert layer.data_level == len(data) - 1, (
            'sanity: 3D auto should pick coarsest'
        )

        layer.locked_data_level = 0
        _draw_layer(layer)

        assert layer.data_level == 0

    def test_unlock_restores_auto_2d(self, Layer):
        """Setting locked_data_level back to None restores automatic
        level selection in 2D."""
        data = _make_multiscale_2d()
        layer = Layer(data, multiscale=True)

        _draw_layer(layer)
        auto_level = layer.data_level

        layer.locked_data_level = 0
        _draw_layer(layer)
        assert layer.data_level == 0

        layer.locked_data_level = None
        _draw_layer(layer)
        assert layer.data_level == auto_level

    def test_unlock_restores_auto_3d(self, Layer):
        """Setting locked_data_level back to None restores coarsest-level
        default in 3D."""
        data = _make_multiscale_3d()
        layer = Layer(data, multiscale=True)

        layer._slice_dims(Dims(ndim=3, ndisplay=3))
        layer.locked_data_level = 0
        _draw_layer(layer)
        assert layer.data_level == 0

        layer.locked_data_level = None
        _draw_layer(layer)
        assert layer.data_level == len(data) - 1

    def test_ndisplay_transitions(self, Layer):
        """Lock and auto behaviour across 2D/3D switches."""
        data = _make_multiscale_3d()
        layer = Layer(data, multiscale=True)
        coarsest = len(data) - 1

        # -- auto 2D->3D: should pick coarsest
        _draw_layer(layer)
        layer._slice_dims(Dims(ndim=3, ndisplay=3))
        _draw_layer(layer)
        assert layer.data_level == coarsest

        # -- auto 3D->2D: should leave coarsest-level default behind
        layer._slice_dims(Dims(ndim=3, ndisplay=2))
        _draw_layer(layer)
        assert layer.data_level != coarsest

        # -- locked 2D->3D: lock should persist
        layer.locked_data_level = 0
        _draw_layer(layer)
        assert layer.data_level == 0

        layer._slice_dims(Dims(ndim=3, ndisplay=3))
        _draw_layer(layer)
        assert layer.data_level == 0

        # -- locked 3D->2D: lock should persist
        layer._slice_dims(Dims(ndim=3, ndisplay=2))
        _draw_layer(layer)
        assert layer.data_level == 0

    def test_locked_level_refreshes_on_stale_data_level(self, Layer):
        """_update_draw refreshes when _data_level drifts from locked."""
        data = _make_multiscale_3d()
        layer = Layer(data, multiscale=True)

        layer._slice_dims(Dims(ndim=3, ndisplay=3))
        layer.locked_data_level = 0
        _draw_layer(layer)
        assert layer.data_level == 0

        # Simulate _data_level drifting out of sync (e.g. internal reset)
        layer._data_level = len(data) - 1
        _draw_layer(layer)
        # _update_draw should have corrected it back to the locked level
        assert layer.data_level == 0

    def test_locked_level_loads_correct_data_3d(self, Layer):
        """Verify the actual pixel values match the locked level."""
        data = _make_multiscale_3d()
        layer = Layer(data, multiscale=True)

        layer._slice_dims(Dims(ndim=3, ndisplay=3))
        _draw_layer(layer)

        for level_idx in range(len(data)):
            expected_value = (level_idx + 1) * 10
            layer.locked_data_level = level_idx
            layer._slice_dims(Dims(ndim=3, ndisplay=3))
            view = layer._data_view
            assert np.all(view == expected_value), (
                f'Level {level_idx}: expected all {expected_value}, '
                f'got unique values {np.unique(view)}'
            )


def test_set_view_slice_3d_multiscale_corners_without_viewer():
    """Switching a multiscale layer to ndisplay=3 without a viewer
    (no _update_draw) must set corner_pixels and _data_level to the
    coarsest level so slicing produces correct 3D data.

    Regression test: without the fix, corner_pixels stayed at the 2D
    extent and _data_level was not updated, leading to wrong or empty
    slices.
    """
    data = [
        np.full((8, 40, 20), 10, dtype=np.uint8),
        np.full((4, 20, 10), 20, dtype=np.uint8),
        np.full((2, 10, 5), 30, dtype=np.uint8),
    ]
    layer = Image(data, multiscale=True)

    dims_3d = Dims(ndim=3, ndisplay=3)
    layer._slice_dims(dims_3d)

    coarsest = len(data) - 1
    assert layer.data_level == coarsest, (
        f'Expected data_level={coarsest}, got {layer.data_level}'
    )

    expected_shape = np.array(data[coarsest].shape)
    corners = layer.corner_pixels
    actual_extent = corners[1, :] - corners[0, :] + 1
    np.testing.assert_array_equal(
        actual_extent,
        expected_shape,
        err_msg=(
            'corner_pixels should span the full coarsest level '
            'when switching to 3D without a viewer'
        ),
    )


class TestXarrayMetadataInit:
    def test_xarray_image_all_metadata(self):
        """Image inherits axis_labels, scale, translate, units from DataArray."""
        data = xr.DataArray(
            np.random.random((3, 10, 15)),
            dims=['z', 'y', 'x'],
            coords={
                'z': ('z', [10, 15, 20], {'units': 'microns'}),
                'y': ('y', np.arange(100, 110).astype(float), {'units': 'mm'}),
                'x': ('x', np.arange(200, 215).astype(float)),
            },
        )
        layer = Image(data)
        assert layer.axis_labels == ('z', 'y', 'x')
        np.testing.assert_allclose(layer.scale, [5.0, 1.0, 1.0])
        np.testing.assert_allclose(layer.translate, [10.0, 100.0, 200.0])
        assert str(layer.units[0]) == 'micron'
        assert str(layer.units[1]) == 'millimeter'
        assert str(layer.units[2]) == 'pixel'

    @pytest.mark.parametrize(
        ('rgb', 'expected_ndim', 'expected_axis_labels'),
        [
            (None, 2, ('y', 'x')),
            (True, 2, ('y', 'x')),
            (False, 3, ('y', 'x', 'channel')),
        ],
    )
    def test_xarray_rgb_metadata(
        self, rgb, expected_ndim, expected_axis_labels
    ):
        """RGB channel dimensions are excluded only when rgb is enabled."""
        data = xr.DataArray(
            np.zeros((64, 64, 3)),
            dims=['y', 'x', 'channel'],
            coords={
                'y': np.arange(64),
                'x': np.arange(64),
                'channel': np.arange(3),
            },
        )
        layer = Image(data, rgb=rgb)
        assert layer.ndim == expected_ndim
        assert layer.axis_labels == expected_axis_labels
        assert len(layer.scale) == expected_ndim
        assert len(layer.translate) == expected_ndim
        assert len(layer.units) == expected_ndim

    def test_xarray_labels_gets_metadata(self):
        """Labels inherits axis_labels from DataArray dims (no coords)."""
        data = xr.DataArray(
            np.random.randint(0, 5, (5, 10, 15)), dims=['z', 'y', 'x']
        )
        layer = Labels(data)
        assert layer.axis_labels == ('z', 'y', 'x')

    def test_xarray_explicit_overrides(self):
        """Explicit axis_labels, scale, units override auto-inheritance."""
        data = xr.DataArray(
            np.random.random((10, 15)),
            dims=['y', 'x'],
            coords={
                'y': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
                'x': list(range(15)),
            },
        )
        layer = Image(
            data,
            axis_labels=('row', 'col'),
            scale=(3.0, 3.0),
            translate=(5.0, 5.0),
            units=('mm', 'mm'),
        )
        assert layer.axis_labels == ('row', 'col')
        np.testing.assert_allclose(layer.scale, [3.0, 3.0])
        np.testing.assert_allclose(layer.translate, [5.0, 5.0])
        assert str(layer.units[0]) == 'millimeter'

    def test_xarray_multiscale_list(self):
        """List of xarrays (multiscale) inherits metadata from first level."""
        da_full = xr.DataArray(
            np.random.random((10, 20)),
            dims=['y', 'x'],
            coords={
                'y': (
                    'y',
                    np.arange(100, 110).astype(float),
                    {'units': 'microns'},
                ),
                'x': ('x', np.arange(200, 220).astype(float)),
            },
        )
        da_half = xr.DataArray(np.random.random((5, 10)), dims=['y', 'x'])
        layer = Image([da_full, da_half], multiscale=True)
        assert layer.axis_labels == ('y', 'x')
        np.testing.assert_allclose(layer.scale, [1.0, 1.0])
        np.testing.assert_allclose(layer.translate, [100.0, 200.0])
        assert str(layer.units[0]) == 'micron'

    def test_xarray_datetime_coords(self):
        """Datetime coords get a real time unit, scale and translate."""
        data = xr.DataArray(
            np.random.random((3, 10, 15)),
            dims=['time', 'y', 'x'],
            coords={
                'time': np.array(
                    ['2013-01-01', '2013-01-02', '2013-01-03'],
                    dtype='datetime64[ns]',
                ),
                'y': ('y', np.arange(100, 110).astype(float)),
                'x': ('x', np.arange(200, 215).astype(float)),
            },
        )
        layer = Image(data)
        assert layer.axis_labels == ('time', 'y', 'x')
        np.testing.assert_allclose(layer.scale, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(layer.translate, [15706.0, 100.0, 200.0])
        assert str(layer.units[0]) == 'day'
