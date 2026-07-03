"""Tests for settings-driven auto attachment of progressive loading."""

import dask.array as da
import numpy as np
import pytest

from napari.experimental._tests.test_progressive_loading import (
    _wait_for_idle_loader,
)
from napari.layers import Labels
from napari.settings import get_settings


@pytest.fixture
def multiscale_dask():
    """A small chunked multiscale pyramid (eligible for auto attach)."""
    base = np.random.default_rng(0).integers(
        1, 255, size=(256, 256), dtype=np.uint8
    )
    levels = [base, base[::2, ::2].copy(), base[::4, ::4].copy()]
    return [da.from_array(level, chunks=(32, 32)) for level in levels]


@pytest.fixture
def progressive_setting():
    # settings are reset per test by the _fresh_settings autouse fixture
    get_settings().experimental.progressive_loading = True


def test_no_attach_by_default(qtbot, make_napari_viewer, multiscale_dask):
    """With the setting off (default), multiscale layers are untouched."""
    viewer = make_napari_viewer()
    layer = viewer.add_image(
        multiscale_dask, multiscale=True, contrast_limits=(0, 255)
    )
    qtbot.wait(100)  # allow any (wrongly) deferred replacement to run
    assert viewer.layers[0] is layer
    assert 'progressive_loader' not in layer.metadata


def test_attach_when_enabled(
    qtbot, make_napari_viewer, multiscale_dask, progressive_setting
):
    """With the setting on, the layer is replaced by a progressive one."""
    viewer = make_napari_viewer()
    original = viewer.add_image(
        multiscale_dask,
        multiscale=True,
        contrast_limits=(0, 255),
        name='pyramid',
        opacity=0.5,
    )
    qtbot.waitUntil(
        lambda: (
            len(viewer.layers) == 1
            and 'progressive_loader' in viewer.layers[0].metadata
        ),
        timeout=10000,
    )
    layer = viewer.layers[0]
    assert layer is not original
    # appearance carried over from the original layer
    assert layer.name == 'pyramid'
    assert layer.opacity == 0.5
    assert layer.contrast_limits == [0, 255]
    _wait_for_idle_loader(qtbot, layer.metadata['progressive_loader'])


def test_attach_preserves_layer_position(
    qtbot, make_napari_viewer, multiscale_dask, progressive_setting
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((8, 8), dtype=np.uint8), name='plain')
    viewer.add_image(
        multiscale_dask,
        multiscale=True,
        contrast_limits=(0, 255),
        name='pyramid',
    )
    viewer.add_image(np.ones((8, 8), dtype=np.uint8), name='plain2')
    qtbot.waitUntil(
        lambda: 'progressive_loader' in viewer.layers['pyramid'].metadata,
        timeout=10000,
    )
    assert [layer.name for layer in viewer.layers] == [
        'plain',
        'pyramid',
        'plain2',
    ]
    _wait_for_idle_loader(
        qtbot, viewer.layers['pyramid'].metadata['progressive_loader']
    )


def test_labels_attach_when_enabled(
    qtbot, make_napari_viewer, progressive_setting
):
    base = np.random.default_rng(0).integers(
        0, 5, size=(128, 128), dtype=np.uint32
    )
    levels = [
        da.from_array(base, chunks=(32, 32)),
        da.from_array(base[::2, ::2].copy(), chunks=(32, 32)),
    ]
    viewer = make_napari_viewer()
    viewer.add_labels(levels, name='segments')
    qtbot.waitUntil(
        lambda: (
            len(viewer.layers) == 1
            and 'progressive_loader' in viewer.layers[0].metadata
        ),
        timeout=10000,
    )
    layer = viewer.layers[0]
    assert isinstance(layer, Labels)
    assert layer.name == 'segments'
    _wait_for_idle_loader(qtbot, layer.metadata['progressive_loader'])


def test_plain_numpy_multiscale_untouched(
    qtbot, make_napari_viewer, progressive_setting
):
    """In-memory pyramids don't benefit from streaming: left alone."""
    viewer = make_napari_viewer()
    base = np.zeros((64, 64), dtype=np.uint8)
    layer = viewer.add_image([base, base[::2, ::2]], multiscale=True)
    qtbot.wait(100)
    assert viewer.layers[0] is layer
    assert 'progressive_loader' not in layer.metadata


def test_non_multiscale_untouched(
    qtbot, make_napari_viewer, progressive_setting
):
    viewer = make_napari_viewer()
    layer = viewer.add_image(da.zeros((64, 64), chunks=(32, 32)))
    qtbot.wait(100)
    assert viewer.layers[0] is layer
    assert 'progressive_loader' not in layer.metadata


def test_toggle_off_stops_attaching(
    qtbot, make_napari_viewer, multiscale_dask, progressive_setting
):
    """The setting is read per insertion: turning it off applies to new
    layers without touching already-attached loaders."""
    viewer = make_napari_viewer()
    viewer.add_image(
        multiscale_dask,
        multiscale=True,
        contrast_limits=(0, 255),
        name='streamed',
    )
    qtbot.waitUntil(
        lambda: 'progressive_loader' in viewer.layers['streamed'].metadata,
        timeout=10000,
    )
    get_settings().experimental.progressive_loading = False
    plain = viewer.add_image(
        [level.copy() for level in multiscale_dask],
        multiscale=True,
        contrast_limits=(0, 255),
        name='plain',
    )
    qtbot.wait(100)
    assert 'progressive_loader' not in plain.metadata
    assert 'progressive_loader' in viewer.layers['streamed'].metadata
    _wait_for_idle_loader(
        qtbot, viewer.layers['streamed'].metadata['progressive_loader']
    )
