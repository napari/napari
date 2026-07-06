"""Tests for the progressive-loading chunk debug overlay."""

import os
import sys

import dask.array as da
import numpy as np
import pytest

pytest.importorskip('qtpy', reason='requires Qt backend')

pytestmark = pytest.mark.skipif(
    sys.platform == 'darwin' and os.environ.get('CI') == 'true',
    reason='Progressive loading tests hang on macOS CI (no real display)',
)

from napari.experimental import _progressive_loading  # noqa: E402
from napari.experimental._debug_overlay import (  # noqa: E402
    STATE_COLORS,
    ChunkDebugOverlay,
)
from napari.experimental._progressive_loading import (  # noqa: E402
    add_progressive_loading_image,
)
from napari.experimental._virtual_data import (  # noqa: E402
    MultiScaleVirtualData,
    chunk_ids_in_region,
)


@pytest.fixture(autouse=True)
def _no_progress_bars(monkeypatch):
    """See test_streaming_invariants: progress bars wedge Qt timers."""
    monkeypatch.setattr(
        _progressive_loading.ProgressiveLoader,
        '_make_progress',
        lambda self, total, description: None,
    )


@pytest.fixture
def multiscale_arrays():
    base = np.random.default_rng(0).integers(
        1,
        255,
        size=(256, 256),
        dtype=np.uint8,
    )
    levels = [base, base[::2, ::2].copy(), base[::4, ::4].copy()]
    return [da.from_array(level, chunks=(32, 32)) for level in levels]


def _wait_for_idle_loader(qtbot, loader, timeout=30000):
    def idle():
        return (
            loader._worker is None
            and loader._resident_worker is None
            and loader._repair_worker is None
        )

    qtbot.waitUntil(idle, timeout=timeout)


def test_chunk_source_provenance_tracked():
    """fill_unloaded_from and backdrop intervals record chunk provenance."""
    base = np.full((64, 64), 7, dtype=np.uint8)
    coarse = np.full((32, 32), 9, dtype=np.uint8)
    msvd = MultiScaleVirtualData(
        [
            da.from_array(base, chunks=(16, 16)),
            da.from_array(coarse, chunks=(16, 16)),
        ],
    )
    msvd[1].set_interval((0, 0), (32, 32))
    msvd[1].set_offset((slice(0, 32), slice(0, 32)), coarse)
    msvd[1].loaded_chunks.add(((0, 16), (0, 16)))

    msvd[0].set_interval((0, 0), (64, 64))
    msvd[0].loaded_chunks.add(((0, 16), (0, 16)))
    msvd[0].chunk_source[((0, 16), (0, 16))] = 0

    assert msvd.fill_unloaded_from(0, 1)
    # the real chunk keeps its provenance; filled chunks record source 1
    assert msvd[0].chunk_source[((0, 16), (0, 16))] == 0
    assert msvd[0].chunk_source[((16, 32), (16, 32))] == 1
    assert msvd[0].chunk_source[((48, 64), (48, 64))] == 1

    # a backdrop-initialized interval records the backdrop source
    msvd[0].set_interval((0, 0), (64, 64))  # unchanged: no-op
    msvd[0].set_interval((0, 0), (32, 64), backdrop_source=None)
    fresh = MultiScaleVirtualData(
        [
            da.from_array(base, chunks=(16, 16)),
            da.from_array(coarse, chunks=(16, 16)),
        ],
    )
    fresh[1].set_interval((0, 0), (32, 32))
    fresh[1].set_offset((slice(0, 32), slice(0, 32)), coarse)
    fresh[1].loaded_chunks.add(((0, 16), (0, 16)))
    fresh.set_interval(0, (0, 0), (64, 64), backdrop_level=1)
    assert fresh[0].chunk_source[((0, 16), (0, 16))] == 1

    # moving the interval prunes provenance with the chunks
    fresh[0].set_interval((32, 32), (64, 64))
    assert all(
        start >= 32
        for key in fresh[0].chunk_source
        for start, _stop in key
    )


def test_debug_overlay_draws_chunk_grid(
    qtbot,
    make_napari_viewer,
    multiscale_arrays,
):
    """The overlay adds a wireframe per chunk of the rendered interval,
    colored by content state, plus a HUD with the level information."""
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(
        multiscale_arrays,
        viewer=viewer,
        debug_overlay=True,
    )
    loader = layer.metadata['progressive_loader']
    try:
        overlay = loader._debug_overlay
        assert isinstance(overlay, ChunkDebugOverlay)
        _wait_for_idle_loader(qtbot, loader)

        qtbot.waitUntil(
            lambda: overlay._shapes is not None
            and len(overlay._shapes.data) > 0,
            timeout=10000,
        )
        level = int(layer.data_level)
        vdata = loader._data[level]
        lo, hi = vdata.interval
        expected = len(list(chunk_ids_in_region(vdata._boundaries, lo, hi)))
        assert len(overlay._shapes.data) == expected
        assert overlay._shapes in viewer.layers

        # the level is fully loaded once idle: the overlay converges to
        # the "real" color on every wireframe within a poll or two
        def all_real():
            colors = np.asarray(overlay._shapes.edge_color)
            return len(colors) == expected and np.allclose(
                colors,
                np.tile(STATE_COLORS['real'], (expected, 1)),
                atol=1e-3,
            )

        qtbot.waitUntil(all_real, timeout=10000)

        assert f'rendering L{level}' in viewer.text_overlay.text
        assert viewer.text_overlay.visible
    finally:
        loader.close()

    # close removed the shapes layer and restored the HUD
    assert all('chunks' not in existing.name for existing in viewer.layers)
    assert not viewer.text_overlay.visible


def test_debug_overlay_toggle(qtbot, make_napari_viewer, multiscale_arrays):
    """Runtime enable/disable creates and removes the overlay cleanly."""
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    try:
        assert loader._debug_overlay is None
        overlay = loader.enable_debug_overlay()
        assert loader._debug_overlay is overlay
        assert loader.enable_debug_overlay() is overlay  # idempotent
        _wait_for_idle_loader(qtbot, loader)
        qtbot.waitUntil(lambda: overlay._shapes is not None, timeout=10000)

        loader.disable_debug_overlay()
        assert loader._debug_overlay is None
        assert overlay._shapes not in viewer.layers
        assert overlay._timer is None
    finally:
        loader.close()
