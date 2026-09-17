"""Does a backdrop put content where the level it replaces would draw it?

Method: make the data encode its own coordinates. base[y, x] = x, block-mean
downsampled, so level L index k holds

    k * f + (f - 1) / 2

which is exactly the base coordinate of that pixel's centre. Any resample
that preserves position must therefore reproduce that value.

  1. pins the reference: napari places level L index m at world
     m*f + (f-1)/2 (measured off the live vispy transform)
  2. zoom-in backdrops (coarse source -> fine target) are unbiased
  3. zoom-out backdrops (fine source -> coarse target) are displaced by
     exactly half a source pixel -- currently xfail, the jiggle reported
     on PR #9067 (review comment on _virtual_data.py:610)
  4. the displacement vanishes for odd scale factors, which identifies
     the mechanism: for even ratios the target pixel centre lands exactly
     on a source pixel boundary, and truncating `(i + 0.5) * ratio` always
     breaks that tie the same way.

Not covered: driving the transition from a real zoom gesture. Level
selection comes from canvas draws, which do not run in the offscreen test
canvas (data_level stays 0 for any camera.zoom). The fills below go
through the same loader method the transition uses,
_backdrop_fill_layered, whose docstring documents the finer-source
(zoom-out) case as intentional.
"""

import itertools
import os
import sys

import dask.array as da
import numpy as np
import pytest

qtpy = pytest.importorskip('qtpy', reason='requires Qt backend')

pytestmark = [
    pytest.mark.skipif(
        sys.platform == 'darwin' and os.environ.get('CI') == 'true',
        reason='Progressive loading tests hang on macOS CI (no real display)',
    ),
    pytest.mark.skipif(
        qtpy.API_NAME.startswith('PySide'),
        reason='QTimer wedge under pytest with PySide6; see #9067',
    ),
]

from napari.experimental._progressive_loading import (  # noqa: E402
    _apply_chunk,
    add_progressive_loading_image,
    chunk_slices,
)
from napari.experimental._virtual_data import (  # noqa: E402
    MultiScaleVirtualData,
)

FACTORS = (1, 2, 4, 8)
N = 512


def ramp_pyramid(n, factors):
    """Levels of base[y, x] = x, block-mean downsampled."""
    base = np.tile(np.arange(n, dtype=np.float64), (n, 1))
    out = [base]
    for f in factors[1:]:
        m = n // f
        out.append(base.reshape(m, f, m, f).mean(axis=(1, 3)))
    for f, lvl in zip(factors, out, strict=True):
        k = np.arange(lvl.shape[1])
        np.testing.assert_allclose(lvl[0], k * f + (f - 1) / 2)
    return out


def as_dask(levels):
    return [
        da.from_array(lvl.astype(np.float32), chunks=(64, 64))
        for lvl in levels
    ]


@pytest.fixture
def ramp_levels():
    return as_dask(ramp_pyramid(N, FACTORS))


def _wait_idle(qtbot, loader, timeout=30000):
    qtbot.waitUntil(
        lambda: (
            loader._worker is None
            and loader._resident_worker is None
            and getattr(loader, '_repair_worker', None) is None
        ),
        timeout=timeout,
    )


def _make_loader(qtbot, make_napari_viewer, levels):
    """A live viewer/layer/loader triple.

    Close the loader inside the test body, not from a fixture teardown:
    pytest-qt's leaked-QTimer check finalizes first and would flag the
    still-running debounce timer.
    """
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(
        levels, viewer=viewer, contrast_limits=(0, N)
    )
    loader = layer.metadata['progressive_loader']
    _wait_idle(qtbot, loader)
    return viewer, layer, loader


def _load_fully(md, level):
    vdata = md[level]
    vdata.set_interval(
        np.zeros(vdata.ndim, dtype=int),
        np.asarray(vdata.shape, dtype=int),
    )
    for key in itertools.product(
        *chunk_slices(vdata, interval=vdata.interval)
    ):
        # _apply_chunk, not set_offset: it also records the chunk as
        # loaded, which is what makes the level eligible as a backdrop
        # source
        _apply_chunk(vdata, key, np.asarray(vdata.array[key]))


def _backdrop_shift(loader, levels, factors, target, src):
    """Backdrop `target` from a fully loaded `src` and return the
    placement error of every sample, in base-level pixels."""
    loader._data = MultiScaleVirtualData(levels)
    _load_fully(loader._data, src)

    tgt = loader._data[target]
    lo = np.zeros(tgt.ndim, dtype=int)
    hi = np.asarray(tgt.shape, dtype=int)
    loader._data.set_interval(target, lo, hi)
    assert loader._backdrop_level(target, lo, hi) == src
    assert loader._backdrop_fill_layered(target, lo, hi)

    row = np.asarray(tgt.hyperslice)[0]
    x0 = int(tgt._min_coord[1])
    k = np.arange(x0, x0 + row.size)
    f_t = factors[target]
    return row - (k * f_t + (f_t - 1) / 2)


def test_napari_places_levels_on_the_pixel_centre_convention(
    qtbot,
    make_napari_viewer,
    ramp_levels,
):
    """Level L index m is drawn at world m*f + (f-1)/2, at every level.

    This is the reference the backdrop has to match; measured off the real
    node transform so no convention is assumed.
    """
    viewer, layer, loader = _make_loader(
        qtbot, make_napari_viewer, ramp_levels
    )
    vispy_layer = viewer.window._qt_viewer.canvas.layer_to_visual[layer]
    try:
        for level, f in enumerate(FACTORS):
            layer.locked_data_level = level
            _wait_idle(qtbot, loader)
            qtbot.wait(50)
            cp0_x = int(layer.corner_pixels[0][1])
            matrix = np.asarray(vispy_layer._master_transform.matrix)
            k = np.arange(16)
            # texture index k -> world x. vispy axis order is reversed,
            # so napari's last axis is column 0.
            local = np.stack(
                [
                    k + 0.5,
                    np.full(k.size, 0.5),
                    np.zeros(k.size),
                    np.ones(k.size),
                ],
                axis=1,
            )
            world_x = (local @ matrix)[:, 0]
            expected = (cp0_x + k) * f + (f - 1) / 2
            print(  # noqa: T201
                f'  level {level} (f={f}): '
                f'max|world - (m*f+(f-1)/2)| = '
                f'{np.abs(world_x - expected).max():.6f}'
            )
            np.testing.assert_allclose(world_x, expected, atol=1e-4)
    finally:
        loader.close()


@pytest.mark.parametrize(('target', 'src'), [(0, 3), (0, 2), (1, 3), (1, 2)])
def test_zoom_in_backdrop_is_unbiased(
    qtbot, make_napari_viewer, ramp_levels, target, src
):
    """Coarse source -> fine target: error is pure nearest-neighbour
    quantization -- zero-mean, within half a source pixel."""
    _, _, loader = _make_loader(qtbot, make_napari_viewer, ramp_levels)
    original = loader._data
    try:
        shift = _backdrop_shift(loader, ramp_levels, FACTORS, target, src)
        f_s = FACTORS[src]
        print(  # noqa: T201
            f'  L{src}(f={f_s}) -> L{target}(f={FACTORS[target]}): '
            f'mean={shift.mean():+.4f}  max|.|={np.abs(shift).max():.4f} '
            f'base-px  (NN bound {f_s / 2:.2f})'
        )
        assert abs(shift.mean()) < 1e-9
        assert np.abs(shift).max() <= f_s / 2
    finally:
        # the measurement swapped in its own pyramid; put the real one
        # back so close() tears down the state it set up
        loader._data = original
        loader.close()


@pytest.mark.xfail(
    strict=True,
    reason='PR #9067: zoom-out backdrops are displaced by half a source '
    'pixel; drop this marker when backdrop_for stops point-sampling for '
    'ratio > 1',
)
@pytest.mark.parametrize(('target', 'src'), [(2, 1), (3, 1), (3, 0), (2, 0)])
def test_zoom_out_backdrop_is_not_shifted(
    qtbot, make_napari_viewer, ramp_levels, target, src
):
    """Fine source -> coarse target: currently every sample is displaced
    by the same +f_src/2 -- a shift, not quantization noise.

    At the transition the source level is drawn at roughly one screen
    pixel per source pixel, so this is ~0.5 screen pixels of jump on
    every power-of-two zoom-out -- and ~0.5 back when the real chunks
    land. That is the reported jiggle.
    """
    _, _, loader = _make_loader(qtbot, make_napari_viewer, ramp_levels)
    original = loader._data
    try:
        shift = _backdrop_shift(loader, ramp_levels, FACTORS, target, src)
        f_s, f_t = FACTORS[src], FACTORS[target]
        print(  # noqa: T201
            f'  L{src}(f={f_s}) -> L{target}(f={f_t}): '
            f'mean={shift.mean():+.4f} base-px '
            f'= {shift.mean() / f_s:+.4f} source-px '
            f'= {shift.mean() / f_t:+.4f} target-px  '
            f'(spread {shift.min():+.4f}..{shift.max():+.4f})'
        )
        np.testing.assert_allclose(shift, 0.0, atol=1e-9)
    finally:
        # the measurement swapped in its own pyramid; put the real one
        # back so close() tears down the state it set up
        loader._data = original
        loader.close()


def test_odd_scale_factors_are_not_shifted(qtbot, make_napari_viewer):
    """The shift is an even-ratio tie-break, not a general half-pixel
    error: with odd factors the target centre falls strictly inside a
    source pixel and the same code is exact."""
    odd = (1, 3, 9)
    levels = as_dask(ramp_pyramid(576, odd))
    _, _, loader = _make_loader(qtbot, make_napari_viewer, levels)
    original = loader._data
    try:
        shift = _backdrop_shift(loader, levels, odd, target=2, src=1)
        print(  # noqa: T201
            f'  odd factors {odd}, L1(f=3) -> L2(f=9): '
            f'mean={shift.mean():+.4f}  '
            f'max|.|={np.abs(shift).max():.4f} base-px'
        )
        np.testing.assert_allclose(shift, 0.0, atol=1e-9)
    finally:
        # the measurement swapped in its own pyramid; put the real one
        # back so close() tears down the state it set up
        loader._data = original
        loader.close()
