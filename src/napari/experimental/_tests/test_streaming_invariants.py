"""Deterministic invariants behind smooth large-data browsing.

Wall-clock freeze durations cannot be asserted reliably in CI
(hardware and network variance), but each of the three qualitative
regressions reported when browsing large remote (S3) multiscale data
maps to a mechanism whose violation causes it. These tests pin those
mechanisms:

1. "Still freezes for a few seconds when zooming/panning"
   -> every source-array read stays off the GUI thread
      (:func:`test_camera_move_does_no_sync_read_on_gui_thread`), and
      the synchronous backdrop touches only the viewport-sized region,
      so its cost scales with screen size, not dataset size
      (:func:`test_sync_backdrop_bounded_to_viewport`). The per-frame
      GPU upload bound lives in ``test_glir_metering.py``
      (``test_glir_flush_never_exceeds_frame_budget``).

2. "Stops loading tiles, continues only after some interaction"
   -> a single settled view change runs to full coverage with no
      further events
      (:func:`test_settled_view_change_completes_without_further_events`),
      and a lost present-release cannot veto staged content forever
      (:func:`test_held_present_self_releases_by_deadline`). The
      drain-notification guard lives in ``test_glir_metering.py``
      (``test_metered_upload_within_budget_still_notifies_drain``).

3. "Always displays data -- at lower resolution if the high-res data
   is not yet loaded"
   -> the resident coarsest level covers the whole dataset after init
      (:func:`test_resident_level_covers_volume_after_init`), and a pan
      into unfetched territory is backfilled from coarser levels before
      the high-res chunks arrive
      (:func:`test_pan_to_new_region_backfills_from_coarse`). The
      level-switch case is covered by
      ``test_level_switch_keeps_canvas_filled`` in
      ``test_progressive_loading.py``.

The complementary wall-clock numbers (max main-thread stall, time to
first content, time to coverage) live in the opt-in benchmark
``test_progressive_benchmark.py``.
"""

import itertools
import os
import sys
import threading
import time

import dask.array as da
import numpy as np
import pytest

pytest.importorskip('qtpy', reason='requires Qt backend')

pytestmark = pytest.mark.skipif(
    sys.platform == 'darwin' and os.environ.get('CI') == 'true',
    reason='Progressive loading tests hang on macOS CI (no real display)',
)

from napari.experimental import _progressive_loading  # noqa: E402
from napari.experimental._progressive_loading import (  # noqa: E402
    _chunk_id,
    add_progressive_loading_image,
    chunk_slices,
)


@pytest.fixture(autouse=True)
def _no_progress_bars(monkeypatch):
    """Suppress napari's Qt progress bars for these tests.

    The activity-dock progress bar runs ``processEvents()`` on updates;
    that nested event processing intermittently wedges Qt timer
    dispatch in headless macOS pytest runs. Progress-bar cosmetics are
    not what these invariants test.
    """
    monkeypatch.setattr(
        _progressive_loading.ProgressiveLoader,
        '_make_progress',
        lambda self, total, description: None,
    )


@pytest.fixture
def multiscale_arrays():
    """A small in-memory multiscale pyramid backed by dask.

    Values are drawn from [1, 255], so a zero anywhere in resident data
    is an unfilled hole, never real content.
    """
    base = np.random.default_rng(0).integers(
        1,
        255,
        size=(256, 256),
        dtype=np.uint8,
    )
    levels = [base, base[::2, ::2].copy(), base[::4, ::4].copy()]
    return [da.from_array(level, chunks=(32, 32)) for level in levels]


@pytest.fixture
def big_multiscale_arrays():
    """A pyramid large enough that one interval cannot cover level 0."""
    base = np.random.default_rng(0).integers(
        1,
        255,
        size=(1024, 1024),
        dtype=np.uint8,
    )
    levels = [base, base[::2, ::2].copy(), base[::4, ::4].copy()]
    return [da.from_array(level, chunks=(64, 64)) for level in levels]


@pytest.fixture
def multiscale_3d_arrays():
    base = np.random.default_rng(0).integers(
        1,
        255,
        size=(64, 64, 64),
        dtype=np.uint8,
    )
    levels = [base, base[::2, ::2, ::2].copy(), base[::4, ::4, ::4].copy()]
    return [da.from_array(level, chunks=(16, 16, 16)) for level in levels]


def _wait_for_idle_loader(qtbot, loader, timeout=30000):
    """Wait until the loader has no in-flight fetch workers."""

    def idle():
        return (
            loader._worker is None
            and loader._resident_worker is None
            and getattr(loader, '_repair_worker', None) is None
        )

    qtbot.waitUntil(idle, timeout=timeout)


def _interval_fully_loaded(vdata) -> bool:
    """Every chunk of the resident interval is marked loaded."""
    interval = vdata.interval
    if interval is None:
        return False
    keys = chunk_slices(vdata, interval=interval)
    return all(
        _chunk_id(key) in vdata.loaded_chunks
        for key in itertools.product(*keys)
    )


class _ProxyArray:
    """Base for array wrappers that intercept ``__getitem__``."""

    def __init__(self, array):
        self._array = array

    @property
    def shape(self):
        return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def chunks(self):
        return self._array.chunks


class _ThreadRecordingArray(_ProxyArray):
    """Records the thread ident of every ``__getitem__``.

    Stands in for a remote (S3) zarr level: any ``__getitem__`` on the
    GUI thread would be a synchronous network read there, i.e. a freeze.
    """

    def __init__(self, array, reads: list):
        super().__init__(array)
        self._reads = reads

    def __getitem__(self, key):
        self._reads.append(threading.get_ident())
        return self._array[key]


class _GatedArray(_ProxyArray):
    """Array whose reads block until ``gate`` is set.

    A controllable stand-in for slow remote (S3) chunk reads: with the
    gate cleared, a fetch pass stays in flight indefinitely, so tests
    can assert mid-pass state deterministically instead of racing a
    real-time rate limit.
    """

    def __init__(self, array, gate: threading.Event):
        super().__init__(array)
        self._gate = gate
        #: set as soon as any read has *started* (is blocked on the gate)
        self.touched = threading.Event()

    def __getitem__(self, key):
        self.touched.set()
        if not self._gate.wait(timeout=30):
            raise TimeoutError('gated array was never released')
        return self._array[key]


# ---------- 1. freezes: no synchronous source reads on the GUI thread ----


def test_camera_move_does_no_sync_read_on_gui_thread(
    qtbot,
    make_napari_viewer,
    big_multiscale_arrays,
):
    """A camera move must not read the source arrays on the GUI thread.

    The freeze reported when zooming into large remote data is, by
    definition, synchronous main-thread work during a camera event; the
    dominant candidate is a blocking chunk read. All reads must run on
    the ``@thread_worker`` fetch threads.
    """
    reads: list[int] = []
    wrapped = [
        _ThreadRecordingArray(level, reads) for level in big_multiscale_arrays
    ]
    viewer = make_napari_viewer()
    # explicit contrast limits: estimation is a deliberate one-time read
    # at add time, not part of the camera-move path under test
    layer = add_progressive_loading_image(
        wrapped,
        viewer=viewer,
        contrast_limits=(0, 255),
        interval_max_bytes=256 * 256,
    )
    loader = layer.metadata['progressive_loader']
    try:
        _wait_for_idle_loader(qtbot, loader)
        layer.locked_data_level = 0
        qtbot.waitUntil(
            lambda: loader._data[0].interval is not None,
            timeout=10000,
        )
        _wait_for_idle_loader(qtbot, loader)

        main_thread = threading.get_ident()
        # moving the interval drops out-of-view chunk ids while adding
        # new ones, so compare identities, not counts
        ids_before = set(loader._data[0].loaded_chunks)
        reads.clear()

        # a pan to unfetched territory through the debounced camera
        # path. The camera hooks are invoked directly rather than by
        # writing viewer.camera.*: on a headless viewer that write
        # intermittently wedges Qt's timer dispatch on macOS, and the
        # corner_pixels update is what the draw loop would derive from
        # the camera anyway.
        layer.corner_pixels = np.array([[704, 704], [959, 959]])
        loader._on_interaction()  # the non-debounced camera hook
        loader._debounced_check()  # the debounced camera hook
        qtbot.waitUntil(
            lambda: bool(loader._data[0].loaded_chunks - ids_before),
            timeout=10000,
        )
        _wait_for_idle_loader(qtbot, loader)

        assert reads, 'the camera move should have fetched chunks'
        assert main_thread not in reads, (
            'source array was read synchronously on the GUI thread'
        )
    finally:
        loader.close()


def test_sync_backdrop_bounded_to_viewport(
    qtbot,
    make_napari_viewer,
    big_multiscale_arrays,
):
    """The synchronous 2D backdrop only gathers and patches the visible
    viewport region, never the full interval.

    ``_sync_backdrop_2d`` runs on the GUI thread inside the ``set_data``
    emission, so its cost must scale with the screen, not the dataset;
    the off-screen margin is repaired on a worker thread.
    """
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(
        big_multiscale_arrays,
        viewer=viewer,
        contrast_limits=(0, 255),
    )
    loader = layer.metadata['progressive_loader']
    try:
        _wait_for_idle_loader(qtbot, loader)

        fills: list[tuple] = []
        patches: list[tuple] = []
        loader._backdrop_fill_layered = lambda level, lo, hi: (
            fills.append((level, list(lo), list(hi))) or True
        )
        loader._patch_texture_region = lambda vdata, lo, hi, block=None: (
            patches.append((list(lo), list(hi))) or True
        )
        loader._update_node = lambda: None
        loader._repair_backdrop = lambda: None

        # a 100x100 on-screen viewport inside a 1024x1024 level-0 interval
        displayed = tuple(layer._slice_input.displayed)
        layer._last_data_bbox = (
            displayed,
            np.array([[400.0, 400.0], [499.0, 499.0]]),
        )
        interval_min = np.array([0, 0])
        interval_max = np.array([1024, 1024])
        assert loader._sync_backdrop_2d(0, interval_min, interval_max)

        viewport = ([400, 400], [500, 500])
        assert fills == [(0, *viewport)]
        assert patches == [viewport]
        fill_extent = np.array(fills[0][2]) - np.array(fills[0][1])
        interval_extent = interval_max - interval_min
        assert np.all(fill_extent * 4 < interval_extent), (
            'synchronous backdrop work not bounded to the viewport'
        )
    finally:
        loader.close()


# ---------- 2. stalls: loading completes without further interaction ----


def test_settled_view_change_completes_without_further_events(
    qtbot,
    make_napari_viewer,
    big_multiscale_arrays,
    monkeypatch,
):
    """One settled view change loads its whole viewport with no help.

    Reproduces the reported stall shape: a pan starts a fetch pass, the
    user briefly interacts again (batches buffer during the hold), the
    interaction settles once — and then nothing else happens. The pass
    must still run to full coverage and reconcile on its own; if any
    tile waits for the *next* interaction, this fails.
    """
    gate = threading.Event()
    gate.set()  # open during setup: initial passes run at full speed
    levels = list(big_multiscale_arrays)
    gated = _GatedArray(big_multiscale_arrays[0], gate)
    levels[0] = gated
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(
        levels,
        viewer=viewer,
        contrast_limits=(0, 255),
        interval_max_bytes=256 * 256,
    )
    loader = layer.metadata['progressive_loader']
    try:
        _wait_for_idle_loader(qtbot, loader)
        layer.locked_data_level = 0
        qtbot.waitUntil(
            lambda: loader._data[0].interval is not None,
            timeout=10000,
        )
        _wait_for_idle_loader(qtbot, loader)

        gate.clear()  # reads now block: the pass stays in flight
        gated.touched.clear()
        layer.corner_pixels = np.array([[704, 704], [959, 959]])
        loader._check()  # what the debounce timer delivers after a pan
        assert loader._worker is not None, 'expected an in-flight pass'

        # wait until a fetch is genuinely in flight (blocked on the
        # gate), then start the drag: the interaction hold pauses any
        # further fetches, but the in-flight one completes and its
        # batch must buffer rather than land
        qtbot.waitUntil(gated.touched.is_set, timeout=10000)
        loader._on_interaction()
        assert loader._holding
        # during a real drag every event extends the hold AND re-arms
        # the debounce timer, so the debounced _check never fires
        # mid-drag; model that deterministically instead of racing the
        # 150 ms hold window and 100 ms debounce with repeated calls
        loader._hold_until = time.monotonic() + 30.0
        loader._debounce_timer.stop()
        monkeypatch.setattr(
            loader._debounce_timer,
            'start',
            lambda *args, **kwargs: None,
        )
        # let the fetch generator's batching window (batch_seconds)
        # elapse so the in-flight chunk yields the moment it completes
        # instead of being coalesced into a later, post-hold batch
        qtbot.wait(100)
        gate.set()
        qtbot.waitUntil(lambda: bool(loader._held_batches), timeout=10000)

        # the single settle event; from here on, NO further interaction
        loader._check()
        assert not loader._holding

        qtbot.waitUntil(lambda: loader._worker is None, timeout=30000)
        _wait_for_idle_loader(qtbot, loader)

        vdata = loader._data[0]
        assert _interval_fully_loaded(vdata), (
            'tiles of the settled view were never loaded — loading would '
            'only resume on the next interaction'
        )
        assert loader._held_batches == []
        lo, hi = vdata.interval
        key = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
        np.testing.assert_array_equal(
            np.asarray(vdata[key]),
            np.asarray(big_multiscale_arrays[0][key]),
        )
        # nothing staged is left waiting for an interaction to present
        assert loader._dbuf is None or not loader._dbuf.dirty
    finally:
        gate.set()
        loader.close()


def test_held_present_self_releases_by_deadline(
    qtbot,
    make_napari_viewer,
    multiscale_3d_arrays,
    monkeypatch,
):
    """A present hold that is never released expires on its own.

    ``hold_presents`` vetoes texture swaps so an unprepared interval
    cannot flash; if the matching ``release_presents`` is lost (its
    owner errored or was torn down), the deadline must unblock
    presentation — otherwise staged content is stranded until the next
    interaction, exactly the reported stall.
    """
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_progressive_loading_image(multiscale_3d_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    try:
        _wait_for_idle_loader(qtbot, loader)
        layer.locked_data_level = 0
        qtbot.waitUntil(lambda: loader._dbuf is not None, timeout=10000)
        _wait_for_idle_loader(qtbot, loader)
        dbuf = loader._dbuf

        # settle engagement leftovers: a headless canvas never flushes
        # GLIR, so zero the deadlines and present until the pair is clean
        def clean():
            dbuf._stage_deadline = 0.0
            dbuf._reshape_deadline = 0.0
            dbuf.present()
            return not dbuf.dirty and not dbuf._reshape_pending

        qtbot.waitUntil(clean, timeout=10000)
        # isolate the hold: the pre-flush upload gate never opens headless
        monkeypatch.setattr(dbuf, '_queued_upload_bytes', lambda: 0)
        node = loader._get_volume_node()
        front_before = dbuf._front

        # the default hold is deadline-bounded, never indefinite
        dbuf.hold_presents()
        assert dbuf._present_hold_until - time.monotonic() <= 1.6

        dbuf.hold_presents(timeout=0.3)
        dbuf._suppress_full = False
        node.set_data(np.full(dbuf.shape, 3, dtype=np.uint8))
        dbuf._stage_deadline = 0.0
        assert dbuf.dirty
        assert not dbuf.present(), 'present was not vetoed during the hold'
        assert dbuf._front is front_before

        # release_presents never arrives; the deadline alone must unblock
        qtbot.waitUntil(
            lambda: dbuf.present() or dbuf._front is not front_before,
            timeout=5000,
        )
        assert dbuf._front is not front_before
        assert node._texture is dbuf._front
    finally:
        loader.close()


# ---------- 3. never blank: coarse data everywhere, immediately ----------


def test_resident_level_covers_volume_after_init(
    qtbot,
    make_napari_viewer,
    multiscale_arrays,
):
    """After init the coarsest level is fully resident with real data.

    This is the "always displays data" foundation: wherever the user
    pans or zooms, a coarse backdrop source exists, so the canvas can
    show *something* everywhere before high-res chunks arrive.
    """
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    try:
        _wait_for_idle_loader(qtbot, loader)

        coarsest = loader._data[loader._resident_level]
        lo, hi = coarsest.interval
        assert list(lo) == [0, 0]
        assert list(hi) == list(coarsest.shape)
        # source values are all >= 1: a zero would be a coverage hole
        assert not (coarsest.hyperslice == 0).any()
        assert _interval_fully_loaded(coarsest)
    finally:
        loader.close()


def test_pan_to_new_region_backfills_from_coarse(
    qtbot,
    make_napari_viewer,
    big_multiscale_arrays,
):
    """Panning to unfetched territory shows upsampled coarse data
    before the high-resolution chunks arrive.

    Extends ``test_level_switch_keeps_canvas_filled`` from level
    switches to pans: the moment a pan's fetch pass starts (and while
    it is still streaming), the new interval must already be filled
    from a coarser resident level rather than showing zeros.
    """
    gate = threading.Event()
    gate.set()  # open during setup: initial passes run at full speed
    levels = list(big_multiscale_arrays)
    levels[0] = _GatedArray(big_multiscale_arrays[0], gate)
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(
        levels,
        viewer=viewer,
        contrast_limits=(0, 255),
        interval_max_bytes=256 * 256,
    )
    loader = layer.metadata['progressive_loader']
    try:
        _wait_for_idle_loader(qtbot, loader)
        layer.locked_data_level = 0
        qtbot.waitUntil(
            lambda: loader._data[0].interval is not None,
            timeout=10000,
        )
        _wait_for_idle_loader(qtbot, loader)
        vdata = loader._data[0]

        # block high-res reads: the assertions below run strictly
        # before any high-res chunk can cover the fresh region
        gate.clear()
        fresh = np.array([[704, 704], [959, 959]])
        assert not vdata.covers(fresh[0], fresh[1] + 1)
        layer.corner_pixels = fresh
        loader._check()

        assert loader._worker is not None, 'high-res fetch not in flight'
        lo, hi = vdata.interval
        assert np.all(np.asarray(lo) <= fresh[0])
        assert np.all(np.asarray(hi) >= fresh[1] + 1)
        with vdata.lock:
            zero_fraction = float((vdata.hyperslice == 0).mean())
        # source values are all >= 1: zeros are unfilled pixels
        assert zero_fraction < 0.05, (
            f'{zero_fraction:.0%} of the panned-to viewport is blank '
            'while high-res chunks load'
        )
    finally:
        gate.set()
        _wait_for_idle_loader(qtbot, loader)
        loader.close()


def test_view_refines_through_intermediate_levels(
    qtbot,
    make_napari_viewer,
    big_multiscale_arrays,
):
    """A slow target level should not block intermediate refinement.

    The smoothest large-data viewers resolve a new view coarse-to-fine:
    the full viewport appears at low resolution immediately, then
    sharpens through each pyramid level as data arrives. Here the
    coarsest-level backdrop covers the first step, but nothing fetches
    the levels between the resident coarsest and the target — with the
    target level slow (gated, like cold S3 reads), the view should
    still acquire real fetched data at an intermediate level.
    """
    gate = threading.Event()
    gate.set()
    levels = list(big_multiscale_arrays)
    levels[0] = _GatedArray(big_multiscale_arrays[0], gate)
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(
        levels,
        viewer=viewer,
        contrast_limits=(0, 255),
        interval_max_bytes=256 * 256,
    )
    loader = layer.metadata['progressive_loader']
    try:
        _wait_for_idle_loader(qtbot, loader)
        layer.locked_data_level = 0
        qtbot.waitUntil(
            lambda: loader._data[0].interval is not None,
            timeout=10000,
        )
        _wait_for_idle_loader(qtbot, loader)

        # pan to fresh territory with the target level blocked: only
        # intermediate-level fetches could sharpen the view now
        gate.clear()
        intermediate_before = set(loader._data[1].loaded_chunks)
        layer.corner_pixels = np.array([[704, 704], [959, 959]])
        loader._check()
        qtbot.waitUntil(
            lambda: bool(loader._data[1].loaded_chunks - intermediate_before),
            timeout=10000,
        )
    finally:
        gate.set()
        _wait_for_idle_loader(qtbot, loader)
        loader.close()


def test_rapid_view_changes_never_blank(
    qtbot,
    make_napari_viewer,
    big_multiscale_arrays,
):
    """Back-to-back view changes with a slow target never leave the new
    interval blank, even before any off-thread repair has run.

    Regression: bounding the pass-start backdrop fill to the last
    *rendered* viewport box left freshly exposed screen area black
    during fast pans/zooms — that box lags the camera exactly then.
    """
    gate = threading.Event()
    gate.set()
    levels = list(big_multiscale_arrays)
    levels[0] = _GatedArray(big_multiscale_arrays[0], gate)
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(
        levels,
        viewer=viewer,
        contrast_limits=(0, 255),
        interval_max_bytes=256 * 256,
    )
    loader = layer.metadata['progressive_loader']
    try:
        _wait_for_idle_loader(qtbot, loader)
        layer.locked_data_level = 0
        qtbot.waitUntil(
            lambda: loader._data[0].interval is not None,
            timeout=10000,
        )
        _wait_for_idle_loader(qtbot, loader)
        vdata = loader._data[0]

        gate.clear()  # the target level is slow (cold remote reads)
        # a recorded viewport box that lags the camera, as during a
        # fast drag (the last rendered slice showed a different region)
        displayed = tuple(layer._slice_input.displayed)
        layer._last_data_bbox = (
            displayed,
            np.array([[0.0, 0.0], [63.0, 63.0]]),
        )
        # two rapid moves, no settling in between
        for corners in ([[512, 512], [767, 767]], [[704, 704], [959, 959]]):
            layer.corner_pixels = np.array(corners)
            loader._check()
        with vdata.lock:
            zero_fraction = float((vdata.hyperslice == 0).mean())
        assert zero_fraction < 0.05, (
            f'{zero_fraction:.0%} of the interval is blank right after '
            'rapid moves'
        )
    finally:
        gate.set()
        _wait_for_idle_loader(qtbot, loader)
        loader.close()


def test_repair_requests_chain_when_busy(
    qtbot,
    make_napari_viewer,
    big_multiscale_arrays,
):
    """A repair requested while one runs is chained, never dropped.

    Regression: during fast moves every repair trigger can land while a
    (stale-region) repair is in flight; dropping those requests left
    the fresh region's margin unfilled until some unrelated event.
    """
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(
        big_multiscale_arrays,
        viewer=viewer,
        contrast_limits=(0, 255),
        interval_max_bytes=256 * 256,
    )
    loader = layer.metadata['progressive_loader']
    try:
        _wait_for_idle_loader(qtbot, loader)
        layer.locked_data_level = 0
        qtbot.waitUntil(
            lambda: loader._data[0].interval is not None,
            timeout=10000,
        )
        _wait_for_idle_loader(qtbot, loader)

        fills = []
        release = threading.Event()

        def slow_fill(level, lo, hi):
            fills.append(level)
            release.wait(10)
            return True

        loader._backdrop_fill_layered = slow_fill
        loader._repair_backdrop()
        qtbot.waitUntil(lambda: len(fills) == 1, timeout=10000)
        loader._repair_backdrop()  # busy: must chain, not drop
        assert loader._repair_again
        release.set()
        qtbot.waitUntil(lambda: len(fills) == 2, timeout=10000)
        _wait_for_idle_loader(qtbot, loader)
        assert not loader._repair_again
    finally:
        release.set()
        loader.close()


def test_oversized_3d_corners_are_retiled(
    qtbot,
    make_napari_viewer,
    multiscale_3d_arrays,
):
    """Corner pixels that escaped every 3D cap are retiled before
    slicing.

    A crop far beyond the tile budget becomes a texture the GL driver
    refuses to load; vispy then samples the zero texture and the canvas
    renders black ("GLD_TEXTURE_INDEX_3D is unloadable"). Whatever path
    produced such corners, the loader's final guard must rewrite them
    into a loadable view-centered tile.
    """
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_progressive_loading_image(
        multiscale_3d_arrays,
        viewer=viewer,
        auto_level_3d=False,
        interval_max_bytes=32**3,
        tile_max_bytes_3d=32**3,
    )
    loader = layer.metadata['progressive_loader']
    try:
        _wait_for_idle_loader(qtbot, loader)
        layer.locked_data_level = 0
        _wait_for_idle_loader(qtbot, loader)

        # corners escape the caps (simulating a path the tiling missed):
        # the full 64^3 crop is 16x the 32^3-byte tile budget
        displayed = list(layer._slice_input.displayed)
        layer.corner_pixels = np.array([[0, 0, 0], [63, 63, 63]])
        loader._check()

        extent = (
            layer.corner_pixels[1, displayed]
            - layer.corner_pixels[0, displayed]
            + 1
        )
        assert np.all(extent <= 32), (
            f'oversized 3D corners were not retiled: extent {extent}'
        )
        _wait_for_idle_loader(qtbot, loader)
    finally:
        loader.close()


def test_pan_sharpens_coarse_to_fine(
    qtbot,
    make_napari_viewer,
):
    """A fresh view resolves through the resolutions, coarse to fine.

    Levels carry distinct constant values, so the rendered level's
    content identifies which resolution each pixel currently shows.
    With the target level slow (gated), a pan must first show the
    coarsest backdrop, then sharpen to the intermediate level's data,
    and finally — once the target level unblocks — the real thing.
    """
    value = {0: 30, 1: 20, 2: 10}
    levels = [
        da.from_array(
            np.full((size, size), value[i], dtype=np.uint8),
            chunks=(64, 64),
        )
        for i, size in enumerate((1024, 512, 256))
    ]
    gate = threading.Event()
    gate.set()
    gated = _GatedArray(levels[0], gate)
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(
        [gated, levels[1], levels[2]],
        viewer=viewer,
        contrast_limits=(0, 255),
        interval_max_bytes=256 * 256,
    )
    loader = layer.metadata['progressive_loader']
    try:
        _wait_for_idle_loader(qtbot, loader)
        layer.locked_data_level = 0
        qtbot.waitUntil(
            lambda: loader._data[0].interval is not None,
            timeout=10000,
        )
        _wait_for_idle_loader(qtbot, loader)
        vdata = loader._data[0]

        gate.clear()
        layer.corner_pixels = np.array([[704, 704], [959, 959]])
        loader._check()

        def fraction_at(level_value):
            with vdata.lock:
                hyperslice = vdata.hyperslice
                return float((hyperslice == level_value).mean())

        # step 1: the synchronous backdrop already shows the coarsest
        # level everywhere (never blank), possibly already improved
        assert fraction_at(0) < 0.05

        # step 2: with the target still blocked, the ladder's
        # intermediate fetches sharpen the view to level-1 content
        qtbot.waitUntil(lambda: fraction_at(value[1]) > 0.9, timeout=10000)

        # step 3: releasing the target level completes the pass and the
        # view reaches full resolution without any further interaction
        gate.set()
        qtbot.waitUntil(lambda: loader._worker is None, timeout=30000)
        _wait_for_idle_loader(qtbot, loader)
        assert _interval_fully_loaded(vdata)
        assert fraction_at(value[0]) == 1.0
    finally:
        gate.set()
        _wait_for_idle_loader(qtbot, loader)
        loader.close()
