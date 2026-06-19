"""Progressive (chunk-wise) loading for multiscale images.

This module implements progressive loading on top of napari's standard
multiscale ``Image`` layer. A single layer is added to the viewer; its
per-level data objects are :class:`~napari.experimental._virtual_data.VirtualData`
instances that present the full array shape while only keeping the visible,
chunk-aligned region in memory.

A :class:`ProgressiveLoader` watches the camera and dims, and on every view
change it:

1. reads the data level napari selected (respecting
   ``layer.locked_data_level``) and the visible ``corner_pixels``,
2. moves the level's resident interval to cover the view, initializing
   newly exposed regions from a coarser resident level (so the canvas is
   never empty),
3. fetches the missing chunks on a background thread in priority order
   (view-center first in 2D; camera depth/center-line in 3D), writing each
   chunk into the virtual data and refreshing the layer through napari's
   normal slicing pipeline.

The lowest-resolution level is kept fully resident (up to a size limit),
which provides instant low-resolution context everywhere, powers layer
thumbnails, and serves as the backdrop source for finer levels.

Use :func:`add_progressive_loading_image` to add a progressively loading
image to a viewer, or :func:`add_progressive_loading_labels` for a labels
layer.

This is experimental: expect breaking changes and rough edges, and please
report issues to https://github.com/napari/napari/issues.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import threading
import time
import weakref
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING

import numpy as np

# imported at module load: a lazy first-use import inside a fetch pass
# costs seconds of main-thread time under fetch-thread GIL pressure
from napari.experimental._texture_swap import (
    DoubleBufferedImageTexture,
    DoubleBufferedVolumeTexture,
)
from napari.experimental._virtual_data import (
    MultiScaleVirtualData,
    VirtualData,
    chunk_boundaries,
)
from napari.utils import progress

# Qt imports are deferred so the module can be imported in headless
# environments (no Qt backend) — the tests and pure-data helpers
# (chunk_slices, chunk_priority_*, VirtualData) remain usable.
try:
    from qtpy.QtCore import QTimer

    from napari.qt.threading import thread_worker
except ImportError:
    QTimer = None  # type: ignore[assignment,misc]

    def thread_worker(func=None, **kwargs):  # type: ignore[misc]
        """No-op stand-in so ``@thread_worker`` doesn't crash at import."""
        return func if func is not None else lambda f: f

if TYPE_CHECKING:
    import napari

LOGGER = logging.getLogger(__name__)

#: Maximum size of the always-resident coarsest level, in bytes.
DEFAULT_RESIDENT_MAX_BYTES = 256 * 1024**2
#: Maximum size of a single level's resident interval, in bytes.
DEFAULT_INTERVAL_MAX_BYTES = 512 * 1024**2
#: Show a progress bar (activity dock) for fetch passes with at least this
#: many chunks; short interactive passes stay silent.
PROGRESS_MIN_CHUNKS = 16
#: 3D auto level selection coarsens until the viewport tile needs at most
#: this many chunks, so a pass completes in seconds rather than minutes.
#: Sized so the BYTE cap (TILE_MAX_BYTES_3D) is the binding constraint
#: for ordinary chunkings: a 16e6 tile (251^3) of 32^3 chunks is 512
#: chunks; a 33e6 tile (320^3) is 1000. The old default (384) silently
#: vetoed the level the byte cap was tuned to allow — resolution arrived
#: a full level late. The guard still protects against pathological
#: chunkings (deep pyramids with tiny chunks).
DEFAULT_MAX_CHUNKS_PER_PASS = 1024
#: Maximum size of a 3D sub-volume tile, in bytes. Full-tile GPU uploads
#: (pass start/end) block the GUI for roughly size / 125 MB/s on slow GL
#: drivers, so this is deliberately much smaller than the memory budget.
#: Swept on macOS GL-over-Metal: 16e6 (251^3 tiles) vs 33e6 (322^3) gave
#: 2.8x cheaper raycast draws (21.9 vs 61.5 ms/DRAW) and ~400ms vs
#: ~700ms typical interaction stalls, at the cost of a wider tile-shape
#: vocabulary (covered by the texture pool).
DEFAULT_TILE_MAX_BYTES_3D = int(16e6)


# ---------- chunk geometry ----------


def chunk_slices(data, interval: tuple | None = None) -> list[list[slice]]:
    """Per-dimension lists of chunk slices, optionally clipped to a region.

    Parameters
    ----------
    data : VirtualData or array-like
        Object whose chunk grid should be enumerated.
    interval : tuple of (min_coord, max_coord), optional
        Half-open bounds per dimension. Only chunks intersecting the
        interval are returned.

    Returns
    -------
    list of list of slice
        For each dimension, the slices of every chunk along it. The full
        set of chunk keys is the cartesian product across dimensions.

    """
    if isinstance(data, VirtualData):
        boundaries = data._boundaries
    else:
        boundaries = chunk_boundaries(data)

    result: list[list[slice]] = []
    for dim, bounds in enumerate(boundaries):
        starts, stops = bounds[:-1], bounds[1:]
        if interval is not None:
            min_c = int(interval[0][dim])
            max_c = int(interval[1][dim])
            first = int(np.searchsorted(stops, min_c, side='right'))
            last = int(np.searchsorted(starts, max_c, side='left'))
            starts, stops = starts[first:last], stops[first:last]
        result.append(
            [
                slice(int(start), int(stop))
                for start, stop in zip(starts, stops, strict=True)
            ],
        )
    return result


def get_chunk_center(chunk_key: tuple[slice, ...]) -> np.ndarray:
    """Return the center coordinate of a tuple-of-slices chunk key."""
    return np.array([(sl.start + sl.stop) * 0.5 for sl in chunk_key])


def visual_depth(points, camera) -> np.ndarray:
    """Compute visual depth from camera position to a(n array of) point(s).

    Parameters
    ----------
    points : (N, D) array of float
        An array of N points. This can be one point or many thanks to NumPy
        broadcasting.
    camera : napari.components.Camera
        A camera model specifying a view direction and a center or focus
        point.

    Returns
    -------
    projected_length : (N,) array of float
        Position of the points along the view vector of the camera. These
        can be negative (in front of the center) or positive (behind the
        center).

    """
    view_direction = camera.view_direction
    points_relative_to_camera = points - camera.center
    projected_length = points_relative_to_camera @ view_direction
    return projected_length


def distance_from_camera_center_line(points, camera) -> np.ndarray:
    """Compute distance from a point or array of points to camera center line.

    This is the line aligned to the camera view direction and passing
    through the camera's center point.

    Parameters
    ----------
    points : (N, D) array of float
        An array of N points. This can be one point or many thanks to NumPy
        broadcasting.
    camera : napari.components.Camera
        A camera model specifying a view direction and a center or focus
        point.

    Returns
    -------
    distances : (N,) array of float
        Distances from points to the center line of the camera.

    """
    view_direction = camera.view_direction
    projected_length = visual_depth(points, camera)
    projected = view_direction * np.reshape(projected_length, (-1, 1))
    points_relative_to_camera = points - camera.center
    distances = np.linalg.norm(projected - points_relative_to_camera, axis=-1)
    return distances


def _chunk_keys_product(
    chunk_keys: list[list[slice]],
) -> list[tuple[slice, ...]]:
    return list(itertools.product(*chunk_keys))


def chunk_priority_2D(
    chunk_keys: list[list[slice]],
    min_coord,
    max_coord,
) -> list[tuple[slice, ...]]:
    """Order chunk keys by distance from the center of the view.

    Parameters
    ----------
    chunk_keys : list of list of slice
        Per-dimension chunk slices (see :func:`chunk_slices`).
    min_coord, max_coord : sequence of int
        The visible interval in this level's coordinates.

    Returns
    -------
    list of tuple of slice
        Chunk keys sorted with the most central chunks first.

    """
    view_center = (np.asarray(min_coord) + np.asarray(max_coord)) / 2
    keys = _chunk_keys_product(chunk_keys)
    centers = np.array([get_chunk_center(key) for key in keys])
    if centers.size == 0:
        return []
    distances = np.linalg.norm(centers - view_center, axis=1)
    return [keys[i] for i in np.argsort(distances, kind='stable')]


def chunk_priority_3D(
    chunk_keys: list[list[slice]],
    min_coord,
    max_coord,
    camera_center,
    view_direction,
    center_line_weight: float = 0.5,
) -> list[tuple[slice, ...]]:
    """Order chunk keys front-to-back for 3D rendering.

    Chunks are ordered primarily by depth along the camera's view
    direction — with napari's orthographic camera, the chunk closest to
    the viewer loads first — with the distance from the camera's center
    line as a (down-weighted) secondary term so on-axis chunks lead at
    equal depth.

    Parameters
    ----------
    chunk_keys : list of list of slice
        Per-dimension chunk slices (see :func:`chunk_slices`).
    min_coord, max_coord : sequence of int
        The visible interval in this level's coordinates.
    camera_center : sequence of float
        Camera center in this level's coordinates (displayed dimensions,
        i.e. the last 3 dimensions of the chunk keys).
    view_direction : sequence of float
        Camera view direction (3-vector over the displayed dimensions).
    center_line_weight : float
        Weight of the center-line distance term relative to depth (both
        are in this level's data units).

    Returns
    -------
    list of tuple of slice
        Chunk keys sorted from highest to lowest priority.

    Notes
    -----
    The camera state is sanitized: a degenerate (zero/non-finite) view
    direction, a non-finite camera center/zoom, or values so large that
    the arithmetic overflows — all of which can occur before the 3D
    camera is fully initialized — fall back to plain
    view-center-distance ordering instead of producing NaN priorities.

    """
    keys = _chunk_keys_product(chunk_keys)
    if not keys:
        return []
    centers = np.array([get_chunk_center(key)[-3:] for key in keys])

    view_center = (
        (np.asarray(min_coord, dtype=float) + np.asarray(max_coord)) / 2
    )[-3:]
    center_view_dist = np.linalg.norm(centers - view_center, axis=-1)

    camera_center = np.asarray(camera_center, dtype=float)[-3:]
    view_direction = np.asarray(view_direction, dtype=float)[-3:]
    direction_norm = (
        float(np.linalg.norm(view_direction))
        if np.all(np.isfinite(view_direction))
        else 0.0
    )
    priority = None
    if (
        camera_center.shape == (3,)
        and np.all(np.isfinite(camera_center))
        and direction_norm >= 1e-12
    ):
        view_direction = view_direction / direction_norm
        # Large-magnitude (but finite) camera coordinates can overflow
        # the arithmetic below; compute silently and validate the result.
        with np.errstate(all='ignore'):
            relative = centers - camera_center
            depth = relative @ view_direction
            projected = view_direction * depth[:, np.newaxis]
            center_line_dist = np.linalg.norm(projected - relative, axis=-1)
            candidate = depth + center_line_weight * center_line_dist
        if np.all(np.isfinite(candidate)):
            priority = candidate
    if priority is None:
        # Camera not (yet) in a usable 3D state: order by view center only.
        priority = center_view_dist
    return [keys[i] for i in np.argsort(priority, kind='stable')]


def _chunk_id(chunk_key: tuple[slice, ...]) -> tuple[tuple[int, int], ...]:
    """Hashable identifier for a chunk key."""
    return tuple((int(sl.start), int(sl.stop)) for sl in chunk_key)


def _pack_upload_block(vdata: VirtualData, keys) -> tuple | None:
    """Contiguous copy of the union region of ``keys`` (worker thread).

    Returns ``(low, high, block)`` in absolute coordinates, or ``None``
    when the region is outside the resident interval.
    """
    ndim = vdata.ndim
    low = [min(int(k[d].start) for k in keys) for d in range(ndim)]
    high = [max(int(k[d].stop) for k in keys) for d in range(ndim)]
    with vdata.lock:
        if vdata._min_coord is None:
            return None
        low = [
            max(lo, mn) for lo, mn in zip(low, vdata._min_coord, strict=True)
        ]
        high = [
            min(hi, mx) for hi, mx in zip(high, vdata._max_coord, strict=True)
        ]
        if any(hi <= lo for lo, hi in zip(low, high, strict=True)):
            return None
        source = tuple(
            slice(lo - mn, hi - mn)
            for lo, hi, mn in zip(low, high, vdata._min_coord, strict=True)
        )
        block = np.ascontiguousarray(vdata.hyperslice[source])
    return low, high, block


def _tile_extent_3d_for(dtype: np.dtype, interval_max_bytes: int) -> int:
    """Per-axis extent of 3D sub-volume tiles (isotropic fallback).

    The largest cube that fits the interval memory budget, further
    bounded by the GL 3D texture size limit when available.
    Used as the cap when no per-level shape is known.
    """
    extent = int((interval_max_bytes / np.dtype(dtype).itemsize) ** (1 / 3))
    try:
        from napari._vispy.utils.gl import get_max_texture_sizes

        _, max_3d = get_max_texture_sizes()
        if max_3d is not None:
            extent = min(extent, int(max_3d))
    except Exception:  # pragma: no cover - no GL context  # noqa: BLE001
        pass
    return max(extent, 32)


def _anisotropic_tile_extent(
    shape: np.ndarray,
    max_bytes: int,
    itemsize: int,
    gl_max: int | None = None,
) -> np.ndarray:
    """Compute a per-axis tile extent that fits the byte budget.

    Axes whose full size already fits are kept whole; the remaining
    budget is distributed to the larger axes so anisotropic data
    (e.g. Z=42, Y=304, X=657) uses the budget efficiently instead
    of being capped to a uniform cube.
    """
    shape = np.asarray(shape, dtype=np.int64)
    tile = shape.copy()
    if gl_max is not None:
        tile = np.minimum(tile, gl_max)
    max_elements = max(max_bytes // max(itemsize, 1), 1)
    for _ in range(len(shape)):
        vol = int(np.prod(tile))
        if vol <= max_elements:
            break
        over = np.where(tile > 1)[0]
        if len(over) == 0:
            break
        # shrink the largest axis proportionally
        ratio = (max_elements / vol) ** (1.0 / len(over))
        for ax in over:
            tile[ax] = max(int(tile[ax] * ratio), 1)
    return np.maximum(tile, 1)


# ---------- background fetching ----------


def _key_nbytes(chunk_key: tuple[slice, ...], itemsize: int) -> int:
    """Size in bytes of the data selected by a concrete chunk key."""
    n = int(itemsize)
    for s in chunk_key:
        n *= max(0, int(s.stop) - int(s.start))
    return n


class _FetchRateLimiter:
    """Gate and pace fetch throughput across all fetch workers.

    ``acquire(nbytes)`` blocks the calling *worker* thread, first while
    the limiter is paused (interaction hold), then — when a
    bytes-per-second rate is configured — until issuing a fetch of that
    size keeps the average rate within budget (leaky bucket). Pacing on
    the worker side bounds every downstream cost of chunk delivery:
    GIL pressure from fetch compute, slice/refresh event cascades, and
    GPU upload traffic.

    ``pause()``/``resume()`` suspend fetching entirely while the user
    interacts. ``cancel()`` wakes all sleeping workers immediately so a
    cancelled pass can wind down without waiting out its delays.
    """

    def __init__(self, bytes_per_second: float | None = None):
        self.bytes_per_second = (
            float(bytes_per_second) if bytes_per_second else None
        )
        self._lock = threading.Lock()
        self._next_free = time.monotonic()
        self._cancelled = threading.Event()
        self._go = threading.Event()
        self._go.set()

    def acquire(self, nbytes: int) -> None:
        while not self._go.wait(timeout=1.0):  # paused
            if self._cancelled.is_set():
                return
        if self.bytes_per_second is None or self._cancelled.is_set():
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_free)
            self._next_free = start + nbytes / self.bytes_per_second
        delay = start - time.monotonic()
        if delay > 0:
            self._cancelled.wait(delay)

    def pause(self) -> None:
        self._go.clear()

    def resume(self) -> None:
        self._go.set()

    def cancel(self) -> None:
        self._cancelled.set()
        self._go.set()  # wake paused workers; cancelled passes must exit


@thread_worker
def _fetch_chunks(
    array,
    chunk_queue: list[tuple[slice, ...]],
    num_workers: int = 1,
    apply=None,
    batch_seconds: float = 0.05,
    pack=None,
    limiter: _FetchRateLimiter | None = None,
):
    """Fetch chunks from ``array``, yielding batches of completed keys.

    ``apply(chunk_key, ndarray)`` is called on the *worker* thread for
    every fetched chunk — typically ``VirtualData.set_offset``, which is
    lock-guarded numpy work that has no reason to occupy the GUI thread.
    Completed chunk keys are then yielded to the main thread in batches
    (at most one signal per ``batch_seconds``), so chunk bursts do not
    flood the Qt event loop with per-chunk signal dispatches.

    Yields lists of chunk keys (or ``pack(keys)`` when ``pack`` is
    given — e.g. to precompute a contiguous upload block on the worker
    rather than the GUI thread). With ``num_workers > 1``, chunks are
    fetched by a small thread pool (useful for GIL-releasing compute
    like numba and for remote IO); a bounded in-flight window keeps
    completion order close to the priority order of ``chunk_queue``.
    """
    itemsize = np.dtype(array.dtype).itemsize

    def fetch(chunk_key):
        if limiter is not None:
            # paces the worker BEFORE the fetch, so compute, delivery
            # and GPU upload all follow the configured byte rate
            limiter.acquire(_key_nbytes(chunk_key, itemsize))
        start = time.monotonic()
        chunk = np.asarray(array[chunk_key])
        if apply is not None:
            apply(chunk_key, chunk)
        LOGGER.debug(
            'fetched chunk %s in %.3fs',
            chunk_key,
            time.monotonic() - start,
        )
        return chunk_key

    def emit(batch):
        return pack(batch) if pack is not None else batch

    if num_workers <= 1 or len(chunk_queue) <= 1:
        batch: list = []
        last_yield = time.monotonic()
        for chunk_key in chunk_queue:
            batch.append(fetch(chunk_key))
            now = time.monotonic()
            if now - last_yield >= batch_seconds:
                last_yield = now
                yield emit(batch)
                batch = []
        if batch:
            yield emit(batch)
        return

    pool = ThreadPoolExecutor(max_workers=num_workers)
    pending = iter(chunk_queue)
    in_flight = {
        pool.submit(fetch, chunk_key): chunk_key
        for chunk_key in itertools.islice(pending, num_workers)
    }
    batch = []
    last_yield = time.monotonic()
    try:
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                in_flight.pop(future)
                next_key = next(pending, None)
                if next_key is not None:
                    try:
                        in_flight[pool.submit(fetch, next_key)] = next_key
                    except RuntimeError:  # pool/interpreter shutting down
                        pending = iter(())
                batch.append(future.result())
            now = time.monotonic()
            if batch and (now - last_yield >= batch_seconds):
                last_yield = now
                yield emit(batch)
                batch = []
        if batch:
            yield emit(batch)
    finally:
        # don't block cancellation on fetches that haven't started
        pool.shutdown(wait=False, cancel_futures=True)


class ProgressiveLoader:
    """Stream visible chunks into a multiscale image layer.

    Connects to the viewer's camera and dims events and keeps the layer's
    per-level :class:`VirtualData` intervals in sync with the visible
    region, fetching missing chunks on a background thread in priority
    order. Respects ``layer.locked_data_level`` (the resolution selector in
    the layer controls) because it always loads the level napari selected
    for rendering.

    Normally constructed by :func:`add_progressive_loading_image`; the
    instance is stored in ``layer.metadata['progressive_loader']``.

    Parameters
    ----------
    viewer : napari.Viewer
        The viewer the layer belongs to.
    layer : napari.layers.Image
        A multiscale image layer whose levels are ``VirtualData`` objects.
    data : MultiScaleVirtualData
        The coordinating multiscale wrapper for the layer's levels.
    debounce_ms : int
        Debounce interval for camera/dims events.
    refresh_interval_s : float
        Minimum time between layer refreshes while chunks stream in. The
        effective interval adapts upward when refreshes are expensive
        (e.g. full 3D texture uploads).
    resident_max_bytes : int
        Keep the coarsest level fully in memory if it is at most this big.
    interval_max_bytes : int
        Upper bound for a single level's resident interval.
    auto_level_3d : bool
        In 3D, automatically select the data level from the camera zoom
        (napari itself always renders the coarsest level in 3D). The level
        is driven through the layer's internal level lock without emitting
        events, so the resolution selector still reads "Auto"; choosing an
        explicit level in the selector suspends automatic selection until
        it is set back to "Auto". Levels too large for
        ``interval_max_bytes`` are skipped in favor of coarser ones.
    max_pixel_size_3d : float
        In 3D auto mode, target the coarsest level whose voxels project
        to at most this many screen pixels. Lower values choose finer
        (more expensive) levels sooner when zooming in.
    fetch_workers : int, optional
        Number of threads fetching chunks concurrently within a pass
        (default: up to 4, leaving at least two cores for the GUI).
        Completion order stays close to priority order. Raise this for
        high-latency remote data; lower it (or pace the store, see
        ``GenerativeZarrStore.cpu_relief``) for compute-bound sources.
    max_chunks_per_pass : int
        3D auto level selection coarsens until the viewport tile needs
        at most this many chunks, keeping pass duration reasonable.
    texture_patching : bool
        In 3D, write arriving chunks directly into the existing GPU
        texture (a partial glTexSubImage3D upload) instead of re-slicing
        and re-uploading the whole tile per refresh. The normal slicing
        pipeline still reconciles periodically and at the end of each
        pass. Greatly reduces main-thread blocking for large tiles.
    tile_max_bytes_3d : int
        Upper bound for a 3D sub-volume tile. Each pass performs a full
        tile GPU upload at its boundaries, which blocks the GUI roughly
        in proportion to this size; raise it on fast GPUs for larger
        high-resolution tiles.
    max_bytes_per_second : float, optional
        Rate-limit chunk loading to this many bytes per second (shared
        across all fetch workers). Pacing happens on the worker threads
        before each fetch, so it bounds every per-chunk cost in one
        knob: fetch compute (GIL pressure), slice/refresh event
        cascades, and GPU upload traffic. Loading takes proportionally
        longer; interaction stays smooth. ``None`` (default) is
        unlimited.
    interaction_hold : bool
        Suspend all streaming work while the user interacts (camera
        motion, slider scrubbing): fetch workers pause, arriving chunk
        batches are buffered instead of patched, throttled refreshes
        are deferred, and metered GLIR texture uploads hold. Everything
        resumes when interaction settles (the debounced check). This
        keeps interaction frames free of upload work, slice-completion
        event cascades, and fetch-thread GIL pressure.
    interactive_step_rate : float
        Multiply the volume raycast step size by this factor while the
        user interacts, restoring full quality when interaction
        settles (interactive level-of-detail, as in ParaView/Slicer).
        Raycast cost scales inversely with step size, so 2.0 means
        roughly 2x cheaper GPU frames during drags. 1.0 (or 0)
        disables.

    """

    def __init__(
        self,
        viewer: napari.Viewer,
        layer,
        data: MultiScaleVirtualData,
        *,
        debounce_ms: int = 100,
        refresh_interval_s: float = 0.03,
        resident_max_bytes: int = DEFAULT_RESIDENT_MAX_BYTES,
        interval_max_bytes: int = DEFAULT_INTERVAL_MAX_BYTES,
        auto_level_3d: bool = True,
        max_pixel_size_3d: float = 2.0,
        fetch_workers: int | None = None,
        max_chunks_per_pass: int = DEFAULT_MAX_CHUNKS_PER_PASS,
        texture_patching: bool = True,
        tile_max_bytes_3d: int = DEFAULT_TILE_MAX_BYTES_3D,
        max_bytes_per_second: float | None = None,
        interaction_hold: bool = True,
        interactive_step_rate: float = 4.0,
    ):
        self._viewer = viewer
        self._layer = layer
        self._data = data
        self._refresh_interval_s = refresh_interval_s
        self._interval_max_bytes = interval_max_bytes
        self._auto_level_3d = auto_level_3d
        self._max_pixel_size_3d = float(max_pixel_size_3d)
        self._max_chunks_per_pass = max(int(max_chunks_per_pass), 1)
        self._texture_patching = texture_patching
        self._texture_patches = 0
        self._pass_all_patched = False
        self._needs_final_reconcile = False
        self._last_node_update = 0.0
        env_workers = os.environ.get('NAPARI_PROGRESSIVE_FETCH_WORKERS')
        if env_workers:
            fetch_workers = int(env_workers)
        if fetch_workers is None:
            # leave cores for the GUI event loop: saturating every core
            # with chunk fetches makes the UI unresponsive on CPU-bound
            # (e.g. generative) stores
            try:
                n_cpus = len(os.sched_getaffinity(0))
            except AttributeError:  # pragma: no cover - macOS/Windows
                n_cpus = os.cpu_count() or 3
            fetch_workers = min(4, n_cpus - 2)
        self._fetch_workers = max(int(fetch_workers), 1)
        self._max_bytes_per_second = (
            float(max_bytes_per_second) if max_bytes_per_second else None
        )
        self._limiter: _FetchRateLimiter | None = None
        self._resident_limiter: _FetchRateLimiter | None = None
        self._interaction_hold = bool(interaction_hold)
        self._double_buffer = True
        self._dbuf = None
        self._interactive_step_rate = (
            float(interactive_step_rate)
            if interactive_step_rate and float(interactive_step_rate) > 1.0
            else None
        )
        # (weakref to volume node, saved relative_step_size) while the
        # interactive quality reduction is applied
        self._saved_step: tuple | None = None
        # interaction hold: extended by every camera/scrub event, ended
        # by the debounced _check once interaction settles
        self._hold_until = 0.0
        self._hold_s = max(0.15, 1.5 * debounce_ms / 1000.0)
        self._held_batches: list[tuple] = []
        self._last_step_change_time = 0.0
        self._step_change_min_interval = 0.2
        self._held_refresh = False
        # Level we set through layer._locked_data_level for 3D auto mode
        # (None when we are not driving the level).
        self._auto_locked: int | None = None
        # True while the user has pinned an explicit level in the
        # resolution selector; suspends 3D auto level selection.
        self._user_locked = layer.locked_data_level is not None
        self._closed = False

        self._worker = None
        self._generation = 0
        self._active: tuple | None = None
        self._chunks_done = 0
        self._chunks_total = 0
        self._last_refresh = 0.0
        self._last_refresh_duration = 0.0
        self._pbar = None
        self._resident_pbar = None
        # napari's Qt progress bar calls QApplication.processEvents() on
        # every update, which re-enters event handling *inside* the
        # caller; chunk handlers therefore only accumulate counts here
        # and a timer flushes them from the top of the event loop
        # (see _advance_progress/_flush_progress)
        self._pbar_pending = 0
        self._resident_pbar_pending = 0
        self._pbar_flushing = False
        self._pbar_scheduled = False
        self._pbar_last_flush = 0.0
        self._backdrop_pending = False
        # (level, min, max) of the last synchronous 2D backdrop, whose
        # patches left the GPU texture equal to the hyperslice; lets the
        # matching fetch pass skip its pass-start full-tile upload
        self._synced_backdrop_key: tuple | None = None

        self._resident_worker = None
        self._repair_worker = None
        self._resident_level = len(data) - 1
        self._resident_max_bytes = resident_max_bytes
        self._resident_disabled = False
        self._last_clamp_message: str | None = None

        # QTimer-based debounce: continuous camera motion only triggers a
        # fetch pass once interaction settles.  Using QTimer keeps the
        # callback on the main thread (psygnal's debounced fires from a
        # threading.Timer, which crashes PySide6 when it touches Qt
        # objects).
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(debounce_ms)
        self._debounce_timer.timeout.connect(self._check)

        def _debounced_check(event=None):
            self._debounce_timer.start()

        _debounced_check.cancel = self._debounce_timer.stop
        self._debounced_check = _debounced_check
        self._connections = [
            # fast (non-debounced) path: suspend streaming work the
            # moment interaction starts, so drag frames stay free of
            # uploads, slice cascades and fetch GIL pressure
            (viewer.camera.events, self._on_interaction),
            (viewer.dims.events.current_step, self._on_dims_step_change),
            (viewer.camera.events, self._debounced_check),
            (viewer.dims.events.current_step, self._debounced_check),
            (viewer.dims.events.ndisplay, self._debounced_check),
            (layer.events.locked_data_level, self._debounced_check),
            # the locked_data_level event only fires for user/API writes
            # (3D auto mode bypasses the setter), so it reliably tells us
            # whether the user has pinned a level
            (layer.events.locked_data_level, self._on_user_locked_change),
            (layer.events.visible, self._debounced_check),
            # set_data fires after every (re-)slice, including the ones
            # napari runs when the data level or corners change; _check is
            # a cheap no-op when the view is already covered.
            (layer.events.set_data, self._debounced_check),
            # fast (non-debounced) path: the instant napari slices a level
            # whose interval does not cover the view (i.e. the canvas just
            # went blank), put a backdrop up rather than waiting for the
            # debounced fetch pass
            (layer.events.set_data, self._on_set_data),
        ]
        for emitter, callback in self._connections:
            emitter.connect(callback)
        viewer.layers.events.removed.connect(self._on_layer_removed)
        # engage the interaction hold on the raw pointer events too:
        # a press/wheel precedes the first camera event by one event,
        # so the fetch workers pause before the gesture needs the GIL
        self._canvas_events = []
        with contextlib.suppress(AttributeError):  # headless ViewerModel
            canvas_events = viewer.window._qt_viewer.canvas.events
            for name in ('mouse_press', 'mouse_wheel'):
                emitter = getattr(canvas_events, name, None)
                if emitter is not None:
                    emitter.connect(self._on_interaction)
                    # a click without camera motion must still resume
                    # the paused workers once it settles
                    emitter.connect(self._debounced_check)
                    self._canvas_events.append(emitter)

        # 2D multiscale slicing caches a one-time materialization of the
        # thumbnail (coarsest) level, which would freeze this layer's
        # pre-load (all-zero) content. Disable it: materializing a resident
        # VirtualData is a plain memory copy.
        layer._level_materializer = None

        # Enable 3D sub-volume tiles: locking (or auto-selecting) a level
        # larger than this extent renders a view-centered tile of at most
        # this size, so even the finest levels of huge volumes are usable
        # in 3D. Bounded by the memory budget and the GL 3D texture limit.
        self._tile_extent_3d = _tile_extent_3d_for(
            data.dtype,
            min(interval_max_bytes, tile_max_bytes_3d),
        )
        layer._max_tile_extent_3d = self._tile_extent_3d
        layer._tile_max_bytes_3d = min(interval_max_bytes, tile_max_bytes_3d)
        layer._interval_max_bytes_3d = interval_max_bytes

        layer._render_margin_2d = 2.0
        self._tile_margin_3d = 1.25
        layer._tile_margin_3d = self._tile_margin_3d

        self._check()

    # -- lifecycle --

    def _on_layer_removed(self, event) -> None:
        if event.value is self._layer:
            self.close()

    def close(self) -> None:
        """Disconnect from the viewer and stop all background fetching."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            # kill any pending debounced trigger so no fetch pass can
            # start after close
            self._debounced_check.cancel()
        with contextlib.suppress(Exception):
            self._debounce_timer.timeout.disconnect(self._check)
        self._release_auto_level()
        self._cancel_active()
        from napari.experimental import _glir_metering

        _glir_metering.remove_drain_callback(self._on_uploads_drained)
        self._restore_render_quality()
        if self._dbuf is not None:
            with contextlib.suppress(Exception):
                self._dbuf.close()
            self._dbuf = None
        if self._resident_limiter is not None:
            self._resident_limiter.cancel()
            self._resident_limiter = None
        if self._resident_worker is not None:
            self._resident_worker.quit()
            self._resident_worker = None
        if self._repair_worker is not None:
            with contextlib.suppress(Exception):
                self._repair_worker.quit()
            self._repair_worker = None
        self._close_progress(self._resident_pbar)
        self._resident_pbar = None
        self._resident_pbar_pending = 0
        for emitter, callback in self._connections:
            # RuntimeError: napari emitters can fail to normalize a
            # callback while its owner is mid-teardown
            with contextlib.suppress(ValueError, TypeError, RuntimeError):
                emitter.disconnect(callback)
        self._connections = []
        for emitter in self._canvas_events:
            with contextlib.suppress(Exception):
                emitter.disconnect(self._on_interaction)
            with contextlib.suppress(Exception):
                emitter.disconnect(self._debounced_check)
        self._canvas_events = []
        with contextlib.suppress(ValueError, TypeError, RuntimeError):
            self._viewer.layers.events.removed.disconnect(
                self._on_layer_removed,
            )

    # -- view tracking --

    def _level_interval(self, level: int) -> tuple[np.ndarray, np.ndarray]:
        """Visible half-open interval for ``level``, in level coordinates.

        Displayed dimensions come from the layer's ``corner_pixels`` (which
        napari maintains in the coordinates of the current data level);
        non-displayed dimensions cover only the current dims step.
        """
        layer = self._layer
        vdata = self._data[level]
        ndim = vdata.ndim
        shape = np.asarray(vdata.shape, dtype=np.int64)

        min_coord = np.zeros(ndim, dtype=np.int64)
        max_coord = shape.copy()

        # corner_pixels bound the displayed dimensions in both 2D (the
        # visible canvas region) and 3D (the full level or a sub-volume
        # tile when _max_tile_extent_3d applies)
        displayed = set(layer._slice_input.displayed)
        corners = layer.corner_pixels
        for d in displayed:
            min_coord[d] = corners[0, d]
            max_coord[d] = corners[1, d] + 1

        self._restrict_to_current_step(level, displayed, min_coord, max_coord)

        min_coord = np.clip(min_coord, 0, shape)
        max_coord = np.clip(max_coord, 0, shape)
        return self._clamp_interval(vdata, min_coord, max_coord)

    def _restrict_to_current_step(
        self,
        level: int,
        displayed: set,
        min_coord,
        max_coord,
    ) -> None:
        """Restrict non-displayed dims to the current dims step (in place)."""
        layer = self._layer
        vdata = self._data[level]
        ndim = vdata.ndim
        factors = np.asarray(self._data._scale_factors[level])
        try:
            data_point = np.asarray(
                layer.world_to_data(self._viewer.dims.point),
                dtype=float,
            )
        except (ValueError, IndexError, TypeError):
            # pragma: no cover - layer/viewer dims mismatch fallback
            data_point = np.asarray(self._viewer.dims.point, dtype=float)[
                -ndim:
            ]
        n_point = len(data_point)
        for d in range(ndim):
            if d not in displayed:
                if d >= n_point:
                    # Dimension not tracked by viewer.dims (e.g. RGB
                    # channel): keep the full extent.
                    continue
                point = int(np.round(data_point[d] / factors[d]))
                point = min(max(point, 0), int(vdata.shape[d]) - 1)
                min_coord[d] = point
                max_coord[d] = point + 1

    def _clamp_interval(
        self,
        vdata: VirtualData,
        min_coord,
        max_coord,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shrink an interval around its center to respect the memory cap."""
        min_coord = np.array(min_coord, dtype=np.int64)
        max_coord = np.array(max_coord, dtype=np.int64)
        requested_min, requested_max = min_coord.copy(), max_coord.copy()
        itemsize = vdata.dtype.itemsize
        extent = np.maximum(max_coord - min_coord, 1)
        max_elements = self._interval_max_bytes // itemsize
        clamped = False
        while np.prod(extent, dtype=np.int64) > max_elements:
            widest = int(np.argmax(extent))
            center = (min_coord[widest] + max_coord[widest]) // 2
            half = max(extent[widest] // 4, 1)
            min_coord[widest] = max(center - half, min_coord[widest])
            max_coord[widest] = min(center + half, max_coord[widest])
            new_extent = max(max_coord[widest] - min_coord[widest], 1)
            if new_extent == extent[widest]:  # pragma: no cover - safety
                break
            extent[widest] = new_extent
            clamped = True
        if clamped:
            message = (
                f'progressive loading: visible interval '
                f'[{requested_min.tolist()}, {requested_max.tolist()}) '
                f'exceeds {self._interval_max_bytes} bytes; clamped to '
                f'[{min_coord.tolist()}, {max_coord.tolist()})'
            )
            if message != self._last_clamp_message:
                self._last_clamp_message = message
                LOGGER.warning(message)
        return min_coord, max_coord

    # -- 3D automatic level selection --

    def _on_user_locked_change(self, event=None) -> None:
        """Track explicit level pins made through the resolution selector.

        This only fires for writes through the public
        ``locked_data_level`` setter (3D auto mode writes the private
        attribute directly). The resolution selector widget may *echo*
        the auto-driven value back through the setter when its items are
        rebuilt, so a value equal to the current auto level is not
        treated as a user pin — otherwise auto mode would silently
        suspend itself after the first level change.
        """
        locked = self._layer.locked_data_level
        if locked is None:
            self._user_locked = False
        elif locked != self._auto_locked:
            self._user_locked = True
            self._auto_locked = None
        # else: an echo of the level auto mode set; keep auto mode active.

    def _zoom_target_level_3d(self) -> int:
        """Pick the 3D data level appropriate for the current camera zoom.

        Chooses the coarsest level whose voxels project to at most
        ``max_pixel_size_3d`` screen pixels (so the displayed resolution
        roughly matches the screen, like napari's 2D level selection),
        then falls back to coarser levels until the visible volume fits
        the memory budget. A camera that is not yet initialized (zoom of
        zero, NaN, or inf — e.g. before the window is first shown)
        selects the coarsest level.
        """
        layer = self._layer
        zoom = float(self._viewer.camera.zoom)
        displayed = list(layer._slice_input.displayed)
        n_levels = len(self._data)

        if not np.isfinite(zoom) or zoom <= 0:
            return n_levels - 1

        # screen pixels per level-0 data pixel (layer scale maps data to
        # world; zoom maps world to screen)
        layer_scale = float(
            np.max(np.take(np.asarray(layer.scale), displayed)),
        )
        data_zoom = zoom * layer_scale

        target = 0
        for level in range(n_levels - 1, -1, -1):
            factors = np.take(
                np.asarray(self._data._scale_factors[level]),
                displayed,
            )
            pixel_size = data_zoom * float(np.max(factors))
            if pixel_size <= self._max_pixel_size_3d:
                target = level
                break

        bbox = self._camera_bbox_level0(displayed)

        # Coarsen until the viewport tile fits the memory budget AND can
        # be fetched in a reasonable number of chunks — a fully zoomed-out
        # view of a deep pyramid would otherwise pick a level needing
        # thousands of chunks, taking minutes to sharpen. Finer levels
        # unlock progressively as zooming shrinks the viewport tile.
        for level in range(target, n_levels):
            vdata = self._data[level]
            level_extent = np.take(
                np.asarray(vdata.shape, dtype=np.int64),
                displayed,
            )
            extent = np.minimum(level_extent, self._tile_extent_3d)
            if bbox is not None:
                downsample = np.take(
                    np.asarray(self._data._scale_factors[level]),
                    displayed,
                )
                view_extent = np.ceil((bbox[1] - bbox[0]) / downsample).astype(
                    np.int64,
                )
                visible = np.minimum(
                    np.maximum(view_extent, 1),
                    level_extent,
                )
                if np.any(visible > self._tile_extent_3d):
                    # the tile cap cannot cover the canvas at this
                    # level: a finer-but-partial tile reads as the
                    # volume shrinking while you zoom in — coverage
                    # beats sharpness, prefer the next coarser level.
                    # NOTE feasibility uses the bare visible footprint:
                    # requiring the pan-slack margin too held levels
                    # back a full step (resolution arrived ~2x late);
                    # instead the slack below shrinks to whatever fits
                    # under the cap near the switch boundary and grows
                    # back as zooming shrinks the footprint
                    continue
                wanted = np.minimum(
                    np.ceil(visible * self._tile_margin_3d),
                    level_extent,
                )
                extent = np.minimum(extent, wanted.astype(np.int64))
            nbytes = np.prod(extent, dtype=np.int64) * vdata.dtype.itemsize
            chunk_shape = np.take(
                np.asarray(vdata.chunk_shape, dtype=np.int64),
                displayed,
            )
            n_chunks = np.prod(
                -(-extent // chunk_shape),
                dtype=np.int64,
            )  # ceil-div
            if (
                nbytes <= self._interval_max_bytes
                and n_chunks <= self._max_chunks_per_pass
            ):
                return level
        return n_levels - 1

    def _camera_bbox_level0(self, displayed_axes) -> np.ndarray | None:
        """Approximate visible bbox around the camera, in level-0 coords.

        Sized from the canvas dimensions and zoom so 3D sub-volume tiles
        cover (roughly) what is on screen rather than the whole memory
        budget. In 3D the depth axis (along the view direction) gets a
        smaller extent than the screen-plane axes so the tile tracks the
        rotated frustum.
        """
        camera = self._viewer.camera
        camera_center = np.asarray(camera.center, dtype=float)
        if not np.all(np.isfinite(camera_center)):
            return None
        try:
            world_point = np.array(self._viewer.dims.point, dtype=float)
            world_point[list(displayed_axes)] = camera_center[
                -len(displayed_axes) :
            ]
            data_point = np.asarray(
                self._layer.world_to_data(world_point),
                dtype=float,
            )
        except Exception:  # pragma: no cover - dims mismatch  # noqa: BLE001
            return None
        center = data_point[list(displayed_axes)]
        if not np.all(np.isfinite(center)):
            return None
        zoom = float(camera.zoom)
        try:
            canvas_size = max(self._viewer._canvas_size)
        except Exception:  # pragma: no cover - headless  # noqa: BLE001
            canvas_size = 800
        if not np.isfinite(zoom) or zoom <= 0 or canvas_size <= 0:
            return np.stack([center, center])
        layer_scale = np.take(
            np.asarray(self._layer.scale, dtype=float),
            list(displayed_axes),
        )
        screen_half = (canvas_size / zoom) / 2 / np.maximum(layer_scale, 1e-12)
        # In 3D, project the view direction onto data axes and shrink
        # the depth axis so the bbox tracks the rotated frustum instead
        # of always being a uniform cube.
        half_extent = screen_half.copy()
        if self._viewer.dims.ndisplay == 3:
            try:
                view_dir_world = np.asarray(
                    camera.view_direction, dtype=float
                )[-len(displayed_axes) :]
                view_dir_data = np.abs(view_dir_world) / np.maximum(
                    layer_scale, 1e-12
                )
                view_dir_data /= np.maximum(
                    np.linalg.norm(view_dir_data), 1e-12
                )
                # screen-plane axes get full extent; depth axis gets a
                # fraction proportional to its alignment with view dir
                depth_fraction = np.maximum(view_dir_data, 0.1)
                half_extent = screen_half * np.maximum(
                    1.0 - 0.7 * depth_fraction, 0.3
                )
            except Exception:  # noqa: BLE001
                pass
        return np.stack([center - half_extent, center + half_extent])

    def _apply_auto_level(self) -> None:
        """Drive the layer's data level from zoom while in 3D Auto mode.

        Writes ``layer._locked_data_level`` directly (not the public
        setter) so no ``locked_data_level`` event is emitted and the
        resolution selector keeps displaying "Auto".
        """
        layer = self._layer
        if not self._auto_level_3d or self._user_locked:
            return
        if self._viewer.dims.ndisplay != 3:
            self._release_auto_level()
            return
        displayed_axes = layer._slice_input.displayed
        camera_bbox = self._camera_bbox_level0(displayed_axes)
        if camera_bbox is None:
            # camera not in a usable state (e.g. before the first draw):
            # without a viewport bbox the tile would fall back to the full
            # memory-budget cube, whose synchronous slice can stall the UI
            # for seconds; wait for a valid camera instead
            return
        target = self._zoom_target_level_3d()
        if target == self._auto_locked and layer._locked_data_level == target:
            return
        self._auto_locked = target
        layer._locked_data_level = target
        layer._data_level = target
        # Mirror the corner_pixels update of the locked_data_level setter,
        # centering any sub-volume tile on the camera.
        corners_fn = getattr(layer, '_corners_for_locked_level', None)
        if corners_fn is not None:
            view_dir = getattr(layer, '_view_direction_data', lambda _: None)(
                displayed_axes
            )
            layer.corner_pixels = corners_fn(
                target,
                displayed_axes,
                camera_bbox,
                view_dir,
            )
        else:
            shape_at_level = np.take(
                np.asarray(layer.level_shapes[target]), displayed_axes
            )
            tile_cap = getattr(layer, '_max_tile_extent_3d', None)
            if tile_cap is not None:
                tile_extent = np.minimum(shape_at_level, tile_cap)
            else:
                tile_extent = shape_at_level
            corners = np.zeros((2, layer.ndim), dtype=int)
            if camera_bbox is not None and np.all(np.isfinite(camera_bbox)):
                downsample = np.take(
                    np.asarray(layer.downsample_factors[target]),
                    displayed_axes,
                )
                center = camera_bbox.mean(axis=0).astype(float) / downsample
                center = np.clip(center, 0, shape_at_level - 1)
                low = np.clip(
                    (center - tile_extent / 2).astype(int),
                    0,
                    shape_at_level - tile_extent,
                )
            else:
                low = np.clip(
                    ((shape_at_level - tile_extent) // 2).astype(int),
                    0,
                    shape_at_level - tile_extent,
                )
            high = low + tile_extent
            corners[0, displayed_axes] = low
            corners[1, displayed_axes] = high - 1
            layer.corner_pixels = corners
        # Prepare the new level's interval with a backdrop from the level
        # that was just displayed BEFORE napari re-slices, so the previous
        # resolution stays on screen until new chunks replace it.
        min_coord, max_coord = self._level_interval(target)
        if not np.any(max_coord <= min_coord):
            self._data.set_interval(
                target,
                min_coord,
                max_coord,
                backdrop_level=self._backdrop_level(
                    target,
                    min_coord,
                    max_coord,
                ),
            )
        # Attach the double buffer BEFORE the level-switch re-slice: the
        # FIRST switch of a session otherwise goes through vanilla
        # set_data, where the node matrix applies instantly while the
        # metered content upload lands frames later — old content under
        # the new transform, a one-time visible jump.
        if self._double_buffer and self._viewer.dims.ndisplay == 3:
            node = self._get_display_node()
            if (
                node is not None
                and getattr(node, '_texture', None) is not None
            ):
                with contextlib.suppress(Exception):
                    self._ensure_dbuf(node)
        layer.refresh(extent=False)

    def _release_auto_level(self) -> None:
        """Give level control back to napari (e.g. on 2D or teardown)."""
        layer = self._layer
        if (
            self._auto_locked is not None
            and not self._user_locked
            and layer._locked_data_level == self._auto_locked
        ):
            layer._locked_data_level = None
            layer._reset_data_level()
            # _reset_data_level cleared the lock state; preserve our flag
            self._auto_locked = None
            # the handover re-slice is a one-time full-pipeline refresh
            # (~300ms on large 2D data): run it as its own event instead
            # of inside whatever handler triggered the release
            QTimer.singleShot(
                0,
                lambda: (
                    None if self._closed else self._layer.refresh(extent=False)
                ),
            )
        else:
            self._auto_locked = None

    # -- fetch passes --

    @property
    def _holding(self) -> bool:
        return time.monotonic() < self._hold_until

    def _on_interaction(self, event=None) -> None:
        """Suspend streaming work while the user interacts.

        Fires on every camera/scrub event. The hold window is extended
        each time; the debounced ``_check`` (which fires once
        interaction settles) ends it. Rate limiting alone does not keep
        interaction smooth on slow GL drivers — any upload or slice
        cascade landing in a drag frame stalls it — so during the hold
        nothing lands at all: fetch workers pause, arriving batches are
        buffered, throttled refreshes defer, and the GLIR upload meter
        holds its carry.
        """
        if self._closed or not self._interaction_hold:
            return
        first = not self._holding
        self._hold_until = time.monotonic() + self._hold_s
        if first:
            for limiter in (self._limiter, self._resident_limiter):
                if limiter is not None:
                    limiter.pause()
            self._degrade_render_quality()
        from napari.experimental import _glir_metering

        if _glir_metering.is_installed():
            _glir_metering.hold_uploads_until(self._hold_until)

    def _on_dims_step_change(self, event=None) -> None:
        """Handle a non-displayed dimension change (e.g. time step).

        Unlike camera interaction, each step is a discrete new view.
        Hold the double buffer so the old frame stays visible through
        the re-slice, cancel the old fetch, and start loading the new
        time step immediately.
        """
        if self._closed:
            return
        if self._holding:
            return
        now = time.monotonic()
        if now - self._last_step_change_time < self._step_change_min_interval:
            return
        self._last_step_change_time = now
        # Hold the front buffer *before* anything else so the upcoming
        # re-slice (napari fires set_data on the same event) cannot
        # present zeros — the old tile keeps rendering until the new
        # time point's data arrives and explicitly releases.
        dbuf = self._dbuf3d()
        if dbuf is not None:
            dbuf.hold_presents()
        self._cancel_active()
        self._degrade_render_quality()
        self._check()

    def _degrade_render_quality(self) -> None:
        """Coarsen the raycast step while frames must stay cheap.

        Raycast cost scales inversely with the step size; on a
        saturated GPU the deep command queue behind expensive frames is
        what blocks the main thread in otherwise-cheap GL calls
        (glBufferSubData, glFlush). Applied during interaction AND
        while a texture upload backlog is draining (level switches
        stage a full tile, which the GLIR meter spreads over many
        frames — those frames must be cheap or the drain takes tens of
        seconds). Quality is restored by :meth:`_maybe_restore_quality`
        once interaction has settled and the backlog is gone.
        """
        if self._interactive_step_rate is None or self._saved_step is not None:
            return
        if self._viewer.dims.ndisplay != 3:
            return
        node = self._get_volume_node()
        if node is None:
            return
        try:
            saved = float(node.relative_step_size)
            node.relative_step_size = saved * self._interactive_step_rate
        except Exception:  # noqa: BLE001 # pragma: no cover - node variant
            return
        self._saved_step = (weakref.ref(node), saved)
        # restore is event-driven, not polled: checked at hold end, per
        # delivered batch (_update_node), at pass end, and when the
        # GLIR meter reports its carry fully drained
        from napari.experimental import _glir_metering

        _glir_metering.add_drain_callback(self._on_uploads_drained)

    def _on_uploads_drained(self) -> None:
        """GLIR carry fully drained: swap pending textures, restore LOD."""
        if self._closed:
            return
        if self._dbuf is not None:
            node = self._get_display_node()
            if (
                node is not None
                and self._dbuf.matches(node)
                and self._dbuf.dirty
            ):
                with contextlib.suppress(Exception):
                    self._dbuf.present()
        self._maybe_restore_quality()

    def _upload_backlog_bytes(self) -> int:
        from napari.experimental import _glir_metering

        if not _glir_metering.is_installed():
            return 0
        return _glir_metering.pending_upload_bytes()

    def _maybe_restore_quality(self) -> None:
        """Restore full render quality once frames can afford it."""
        if self._saved_step is None:
            return
        if self._holding and not self._closed:
            return
        from napari.experimental import _glir_metering

        if not self._closed and (
            self._upload_backlog_bytes()
            > _glir_metering.DEFAULT_FRAME_BUDGET_BYTES
        ):
            return  # still draining a big upload; keep frames cheap
        self._restore_render_quality()

    def _restore_render_quality(self) -> None:
        if self._saved_step is None:
            return
        node_ref, saved = self._saved_step
        self._saved_step = None
        node = node_ref()
        if node is None:
            return
        with contextlib.suppress(Exception):  # pragma: no cover - GL
            node.relative_step_size = saved
            node.update()

    def _end_hold(self) -> None:
        self._hold_until = 0.0
        # quality restores via the poll once the upload backlog (e.g.
        # batches buffered during the drag, or a level switch's full
        # tile) has drained — not the instant the pointer stops
        self._maybe_restore_quality()
        for limiter in (self._limiter, self._resident_limiter):
            if limiter is not None:
                limiter.resume()
        if self._held_batches:
            held, self._held_batches = self._held_batches, []
            for generation, vdata, batch in held:
                # stale generations are dropped inside _on_chunks
                self._on_chunks(generation, vdata, batch)
        if self._held_refresh:
            self._held_refresh = False
            self._refresh()

    def _check(self, event=None) -> None:
        """Start a fetch pass if the current view is not fully loaded."""
        if self._closed:
            return
        self._end_hold()
        layer = self._layer
        if not layer.visible:
            self._cancel_active()
            return
        self._apply_auto_level()
        self._ensure_resident()
        level = int(layer.data_level)
        min_coord, max_coord = self._level_interval(level)
        if np.any(max_coord <= min_coord):
            return
        if level == self._resident_level and self._resident_worker is not None:
            # The coarsest level is being filled by the resident worker.
            return
        view_key = (level, tuple(min_coord), tuple(max_coord))
        if view_key == self._active:
            # A pass for exactly this view is in flight or already done.
            if self._needs_final_reconcile and self._worker is None:
                self._needs_final_reconcile = False
                if self._dbuf is not None:
                    # every chunk of the pass was patched, so the GPU
                    # texture already matches what this refresh would
                    # upload — skip the redundant full-tile upload and
                    # let the refresh only fix slice state/thumbnail
                    self._dbuf.suppress_next_full_upload()
                self._refresh(final=True, force=True)
            return
        self._start_fetch(level, min_coord, max_coord)

    def _backdrop_level(self, level: int, min_coord, max_coord) -> int | None:
        """Pick the best level to initialize newly exposed regions from.

        Prefers the level closest in resolution to ``level`` whose resident
        data fully covers the requested region — usually the level that was
        just displayed, so a resolution switch keeps the previous data on
        screen until the new chunks arrive. Falls back to the coarsest
        level with any loaded data.
        """
        n_levels = len(self._data)
        factors = self._data._scale_factors
        ndim = self._data.ndim
        candidates = sorted(
            (c for c in range(n_levels) if c != level),
            key=lambda c: (abs(c - level), c),
        )
        fallback = None
        for cand in candidates:
            src = self._data[cand]
            if not src.loaded_chunks:
                continue
            cand_min = [
                int(
                    np.floor(
                        min_coord[d] * factors[level][d] / factors[cand][d],
                    ),
                )
                for d in range(ndim)
            ]
            cand_max = [
                int(
                    np.ceil(
                        max_coord[d] * factors[level][d] / factors[cand][d],
                    ),
                )
                for d in range(ndim)
            ]
            cand_min = np.clip(cand_min, 0, src.shape)
            cand_max = np.clip(cand_max, 0, src.shape)
            if src.covers(cand_min, cand_max):
                return cand
            if cand == n_levels - 1:
                fallback = cand
        return fallback

    def _on_set_data(self, event=None) -> None:
        """Fast path keeping the canvas filled across level switches.

        napari re-slices immediately when it changes the data level, often
        before any fetch pass has prepared that level's interval — the
        slice then materializes zeros. Detect that here and schedule a
        backdrop (outside the event emission stack) so at most one blank
        frame is shown.
        """
        if self._closed or not self._layer.visible:
            return
        dbuf = self._dbuf3d()
        if dbuf is not None:
            # napari's vispy layer (connected before this handler) just
            # staged the new tile AND applied its matrix; capture the
            # matrix as pending and restore the front-matching one so
            # the still-rendered old tile never draws misplaced. Runs
            # inside the emission — nothing can paint in between.
            dbuf.capture_transform()
        if self._backdrop_pending:
            return
        level = int(self._layer.data_level)
        min_coord, max_coord = self._level_interval(level)
        if np.any(max_coord <= min_coord):
            return
        if self._data[level].covers(min_coord, max_coord):
            return
        if self._sync_backdrop_2d(level, min_coord, max_coord):
            return
        self._backdrop_pending = True
        QTimer.singleShot(0, self._prepare_backdrop)

    def _sync_backdrop_2d(self, level, min_coord, max_coord) -> bool:
        """Backdrop a just-sliced empty 2D level before it can paint.

        A level switch slices the new level's VirtualData before any
        pass has prepared its interval, so the node receives zeros; the
        deferred ``_prepare_backdrop`` refresh races the paint event and
        sometimes loses — a visible black flash between scales. Here,
        still inside the ``set_data`` emission (so before any paint),
        the interval is filled from a coarser level (a memory copy —
        cheap for a 2D view region) and the texture is patched directly,
        skipping the slicing pipeline, which must not re-enter. Returns
        False (caller falls back to the deferred path) when the GPU
        patch cannot be validated, e.g. in 3D where this path is not
        used.
        """
        if self._viewer.dims.ndisplay != 2:
            return False
        vdata = self._data[level]
        if vdata.ndim < 2:
            return False
        # Only the on-screen region is prepared here — the slice (and
        # texture) extend a render margin beyond the viewport, and an
        # upsampling gather plus pixel copy of the full margin-sized
        # tile costs tens of ms on the main thread. The margin starts
        # as (invisible) zeros and is repaired off-thread below.
        view_min, view_max = self._viewport_box_2d(level, min_coord, max_coord)
        try:
            self._data.set_interval(level, min_coord, max_coord)
            self._backdrop_fill_layered(level, view_min, view_max)
        except Exception:  # noqa: BLE001 # pragma: no cover - degenerate
            return False
        patched = self._patch_texture_region(
            vdata,
            [int(c) for c in view_min],
            [int(c) for c in view_max],
        )
        if not patched:
            # validation failed (e.g. cold start: the node's texture
            # re-spec is still deferred to its first draw, so shapes
            # don't match) — push the backdrop crop through set_data,
            # which handles the re-spec itself
            patched = self._set_node_data_from_hyperslice(vdata)
        if patched:
            # writes staged into the back texture; swap it in now so
            # the very next paint shows the backdrop, not the zeros
            self._update_node()
            # the texture now equals the hyperslice; if the fetch pass
            # starts on this same interval, its forced backdrop refresh
            # may skip the redundant full-tile upload
            self._synced_backdrop_key = (
                level,
                tuple(int(c) for c in min_coord),
                tuple(int(c) for c in max_coord),
            )
            # backfill the off-screen margin on a worker thread
            self._repair_backdrop()
        return patched

    def _backdrop_fill_layered(self, level, lo, hi) -> bool:
        """Fill unloaded regions of ``[lo, hi)`` from every level with
        useful resident data, coarsest first so finer sources overwrite
        their overlap.

        Unlike :meth:`_backdrop_level` (which requires one fully
        covering source), this reuses *partially* covering levels —
        including levels finer than the target, so zooming back out
        reuses already-fetched detail instead of restarting from the
        coarsest level.
        """
        n_levels = len(self._data)
        factors = self._data._scale_factors
        ndim = self._data.ndim
        # single-cover shortcut: when the closest-resolution source
        # fully covers the region, one gather suffices — layering
        # coarse->fine would run one gather per level with data, all
        # but the last overwritten
        full = self._backdrop_level(level, lo, hi)
        if full is not None and full != level:
            src = self._data[full]
            cand_lo = [
                int(np.floor(lo[d] * factors[level][d] / factors[full][d]))
                for d in range(ndim)
            ]
            cand_hi = [
                int(np.ceil(hi[d] * factors[level][d] / factors[full][d]))
                for d in range(ndim)
            ]
            if src.covers(
                np.clip(cand_lo, 0, src.shape),
                np.clip(cand_hi, 0, src.shape),
            ):
                with contextlib.suppress(Exception):
                    return bool(
                        self._data.fill_unloaded_from(
                            level,
                            full,
                            region=(list(lo), list(hi)),
                        ),
                    )
        wrote = False
        # high index = coarse; iterate coarse -> fine, skipping target
        for cand in range(n_levels - 1, -1, -1):
            if cand == level:
                continue
            src = self._data[cand]
            if not src.loaded_chunks:
                continue
            with src.lock:
                if src._min_coord is None:
                    continue
                src_min = [int(c) for c in src._min_coord]
                src_max = [int(c) for c in src._max_coord]
            # source interval expressed in target-level coordinates
            ratio = [factors[cand][d] / factors[level][d] for d in range(ndim)]
            region_lo = [
                max(int(lo[d]), int(np.ceil(src_min[d] * ratio[d])))
                for d in range(ndim)
            ]
            region_hi = [
                min(int(hi[d]), int(np.floor(src_max[d] * ratio[d])))
                for d in range(ndim)
            ]
            if any(b <= a for a, b in zip(region_lo, region_hi, strict=True)):
                continue
            with contextlib.suppress(Exception):
                wrote = (
                    self._data.fill_unloaded_from(
                        level,
                        cand,
                        region=(region_lo, region_hi),
                    )
                    or wrote
                )
        return wrote

    def _viewport_box_2d(self, level, min_coord, max_coord):
        """The visible (no-margin) region of ``level``, clamped to the
        given interval. Non-displayed dims keep the interval's bounds
        (the current-step slab). Falls back to the full interval when no
        viewport bbox has been recorded yet."""
        lo = [int(c) for c in min_coord]
        hi = [int(c) for c in max_coord]
        bbox = getattr(self._layer, '_last_data_bbox', None)
        displayed = tuple(self._layer._slice_input.displayed)
        if bbox is None or bbox[0] != displayed or len(displayed) != 2:
            return lo, hi
        factors = np.take(
            np.asarray(self._data._scale_factors[level], dtype=float),
            list(displayed),
        )
        corners = np.asarray(bbox[1], dtype=float) / factors
        for i, d in enumerate(displayed):
            view_lo = int(max(np.floor(corners[0, i]), lo[d]))
            view_hi = int(min(np.ceil(corners[1, i]) + 1, hi[d]))
            if view_hi <= view_lo:
                return [int(c) for c in min_coord], [int(c) for c in max_coord]
            lo[d], hi[d] = view_lo, view_hi
        return lo, hi

    def _set_node_data_from_hyperslice(self, vdata: VirtualData) -> bool:
        """Replace the node's data with the current corner-pixels crop
        (at the current step of every non-displayed dim)."""
        node = self._get_display_node(2)
        if node is None:
            return False
        plane = self._displayed_plane(int(self._layer.data_level))
        if plane is None:
            return False
        displayed, steps = plane
        if len(displayed) != 2:
            return False
        corners = self._layer.corner_pixels
        translate = vdata.translate
        displayed_set = set(displayed)
        src = tuple(
            steps[d] - int(translate[d])
            if d in steps
            else slice(
                int(corners[0, d]) - int(translate[d]),
                int(corners[1, d]) + 1 - int(translate[d]),
            )
            if d in displayed_set
            else slice(0, int(vdata.shape[d]))
            for d in range(vdata.ndim)
        )
        try:
            from napari._vispy.utils.gl import fix_data_dtype

            with vdata.lock:
                crop = np.ascontiguousarray(vdata.hyperslice[src])
            if crop.ndim not in (2, 3) or 0 in crop.shape:
                return False
            node.set_data(fix_data_dtype(crop))
        except Exception:  # noqa: BLE001 # pragma: no cover - GL mismatch
            return False
        return True

    def _prepare_backdrop(self) -> None:
        self._backdrop_pending = False
        if self._closed or not self._layer.visible:
            return
        level = int(self._layer.data_level)
        min_coord, max_coord = self._level_interval(level)
        if np.any(max_coord <= min_coord):
            return
        if self._data[level].covers(min_coord, max_coord):
            return
        # Set the interval cheap (carry-over + zeros): the full-tile
        # upsample gather scales with the tile cap (200ms+ on the main
        # thread at 33 MB), and a fetch pass with this same shape
        # usually follows anyway. 2D fills the visible region
        # synchronously (bounded work; usually a no-op right after
        # _sync_backdrop_2d); the off-screen margin fills on the repair
        # worker. In 3D carried-over content plus the double buffer
        # keep the canvas filled until repaired content presents.
        self._data.set_interval(level, min_coord, max_coord)
        if self._viewer.dims.ndisplay == 2:
            with contextlib.suppress(Exception):
                self._backdrop_fill_layered(
                    level,
                    [int(c) for c in min_coord],
                    [int(c) for c in max_coord],
                )
        dbuf = self._dbuf3d()
        if dbuf is not None:
            # the refresh below stages zeros + carry-over; keep the
            # front on screen until the repair worker lands content
            dbuf.hold_presents()
        self._repair_backdrop()
        self._refresh(force=True)

    def _start_fetch(self, level: int, min_coord, max_coord) -> None:
        self._cancel_active()
        vdata = self._data[level]

        # Set the interval cheap (carry-over + zeros): the backdrop
        # upsample gather is too much main-thread time at pass start
        # (the whole 16 MB tile in 3D; interval-sized regions in 2D).
        # 2D fills the visible region synchronously (bounded work, and
        # usually already done by _sync_backdrop_2d — refilling skips
        # nothing visible since fills are idempotent over unloaded
        # chunks); everything else fills on the repair worker. In 3D
        # the double buffer keeps rendering the previous tile until the
        # filled content presents.
        self._data.set_interval(level, min_coord, max_coord)
        if self._viewer.dims.ndisplay == 2:
            with contextlib.suppress(Exception):
                self._backdrop_fill_layered(
                    level,
                    [int(c) for c in min_coord],
                    [int(c) for c in max_coord],
                )
        self._repair_backdrop()
        self._active = (level, tuple(min_coord), tuple(max_coord))

        interval = vdata.interval
        keys = chunk_slices(vdata, interval=interval)
        if self._viewer.dims.ndisplay == 3:
            queue = self._prioritize_3d(level, keys, interval)
        else:
            queue = chunk_priority_2D(keys, interval[0], interval[1])
        queue = [
            key for key in queue if _chunk_id(key) not in vdata.loaded_chunks
        ]

        if not queue:
            # Everything visible is already resident (e.g. carried over
            # from the previous interval); make sure the canvas shows it.
            self._refresh(final=True)
            # In 3D the refresh triggers an async re-slice that stages
            # data into the back buffer via set_data_staged.  Without a
            # fetch worker there is no _on_fetch_finished to present the
            # staged content.  Register the GLIR drain callback so the
            # back buffer swaps to front once its uploads complete.
            dbuf = self._dbuf3d()
            if dbuf is not None:
                dbuf.release_presents()
                from napari.experimental import _glir_metering

                _glir_metering.add_drain_callback(self._on_uploads_drained)
            return

        LOGGER.debug(
            'starting fetch pass: level=%d interval=%s chunks=%d',
            level,
            interval,
            len(queue),
        )

        self._generation += 1
        generation = self._generation
        self._chunks_done = 0
        self._chunks_total = len(queue)
        self._pass_all_patched = True
        self._needs_final_reconcile = False
        if self._dbuf is not None:
            # a new pass invalidates any pending "texture already
            # matches" assertion from a previous reconcile
            self._dbuf._suppress_full = False
        self._pbar = self._make_progress(
            len(queue),
            f'{self._layer.name}: loading level {level}',
        )

        # Attach the texture double buffer BEFORE the backdrop refresh
        # below, so the pass's first full-tile upload (and any tile
        # reallocation) is staged off the rendered path instead of
        # re-specifying the bound texture in place.
        if self._double_buffer:
            node = self._get_display_node()
            if (
                node is not None
                and getattr(node, '_texture', None) is not None
            ):
                with contextlib.suppress(Exception):
                    self._ensure_dbuf(node)

        # Show carried-over and backdrop content before the first chunk
        # arrives so the canvas is never empty while fetching. This is
        # a full-tile texture upload: keep frames cheap while the GLIR
        # meter drains it (quality restores once the backlog is gone).
        # When the synchronous backdrop already patched this exact
        # interval, the GPU texture equals the hyperslice (patches and
        # full refreshes both preserve that) — skip the redundant
        # full-tile upload; the refresh still fixes slice state.
        if self._dbuf is not None and self._synced_backdrop_key == (
            level,
            tuple(min_coord),
            tuple(max_coord),
        ):
            self._dbuf.suppress_next_full_upload()
        self._synced_backdrop_key = None
        dbuf = self._dbuf3d()
        if dbuf is not None:
            # this pass-start rewrite is zeros + carry-over until the
            # repair worker lands its backdrop; the front (previous
            # tile, correctly placed via the transform hold) renders
            # meanwhile
            dbuf.hold_presents()
        self._degrade_render_quality()
        self._refresh(force=True)

        def apply(chunk_key, chunk, vdata=vdata):
            # worker thread: lock-guarded numpy writes; the main thread
            # only handles GPU patching and bookkeeping per batch
            vdata.set_offset(chunk_key, chunk)
            vdata.loaded_chunks.add(_chunk_id(chunk_key))

        def pack(keys, vdata=vdata):
            # worker thread: precompute the contiguous union-region block
            # for the coalesced GPU upload, so the main thread never
            # copies pixel data
            return keys, _pack_upload_block(vdata, keys)

        use_pack = (
            self._texture_patching
            and self._viewer.dims.ndisplay in (2, 3)
            and vdata.ndim >= self._viewer.dims.ndisplay
        )
        self._limiter = self._make_limiter()
        worker = _fetch_chunks(
            vdata.array,
            queue,
            num_workers=self._fetch_workers,
            apply=apply,
            pack=pack if use_pack else None,
            limiter=self._limiter,
        )
        worker.yielded.connect(
            lambda batch: self._on_chunks(generation, vdata, batch),
        )
        worker.finished.connect(lambda: self._on_fetch_finished(generation))
        self._worker = worker
        worker.start()

    def _prioritize_3d(self, level, keys, interval):
        camera = self._viewer.camera
        factors = np.asarray(self._data._scale_factors[level])
        displayed = list(self._layer._slice_input.displayed)[-3:]
        if len(displayed) < 3:
            # mid ndisplay transition: displayed dims not 3D yet
            return chunk_priority_2D(keys, interval[0], interval[1])
        layer_scale = np.take(
            np.asarray(self._layer.scale, dtype=float),
            displayed,
        )
        camera_center = np.asarray(camera.center, dtype=float) / (
            np.take(factors, displayed) * np.maximum(layer_scale, 1e-12)
        )
        # chunk_priority_3D sanitizes degenerate camera state internally
        return chunk_priority_3D(
            keys,
            interval[0],
            interval[1],
            camera_center=camera_center,
            view_direction=camera.view_direction,
        )

    def _make_progress(self, total: int, description: str):
        """Best-effort progress bar shown in the napari activity dock.

        Disabled in 3D: the chunk fill-in is visible feedback there, and
        Qt progress updates call processEvents(), which costs main-thread
        time precisely when streaming is busiest.
        """
        if total < PROGRESS_MIN_CHUNKS:
            return None
        if self._viewer.dims.ndisplay == 3:
            return None
        try:
            return progress(total=total, desc=description)
        except Exception:  # noqa: BLE001 # pragma: no cover - cosmetic
            return None

    @staticmethod
    def _close_progress(pbar) -> None:
        if pbar is not None:
            with contextlib.suppress(Exception):
                pbar.close()

    def _advance_progress(
        self,
        count: int = 1,
        resident: bool = False,
    ) -> None:
        """Accumulate progress for the timer-driven flush.

        ``QtLabeledProgressBar.setValue`` runs ``processEvents()``;
        calling it from a chunk handler re-enters the event loop *inside*
        that handler, nesting queued chunk deliveries and repaints there
        (the dominant 2D streaming stall on slow GL stacks). Chunk
        handlers only increment counters; the Qt update runs from a
        timer at the top of the event loop.
        """
        if resident:
            self._resident_pbar_pending += count
        else:
            self._pbar_pending += count
        if self._closed or self._pbar_scheduled:
            return
        now = time.monotonic()
        if now - self._pbar_last_flush < 0.2:
            # ~5 flushes/s: the next batch after the window flushes,
            # and the pass-end close covers the tail
            return
        self._pbar_last_flush = now
        self._pbar_scheduled = True
        try:
            # zero-delay one-shot: fires on the next event-loop pass
            # (outside any chunk handler) and self-disposes, so nothing
            # outlives the loader
            QTimer.singleShot(0, self._flush_progress)
        except Exception:  # noqa: BLE001 # pragma: no cover - no Qt
            self._pbar_scheduled = False

    def _flush_progress(self) -> None:
        """Deferred callback: push accumulated counts to the Qt bars."""
        self._pbar_scheduled = False
        if self._closed or self._pbar_flushing:
            return
        self._pbar_flushing = True
        try:
            for pbar, attr in (
                (self._pbar, '_pbar_pending'),
                (self._resident_pbar, '_resident_pbar_pending'),
            ):
                count = getattr(self, attr)
                if count:
                    setattr(self, attr, 0)
                    if pbar is not None:
                        with contextlib.suppress(Exception):
                            pbar.update(count)
        finally:
            self._pbar_flushing = False

    def _make_limiter(self) -> _FetchRateLimiter:
        limiter = _FetchRateLimiter(self._max_bytes_per_second)
        if self._holding:
            limiter.pause()
        return limiter

    def _cancel_active(self) -> None:
        self._generation += 1
        self._held_batches.clear()  # all stale now
        if self._limiter is not None:
            # wake workers sleeping on rate pacing so the pass winds
            # down promptly
            self._limiter.cancel()
            self._limiter = None
        if self._worker is not None:
            self._worker.quit()
            self._worker = None
        self._active = None
        self._close_progress(self._pbar)
        self._pbar = None
        self._pbar_pending = 0

    def _on_chunks(self, generation: int, vdata: VirtualData, batch) -> None:
        """Handle a batch of fetched chunks (already applied off-thread)."""
        if generation != self._generation or self._closed:
            return
        if self._holding:
            # interaction in progress: no GPU patches, no refreshes, no
            # progress churn; _end_hold replays these once it settles
            self._held_batches.append((generation, vdata, batch))
            return
        block = None
        if isinstance(batch, tuple):
            batch, block = batch
        self._chunks_done += len(batch)
        if self._pbar is not None:
            self._advance_progress(len(batch))
        final = self._chunks_done >= self._chunks_total
        if final:
            self._close_progress(self._pbar)
            self._pbar = None
        # While the pass is streaming, patched chunks keep the GPU texture
        # identical to what a pipeline refresh would produce: each batch
        # is one coalesced partial upload, and the only full re-slice +
        # re-upload runs once per pass — at its start (backdrop) and,
        # deferred to idle, after its end. Mid-pass full uploads were the
        # main remaining UI stalls on slow GL drivers.
        patched = self._texture_patching and self._patch_texture_batch(
            vdata,
            batch,
            block=block,
        )
        if not patched:
            self._pass_all_patched = False
            self._refresh(final=final)
            return
        now = time.monotonic()
        if final:
            self._update_node()
            if self._pass_all_patched:
                # The texture already shows every chunk. Defer the full
                # consistency refresh (thumbnail, slice state) to the
                # next interaction, where it folds into the pass-start
                # refresh instead of stalling the moment loading ends.
                self._needs_final_reconcile = True
            else:
                self._refresh(final=True)
        elif now - self._last_node_update >= max(
            self._refresh_interval_s,
            0.05,
        ):
            self._last_node_update = now
            self._update_node()

    def _on_fetch_finished(self, generation: int) -> None:
        if generation != self._generation or self._closed:
            return
        self._worker = None
        dbuf = self._dbuf3d()
        if dbuf is not None:
            # belt for the present veto: the pass is over, so whatever
            # is staged (chunks, backdrop) is the best content there is
            dbuf.release_presents()
            with contextlib.suppress(Exception):
                dbuf.present()
        self._maybe_restore_quality()

    def _get_display_node(self, ndisplay: int | None = None):
        try:
            visual = self._viewer.window._qt_viewer.layer_to_visual[
                self._layer
            ]
            if ndisplay is None:
                ndisplay = self._viewer.dims.ndisplay
            return visual._layer_node.get_node(ndisplay)
        except (KeyError, AttributeError, RuntimeError):  # pragma: no cover
            return None

    def _get_volume_node(self):
        return self._get_display_node(3)

    def _dbuf3d(self) -> DoubleBufferedVolumeTexture | None:
        """The 3D double buffer, when that is what is attached."""
        dbuf = self._dbuf
        if isinstance(dbuf, DoubleBufferedVolumeTexture):
            return dbuf
        return None

    def _patch_texture(self, vdata: VirtualData, chunk_key) -> bool:
        """Write one chunk's region into the existing GPU texture."""
        low = [int(sl.start) for sl in chunk_key]
        high = [int(sl.stop) for sl in chunk_key]
        return self._patch_texture_region(vdata, low, high)

    def _displayed_plane(self, level: int):
        """The displayed dims and the fixed level-coordinate steps of
        every other dim for the current slice.

        Returns ``(displayed, steps)`` where ``steps`` maps each
        non-displayed dim to its integer position at ``level`` (the
        same rounding the slicing pipeline applies), or ``None`` when
        the rendered slice is not a plain single plane that patches can
        represent: transposed display order, thick-slice projections,
        or an unusable slice point.
        """
        layer = self._layer
        displayed = list(layer._slice_input.displayed)
        if displayed != sorted(displayed):
            return None
        ndim = self._data.ndim
        try:
            data_slice = layer._data_slice
            factors = np.asarray(
                layer.downsample_factors[level],
                dtype=float,
            )
        except Exception:  # noqa: BLE001 # pragma: no cover - no slice yet
            return None
        steps: dict[int, int] = {}
        n_slice = len(data_slice.point)
        for d in range(ndim):
            if d in displayed:
                continue
            if d >= n_slice:
                # Dimension not tracked by the slicer (e.g. RGB
                # channel): fully present in both hyperslice and
                # texture — skip, don't step.
                continue
            point = data_slice.point[d]
            if not np.isfinite(point):
                return None
            for margin in (
                data_slice.margin_left[d],
                data_slice.margin_right[d],
            ):
                if np.isfinite(margin) and margin:
                    # thick-slice projection: the rendered plane is a
                    # reduction over a slab, not a copyable plane
                    return None
            steps[d] = int(np.round(float(point) / factors[d]))
        return displayed, steps

    def _patch_texture_batch(
        self,
        vdata: VirtualData,
        batch,
        block=None,
    ) -> bool:
        """Upload a batch of chunks as ONE coalesced texture update.

        Uploads the union bounding box of the batch from the hyperslice
        in a single partial transfer — front-to-back ordering makes
        batches spatially coherent, and many small GL calls cost more
        than one slightly larger one. Voxels inside the box that are not
        loaded yet simply re-upload their current (backdrop) content.
        """
        if not batch:
            return True
        ndisplay = self._viewer.dims.ndisplay
        if ndisplay not in (2, 3) or vdata.ndim < ndisplay:
            return False
        if block is not None:
            low, high, data = block
            return self._patch_texture_region(vdata, low, high, block=data)
        ndim = vdata.ndim
        low = [min(int(key[d].start) for key in batch) for d in range(ndim)]
        high = [max(int(key[d].stop) for key in batch) for d in range(ndim)]
        return self._patch_texture_region(vdata, low, high)

    def _patch_texture_region(
        self,
        vdata: VirtualData,
        low,
        high,
        block=None,
    ) -> bool:
        """Write an absolute-coordinate region into the GPU texture.

        A partial texture upload (glTexSubImage2D/3D) is orders of
        magnitude cheaper than the re-slice plus whole-texture upload of
        a pipeline refresh. Works for data of any dimensionality: the
        texture holds the displayed dims' corner-pixels crop at the
        current step of every other dim. Only used when the texture
        demonstrably matches the current interval; any mismatch falls
        back to a normal refresh.
        """
        ndisplay = self._viewer.dims.ndisplay
        ndim = vdata.ndim
        if (
            ndisplay not in (2, 3)
            or ndim < ndisplay
            or len(low) != ndim
            or vdata.interval is None
        ):
            return False
        plane = self._displayed_plane(int(self._layer.data_level))
        if plane is None:
            return False
        displayed, steps = plane
        if len(displayed) != ndisplay:
            return False
        # regions that miss the displayed plane have nothing to upload —
        # the hyperslice already holds them for other steps
        for d, step in steps.items():
            if not int(low[d]) <= step < int(high[d]):
                return True
        node = self._get_display_node(ndisplay)
        if node is None:
            return False
        try:
            texture = node._texture
            dbuf = self._ensure_dbuf(node) if self._double_buffer else None
            # during a pending reshape the bound texture still has the
            # old tile's shape while staged patches target the new one,
            # so validate against the pair's staging shape
            tex_shape = (
                dbuf.shape
                if dbuf is not None
                else tuple(texture.shape[:ndisplay])
            )
        except (AttributeError, TypeError, RuntimeError):  # pragma: no cover
            return False
        # The texture holds the corner-pixels crop of the level (the
        # rendered tile), which sits inside the chunk-aligned interval.
        corners = self._layer.corner_pixels
        box_min = {d: int(corners[0, d]) for d in displayed}
        box_max = {d: int(corners[1, d]) + 1 for d in displayed}
        if tex_shape != tuple(box_max[d] - box_min[d] for d in displayed):
            # texture not (yet) synced to the current tile
            return False
        region_low = [int(v) for v in low]
        lo = {d: max(int(low[d]), box_min[d]) for d in displayed}
        hi = {d: min(int(high[d]), box_max[d]) for d in displayed}
        if any(hi[d] <= lo[d] for d in displayed):
            return False
        offset = tuple(lo[d] - box_min[d] for d in displayed)
        displayed_set = set(displayed)
        try:
            if block is not None:
                # precomputed contiguous copy from the fetch worker:
                # index the displayed sub-box (and the plane of every
                # other dim) out of it
                inner = tuple(
                    steps[d] - region_low[d]
                    if d in steps
                    else slice(lo[d] - region_low[d], hi[d] - region_low[d])
                    if d in displayed_set
                    else slice(None)
                    for d in range(ndim)
                )
                sub = block[inner]
                expected = tuple(hi[d] - lo[d] for d in displayed)
                if sub.shape[: len(expected)] != expected:
                    return False
                sub = np.ascontiguousarray(sub)
            else:
                # region in absolute coords -> hyperslice coords
                translate = vdata.translate
                source = tuple(
                    steps[d] - int(translate[d])
                    if d in steps
                    else slice(
                        lo[d] - int(translate[d]),
                        hi[d] - int(translate[d]),
                    )
                    if d in displayed_set
                    else slice(None)
                    for d in range(ndim)
                )
                with vdata.lock:
                    sub = np.ascontiguousarray(vdata.hyperslice[source])
            # napari uploads GL-incompatible dtypes (e.g. int16) as a
            # converted type; patches must match the texture's dtype
            tex_dtype = getattr(texture, '_data_dtype', None)
            if tex_dtype is not None and sub.dtype != tex_dtype:
                sub = sub.astype(tex_dtype)
            if dbuf is not None:
                # stream into the back texture; draws keep sampling the
                # untouched front texture until the next present()
                dbuf.stage(offset, sub)
            else:
                texture.set_data(sub, offset=offset)
        except Exception:  # noqa: BLE001 # pragma: no cover - GL mismatch
            return False
        self._texture_patches += 1
        return True

    def _ensure_dbuf(self, node):
        """(Re)build the double-buffered texture pair for ``node``."""
        dbuf_cls = (
            DoubleBufferedVolumeTexture
            if self._viewer.dims.ndisplay == 3
            else DoubleBufferedImageTexture
        )
        dbuf = self._dbuf
        if dbuf is not None and type(dbuf) is dbuf_cls and dbuf.matches(node):
            return dbuf
        texture = getattr(node, '_texture', None)
        if texture is None or getattr(texture, '_data_dtype', None) is None:
            # the node still holds vispy's unresolved placeholder (e.g.
            # the 10x10 RGBA checkerboard, internalformat unsettled):
            # building a sibling from it raises a channel mismatch.
            # Not an unsupported configuration — just too early; retry
            # on a later patch, after the first real upload resolves it.
            return None
        pool = []
        if dbuf is not None:
            # carry the retired-texture pool across rebuilds (same GL
            # context): pooled textures stay reusable and rebuilds do
            # not pay delete + reallocate GPU syncs. Only between same
            # texture classes — a 2D/3D switch closes the pool instead.
            if type(dbuf) is dbuf_cls:
                pool, dbuf._pool = dbuf._pool, []
            with contextlib.suppress(Exception):
                dbuf.close()
            self._dbuf = None
        try:
            dbuf = dbuf_cls(node, pool=pool)
            dbuf.attach_set_data()
        except Exception:  # noqa: BLE001 - unexpected texture class
            LOGGER.warning(
                'texture double buffering unavailable; falling back to '
                'in-place texture patches',
                exc_info=True,
            )
            self._double_buffer = False
            for _key, tex in pool:
                with contextlib.suppress(Exception):
                    tex.delete()
            return None
        self._dbuf = dbuf
        return dbuf

    def _update_node(self) -> None:
        node = self._get_display_node()
        if node is not None:
            if self._dbuf is not None and self._dbuf.matches(node):
                try:
                    self._dbuf.present()
                except Exception:  # noqa: BLE001 # pragma: no cover - GL
                    LOGGER.warning(
                        'texture present failed; dropping double buffer',
                        exc_info=True,
                    )
                    with contextlib.suppress(Exception):
                        self._dbuf.close()
                    self._dbuf = None
            with contextlib.suppress(RuntimeError):
                node.update()
        self._maybe_restore_quality()

    def _refresh(self, final: bool = False, force: bool = False) -> None:
        # During interaction, defer throttled refreshes entirely (the
        # reload -> async re-slice cascade is main-thread work that
        # would land in drag frames); forced refreshes — backdrops at
        # pass boundaries, where the canvas would otherwise be wrong —
        # still go through.
        if self._holding and not force:
            self._held_refresh = True
            return
        # Adaptive throttle: refresh as often as every chunk (so loading is
        # visibly progressive) while never spending more than ~half the
        # time re-slicing — large 3D volumes re-upload the whole texture
        # per refresh, so their interval backs off automatically.
        now = time.monotonic()
        min_interval = max(
            self._refresh_interval_s,
            2.0 * self._last_refresh_duration,
        )
        if not (final or force) and now - self._last_refresh < min_interval:
            return
        self._last_refresh = now
        start = time.monotonic()
        self._layer.refresh(extent=False, highlight=False, thumbnail=final)
        self._last_refresh_duration = time.monotonic() - start

    def _repair_backdrop(self) -> None:
        """Backfill unloaded regions of the active level from the coarsest.

        The upsampling gather can take seconds for large tiles, so it runs
        on a background thread (VirtualData is lock-guarded) limited to
        the currently rendered region; the layer refreshes on completion.
        """
        level = int(self._layer.data_level)
        if level == self._resident_level:
            return
        min_coord, max_coord = self._level_interval(level)
        if np.any(max_coord <= min_coord):
            return
        if self._repair_worker is not None:
            return  # one repair at a time; _check will re-trigger if needed

        @thread_worker
        def repair():
            return self._backdrop_fill_layered(level, min_coord, max_coord)

        worker = repair()

        def on_done(wrote):
            if self._repair_worker is worker:
                self._repair_worker = None
            if self._closed:
                return
            dbuf = self._dbuf3d()
            if dbuf is not None:
                dbuf.release_presents()
            if wrote:
                self._refresh(force=True)
            elif dbuf is not None:
                # nothing to fill (fully carried over): present whatever
                # is staged now that the veto is lifted
                with contextlib.suppress(Exception):
                    dbuf.present()

        worker.returned.connect(on_done)
        worker.errored.connect(lambda _e: on_done(False))  # pragma: no cover
        self._repair_worker = worker
        worker.start()

    # -- resident coarsest level --

    def _resident_target_interval(
        self,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """The interval of the coarsest level worth keeping resident.

        Prefers the full level (so backdrops/thumbnails work everywhere);
        if that exceeds the memory budget — e.g. the coarsest level of a
        large timelapse spans every timepoint — falls back to the full
        *displayed* extent at the current step of the other dimensions.
        Returns ``None`` if even that is too large.
        """
        vdata = self._data[self._resident_level]
        itemsize = vdata.dtype.itemsize
        if vdata.size * itemsize <= self._resident_max_bytes:
            return (
                np.zeros(vdata.ndim, dtype=np.int64),
                np.asarray(vdata.shape, dtype=np.int64),
            )

        min_coord = np.zeros(vdata.ndim, dtype=np.int64)
        max_coord = np.asarray(vdata.shape, dtype=np.int64)
        displayed = set(self._layer._slice_input.displayed)
        self._restrict_to_current_step(
            self._resident_level,
            displayed,
            min_coord,
            max_coord,
        )
        nbytes = np.prod(max_coord - min_coord, dtype=np.int64) * itemsize
        if nbytes > self._resident_max_bytes:
            if not self._resident_disabled:
                self._resident_disabled = True
                LOGGER.warning(
                    'coarsest level slice is %.0f MB (> %.0f MB); not '
                    'keeping it resident. Backdrops and thumbnails will be '
                    'limited to the visible region.',
                    nbytes / 1e6,
                    self._resident_max_bytes / 1e6,
                )
            return None
        return min_coord, max_coord

    def _ensure_resident(self) -> None:
        """Keep the coarsest level resident around the current view.

        The resident level provides instant low-resolution context when
        panning/zooming, backs the layer thumbnail, and is the backdrop
        source for finer levels. When the resident interval no longer
        covers the current dims step (e.g. the time slider moved), it is
        refilled.
        """
        target = self._resident_target_interval()
        if target is None:
            return
        min_coord, max_coord = target
        vdata = self._data[self._resident_level]
        if vdata.covers(min_coord, max_coord):
            if self._resident_worker is not None:
                return
            keys = chunk_slices(vdata, interval=(min_coord, max_coord))
            if all(
                _chunk_id(key) in vdata.loaded_chunks
                for key in itertools.product(*keys)
            ):
                return
        self._start_resident_fill(min_coord, max_coord)

    def _start_resident_fill(self, min_coord, max_coord) -> None:
        if self._resident_limiter is not None:
            self._resident_limiter.cancel()
            self._resident_limiter = None
        if self._resident_worker is not None:
            self._resident_worker.quit()
            self._resident_worker = None
        self._close_progress(self._resident_pbar)
        self._resident_pbar = None
        self._resident_pbar_pending = 0
        vdata = self._data[self._resident_level]
        vdata.set_interval(min_coord, max_coord)
        interval = vdata.interval
        keys = chunk_slices(vdata, interval=interval)
        queue = [
            key
            for key in chunk_priority_2D(keys, interval[0], interval[1])
            if _chunk_id(key) not in vdata.loaded_chunks
        ]
        if not queue:
            return

        self._resident_pbar = self._make_progress(
            len(queue),
            f'{self._layer.name}: loading overview',
        )

        def apply(chunk_key, chunk, vdata=vdata):
            vdata.set_offset(chunk_key, chunk)
            vdata.loaded_chunks.add(_chunk_id(chunk_key))

        self._resident_limiter = self._make_limiter()
        worker = _fetch_chunks(
            vdata.array,
            queue,
            num_workers=self._fetch_workers,
            apply=apply,
            limiter=self._resident_limiter,
        )

        def on_chunk(batch, vdata=vdata):
            if self._closed or self._resident_worker is not worker:
                return
            if self._resident_pbar is not None:
                self._advance_progress(len(batch), resident=True)
            if self._layer.data_level == self._resident_level:
                self._refresh()

        def on_finished():
            if self._closed or self._resident_worker is not worker:
                return
            self._resident_worker = None
            self._close_progress(self._resident_pbar)
            self._resident_pbar = None
            # If the active level's interval was initialized before this
            # fill finished, its backdrop is zeros; repair the regions
            # that have no real chunks yet so the canvas shows the
            # (coarse) volume immediately.
            self._repair_backdrop()
            self._refresh(final=True)
            # Re-evaluate coverage now that backdrops are available.
            self._check()

        worker.yielded.connect(on_chunk)
        worker.finished.connect(on_finished)
        self._resident_worker = worker
        worker.start()


# ---------- public entry point ----------


def _estimate_contrast_limits(array) -> tuple[float, float] | None:
    """Estimate contrast limits from a central sample of an array."""
    try:
        key = []
        for size in array.shape:
            center = int(size) // 2
            half = min(int(size), 256) // 2
            key.append(slice(max(center - half, 0), center + half + 1))
        sample = np.asarray(array[tuple(key)])
    except Exception:  # pragma: no cover - estimation is best-effort
        LOGGER.exception('contrast limit estimation failed')
        return None
    if sample.size == 0:
        return None
    low = float(np.min(sample))
    high = float(np.max(sample))
    if low == high:
        high = low + 1
    return low, high


def add_progressive_loading_image(
    img,
    viewer: napari.Viewer | None = None,
    contrast_limits: tuple[float, float] | None = None,
    colormap: str = 'gray',
    rendering: str = 'attenuated_mip',
    name: str | None = None,
    auto_level_3d: bool = True,
    max_pixel_size_3d: float = 2.0,
    interval_max_bytes: int = DEFAULT_INTERVAL_MAX_BYTES,
    tile_max_bytes_3d: int = DEFAULT_TILE_MAX_BYTES_3D,
    max_bytes_per_second: float | None = None,
    interaction_hold: bool = True,
    interactive_step_rate: float = 4.0,
    **layer_kwargs,
):
    """Add a progressively loading multiscale image to a viewer.

    The image is added as a *single* multiscale layer. Chunks of the level
    napari selects for rendering are streamed in on a background thread,
    nearest to the view center first, while coarser data is shown as a
    backdrop. The layer's resolution selector (``locked_data_level``) is
    respected.

    Parameters
    ----------
    img : sequence of array-like
        Multiscale image data, highest resolution first. Levels may be
        zarr arrays, dask arrays, or anything implementing ``shape``,
        ``dtype``, ``chunks``/``chunksize`` and ``__getitem__``.
    viewer : napari.Viewer, optional
        The viewer to add the image to. A new one is created if not given.
    contrast_limits : tuple of float, optional
        Contrast limits for the layer. If not given, they are estimated
        from a central sample of the coarsest level.
    colormap : str
        Colormap for the layer.
    rendering : str
        3D rendering mode for the layer.
    name : str, optional
        Layer name.
    auto_level_3d : bool
        In 3D, automatically pick the rendered data level from the camera
        zoom (napari itself always uses the coarsest level in 3D). The
        resolution selector stays on "Auto"; pinning an explicit level
        there suspends automatic selection.
    max_pixel_size_3d : float
        Tuning knob for 3D auto level selection: target the coarsest
        level whose voxels project to at most this many screen pixels.
    interval_max_bytes : int
        Memory budget for a single level's resident interval.
    tile_max_bytes_3d : int
        Upper bound for a 3D sub-volume tile; bounds the cost of the
        full-tile GPU uploads at pass boundaries (roughly
        size / 125 MB/s of GUI blocking on slow GL drivers).
    max_bytes_per_second : float, optional
        Rate-limit chunk loading (see
        :class:`ProgressiveLoader`). ``None`` = unlimited.
    interaction_hold : bool
        Suspend all streaming work while the user interacts (see
        :class:`ProgressiveLoader`).
    interactive_step_rate : float
        Coarsen the volume raycast step by this factor during
        interaction, restoring full quality on settle (see
        :class:`ProgressiveLoader`). 1.0 disables.
    **layer_kwargs
        Additional keyword arguments passed to ``viewer.add_image``.

    Returns
    -------
    napari.layers.Image
        The created layer. The active :class:`ProgressiveLoader` is stored
        in ``layer.metadata['progressive_loader']``.

    """
    if viewer is None:
        from napari import Viewer

        viewer = Viewer()

    env_tile = os.environ.get('NAPARI_PROGRESSIVE_TILE_MAX_BYTES_3D')
    if env_tile:
        tile_max_bytes_3d = int(float(env_tile))

    data = MultiScaleVirtualData(img)

    # vispy renders with float32: world extents beyond 2**24 lose pixel
    # precision and 3D rendering goes blank entirely. If the caller did
    # not specify a scale, normalize the world size of very deep pyramids.
    scale = layer_kwargs.get('scale')
    if scale is None:
        max_extent = float(max(data.shape))
        # vanilla napari 3D rendering goes blank for world extents at or
        # beyond 2**22 (measured; float32 precision in the render path)
        limit = float(2**21)
        if max_extent > limit:
            factor = 2.0 ** -int(np.ceil(np.log2(max_extent / limit)))
            layer_kwargs['scale'] = (factor,) * data.ndim
            LOGGER.warning(
                'image extent %.3g exceeds float32 rendering precision; '
                'scaling the layer by %g to keep it renderable. Pass '
                'scale= explicitly to override.',
                max_extent,
                factor,
            )

    if contrast_limits is None:
        contrast_limits = _estimate_contrast_limits(data.arrays[-1])

    from napari.layers import Image

    # Construct the layer directly (instead of viewer.add_image) so the
    # 3D tile extent is set before the layer controls are built — the
    # resolution selector then knows fine levels render as sub-volume
    # tiles and does not disable them.
    layer = Image(
        data._data,
        multiscale=True,
        contrast_limits=contrast_limits,
        colormap=colormap,
        rendering=rendering,
        name=name,
        **layer_kwargs,
    )
    tile_bytes = min(interval_max_bytes, tile_max_bytes_3d)
    layer._max_tile_extent_3d = _tile_extent_3d_for(data.dtype, tile_bytes)
    layer._tile_max_bytes_3d = tile_bytes
    viewer.layers.append(layer)
    # Slice off the main thread: refreshes materialize the visible tile
    # (np.asarray over up to hundreds of MB), which would otherwise block
    # the UI. Layer.refresh only routes through the async slicer when the
    # experimental setting is on; VirtualData access is lock-guarded, so
    # concurrent slicing is safe.
    from napari.settings import get_settings

    get_settings().experimental.async_ = True
    viewer._layer_slicer._force_sync = False
    # Meter GLIR 3D texture uploads: without this, vispy drains every
    # queued glTexSubImage3D inside the next draw — including interaction
    # frames — which stalls the GUI for seconds on slow GL drivers
    # (macOS GL-over-Metal).
    from napari.experimental import _glir_metering

    _glir_metering.install()
    loader = ProgressiveLoader(
        viewer,
        layer,
        data,
        auto_level_3d=auto_level_3d,
        max_pixel_size_3d=max_pixel_size_3d,
        interval_max_bytes=interval_max_bytes,
        tile_max_bytes_3d=tile_max_bytes_3d,
        max_bytes_per_second=max_bytes_per_second,
        interaction_hold=interaction_hold,
        interactive_step_rate=interactive_step_rate,
    )
    layer.metadata['progressive_loader'] = loader
    with contextlib.suppress(AttributeError):
        # stop all background work when the window goes away
        viewer.window._qt_window.destroyed.connect(loader.close)
    return layer


def add_progressive_loading_labels(
    labels,
    viewer: napari.Viewer | None = None,
    name: str | None = None,
    auto_level_3d: bool = True,
    max_pixel_size_3d: float = 2.0,
    interval_max_bytes: int = DEFAULT_INTERVAL_MAX_BYTES,
    tile_max_bytes_3d: int = DEFAULT_TILE_MAX_BYTES_3D,
    max_bytes_per_second: float | None = None,
    interaction_hold: bool = True,
    interactive_step_rate: float = 4.0,
    **layer_kwargs,
):
    """Add a progressively loading multiscale labels layer to a viewer.

    Works identically to :func:`add_progressive_loading_image` but creates
    a :class:`~napari.layers.Labels` layer.  Editing is disabled (napari
    disables editing for multiscale labels), so the layer is read-only.

    Parameters
    ----------
    labels : sequence of array-like
        Multiscale label data, highest resolution first.  Must be integer
        typed.  Levels may be zarr arrays, dask arrays, or anything
        implementing ``shape``, ``dtype``, ``chunks``/``chunksize`` and
        ``__getitem__``.
    viewer : napari.Viewer, optional
        The viewer to add the layer to.  A new one is created if not given.
    name : str, optional
        Layer name.
    auto_level_3d : bool
        Automatically pick the rendered data level from the camera zoom
        in 3D.
    max_pixel_size_3d : float
        Tuning knob for 3D auto level selection.
    interval_max_bytes : int
        Memory budget for a single level's resident interval.
    tile_max_bytes_3d : int
        Upper bound for a 3D sub-volume tile.
    max_bytes_per_second : float, optional
        Rate-limit chunk loading.  ``None`` = unlimited.
    interaction_hold : bool
        Suspend all streaming work while the user interacts.
    interactive_step_rate : float
        Coarsen the volume raycast step by this factor during interaction.
    **layer_kwargs
        Additional keyword arguments passed to the ``Labels`` constructor.

    Returns
    -------
    napari.layers.Labels
        The created layer.  The active :class:`ProgressiveLoader` is stored
        in ``layer.metadata['progressive_loader']``.
    """
    if viewer is None:
        from napari import Viewer

        viewer = Viewer()

    env_tile = os.environ.get('NAPARI_PROGRESSIVE_TILE_MAX_BYTES_3D')
    if env_tile:
        tile_max_bytes_3d = int(float(env_tile))

    data = MultiScaleVirtualData(labels)

    scale = layer_kwargs.get('scale')
    if scale is None:
        max_extent = float(max(data.shape))
        limit = float(2**21)
        if max_extent > limit:
            factor = 2.0 ** -int(np.ceil(np.log2(max_extent / limit)))
            layer_kwargs['scale'] = (factor,) * data.ndim
            LOGGER.warning(
                'label extent %.3g exceeds float32 rendering precision; '
                'scaling the layer by %g to keep it renderable. Pass '
                'scale= explicitly to override.',
                max_extent,
                factor,
            )

    from napari.layers import Labels

    layer = Labels(
        data._data,
        multiscale=True,
        name=name,
        **layer_kwargs,
    )
    tile_bytes = min(interval_max_bytes, tile_max_bytes_3d)
    layer._max_tile_extent_3d = _tile_extent_3d_for(data.dtype, tile_bytes)
    layer._tile_max_bytes_3d = tile_bytes
    layer._interval_max_bytes_3d = interval_max_bytes
    viewer.layers.append(layer)

    from napari.settings import get_settings

    get_settings().experimental.async_ = True
    viewer._layer_slicer._force_sync = False

    from napari.experimental import _glir_metering

    _glir_metering.install()
    loader = ProgressiveLoader(
        viewer,
        layer,
        data,
        auto_level_3d=auto_level_3d,
        max_pixel_size_3d=max_pixel_size_3d,
        interval_max_bytes=interval_max_bytes,
        tile_max_bytes_3d=tile_max_bytes_3d,
        max_bytes_per_second=max_bytes_per_second,
        interaction_hold=interaction_hold,
        interactive_step_rate=interactive_step_rate,
    )
    layer.metadata['progressive_loader'] = loader
    with contextlib.suppress(AttributeError):
        viewer.window._qt_window.destroyed.connect(loader.close)
    return layer
