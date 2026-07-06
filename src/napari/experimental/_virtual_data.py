"""Virtual array wrappers used by progressive loading.

This module provides array-like objects that present the full shape of a
(potentially enormous) array while only keeping a small, chunk-aligned
region of it in memory:

- :class:`VirtualData` wraps a single scale level. It satisfies napari's
  ``LayerDataProtocol`` (``shape``, ``dtype``, ``ndim``, ``size``,
  ``__getitem__``) so it can be used directly as one level of a multiscale
  ``Image`` layer. Reads outside the in-memory interval return zeros.
- :class:`VirtualArrayView` is the lazy result of indexing a
  :class:`VirtualData`. It records the requested region in absolute
  coordinates and only materializes (zero-padded) data when converted with
  ``np.asarray``. This lets napari's slicing machinery compose multiple
  indexing operations (e.g. first the displayed-axis crop, then the
  current-step point selection) before any data is copied.
- :class:`MultiScaleVirtualData` coordinates one :class:`VirtualData` per
  scale level and can initialize newly exposed regions from a coarser,
  fully resident level so the canvas never shows empty space while chunks
  stream in.

All hyperslice accesses are guarded by a per-``VirtualData`` reentrant
lock so background fetch threads and the main thread can safely interleave.
"""

from __future__ import annotations

import itertools
import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)


def _regular_chunk_sizes(
    shape: Sequence[int],
    chunk_shape: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Expand a regular chunk shape into clipped per-dimension sizes."""
    out = []
    for size, step in zip(shape, chunk_shape, strict=True):
        size, step = int(size), max(int(step), 1)
        n_full, remainder = divmod(size, step)
        out.append((step,) * n_full + ((remainder,) if remainder else ()))
    return tuple(out)


def chunk_sizes_for(array) -> tuple[tuple[int, ...], ...]:
    """Per-dimension sequences of chunk data sizes, clipped to the array.

    Follows the dask ``Array.chunks`` convention: one inner tuple per
    dimension whose values sum to the array's extent along it. Supports
    dask arrays (``chunks``), zarr arrays with regular *or* rectilinear
    chunk grids (``read_chunk_sizes``, or ``chunks`` on zarr versions
    without rectilinear support), and falls back to a bounded chunk
    shape for plain ndarrays.
    """
    # zarr with rectilinear-chunk-grid support: regular and rectilinear
    # grids both expose clipped per-dimension sizes (``.chunks`` raises
    # NotImplementedError for rectilinear grids, so it comes first)
    sizes = getattr(array, 'read_chunk_sizes', None)
    if sizes is not None:
        return tuple(tuple(int(c) for c in dim) for dim in sizes)
    chunks = getattr(array, 'chunks', None)
    if chunks is not None:
        if all(isinstance(c, (int, np.integer)) for c in chunks):
            # regular grid as a chunk shape (older zarr, h5py, ...)
            return _regular_chunk_sizes(array.shape, chunks)
        # dask-style tuple of per-dimension chunk size tuples
        return tuple(tuple(int(c) for c in dim) for dim in chunks)
    return _regular_chunk_sizes(
        array.shape,
        tuple(min(int(s), 256) for s in array.shape),
    )


def chunk_shape_for(array) -> tuple[int, ...]:
    """Return a per-dimension (maximum) chunk shape for an array.

    For irregular (rectilinear/dask) grids this is the largest chunk
    along each dimension, matching dask's ``chunksize``.
    """
    chunksize = getattr(array, 'chunksize', None)  # dask
    if chunksize is not None:
        return tuple(int(c) for c in chunksize)
    return tuple(
        max(sizes) if sizes else 1 for sizes in chunk_sizes_for(array)
    )


def chunk_boundaries(array) -> list[np.ndarray]:
    """Per-dimension sorted arrays of chunk boundary positions.

    Each entry covers ``0`` through ``shape[dim]`` inclusive, so consecutive
    pairs of values describe one chunk along that dimension. Irregular
    (dask or zarr rectilinear) chunks are honored exactly.
    """
    return [
        np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        for sizes in chunk_sizes_for(array)
    ]


def chunk_ids_in_region(boundaries, lo, hi):
    """Iterate chunk ids (tuples of ``(start, stop)``) intersecting a region.

    ``boundaries`` is the per-dimension boundary list of
    :func:`chunk_boundaries`; ``[lo, hi)`` is a half-open region in the
    same coordinates.
    """
    per_dim = []
    for dim, bounds in enumerate(boundaries):
        starts, stops = bounds[:-1], bounds[1:]
        first = int(np.searchsorted(stops, int(lo[dim]), side='right'))
        last = int(np.searchsorted(starts, int(hi[dim]), side='left'))
        per_dim.append(
            [(int(starts[i]), int(stops[i])) for i in range(first, last)],
        )
    return itertools.product(*per_dim)


class VirtualArrayView:
    """A lazy, zero-padded view into a :class:`VirtualData`.

    The view tracks the requested region in the *absolute* coordinates of
    the wrapped array. Indexing a view returns another view (with the keys
    composed), and ``np.asarray`` materializes the data: regions inside the
    currently resident interval are copied from the hyperslice and
    everything else is zero.
    """

    def __init__(
        self,
        data: VirtualData,
        index: tuple[int | tuple[int, int], ...],
    ):
        # index has one entry per dimension of ``data``:
        # an int collapses the dimension, a (start, stop) pair keeps it.
        self._data = data
        self._index = index

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(
            stop - start
            for entry in self._index
            if isinstance(entry, tuple)
            for start, stop in [entry]
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def __getitem__(self, key) -> VirtualArrayView:
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            n_missing = self.ndim - sum(1 for k in key if k is not Ellipsis)
            expanded: list = []
            for k in key:
                if k is Ellipsis:
                    expanded.extend([slice(None)] * n_missing)
                else:
                    expanded.append(k)
            key = tuple(expanded)

        out: list[int | tuple[int, int]] = []
        key_pos = 0
        for entry in self._index:
            if isinstance(entry, int):
                out.append(entry)
                continue
            start, stop = entry
            n = stop - start
            k = key[key_pos] if key_pos < len(key) else slice(None)
            key_pos += 1
            if isinstance(k, slice):
                k_start, k_stop, k_step = k.indices(n)
                if k_step != 1:
                    raise IndexError(
                        'VirtualArrayView only supports step-1 slices, '
                        f'got {k!r}',
                    )
                out.append((start + k_start, start + max(k_stop, k_start)))
            elif isinstance(k, (int, np.integer)):
                idx = int(k)
                if idx < 0:
                    idx += n
                if not 0 <= idx < n:
                    raise IndexError(
                        f'index {k} is out of bounds for dimension of size {n}',
                    )
                out.append(start + idx)
            else:
                raise IndexError(
                    f'unsupported index for VirtualArrayView: {k!r}',
                )
        if key_pos < len(key):
            raise IndexError(
                f'too many indices: array is {self.ndim}-dimensional, '
                f'but {len(key)} were indexed',
            )
        return VirtualArrayView(self._data, tuple(out))

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        data = self._data
        out = np.full(self.shape, data.fill_value, dtype=data.dtype)
        with data.lock:
            if data._min_coord is not None:
                src_key: list = []
                dst_key: list = []
                inside = True
                for dim, entry in enumerate(self._index):
                    min_c = data._min_coord[dim]
                    max_c = data._max_coord[dim]
                    if isinstance(entry, int):
                        if not min_c <= entry < max_c:
                            inside = False
                            break
                        src_key.append(entry - min_c)
                    else:
                        start, stop = entry
                        lo = max(start, min_c)
                        hi = min(stop, max_c)
                        if hi <= lo:
                            inside = False
                            break
                        src_key.append(slice(lo - min_c, hi - min_c))
                        dst_key.append(slice(lo - start, hi - start))
                if inside:
                    out[tuple(dst_key)] = data.hyperslice[tuple(src_key)]
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    def transpose(self, axes=None) -> np.ndarray:
        """Materialize and transpose (numpy compatibility)."""
        return np.asarray(self).transpose(axes)

    def __repr__(self) -> str:
        return (
            f'<VirtualArrayView shape={self.shape} dtype={self.dtype} '
            f'index={self._index}>'
        )


class VirtualData:
    """Present a full-size array while holding only one region in memory.

    ``VirtualData`` wraps one scale level's array (zarr, dask, numpy, ...).
    It reports the wrapped array's full ``shape``/``dtype`` but only stores
    the data within the current *interval* (set via :meth:`set_interval`) in
    an in-memory numpy array called the *hyperslice*. The interval is always
    aligned outward to chunk boundaries of the wrapped array.

    Indexing uses the wrapped array's (absolute) coordinates and returns a
    lazy :class:`VirtualArrayView`; reads outside the interval materialize
    as zeros. Writes go through :meth:`set_offset`, which clips the value to
    the resident interval.

    Attributes
    ----------
    array : array-like
        The wrapped array (not copied; only read from).
    hyperslice : np.ndarray
        In-memory data for the current interval.
    translate : tuple of int
        Offset of the hyperslice origin from the wrapped array's origin.
    lock : threading.RLock
        Guards ``hyperslice``/interval mutation and access.

    """

    def __init__(self, array, scale_level: int = 0, fill_value=0):
        self.array = array
        self.dtype = np.dtype(array.dtype)
        self.shape = tuple(int(s) for s in array.shape)
        self.ndim = len(self.shape)
        self.scale_level = scale_level
        self.fill_value = fill_value

        self.lock = threading.RLock()
        self.translate: tuple[int, ...] = (0,) * self.ndim
        self.hyperslice = np.full((0,) * self.ndim, fill_value, dtype=self.dtype)
        self._min_coord: list[int] | None = None
        self._max_coord: list[int] | None = None
        self._boundaries = chunk_boundaries(array)
        self._chunk_shape = chunk_shape_for(array)
        # Chunk keys (tuples of (start, stop) pairs) whose data is resident
        # in the hyperslice. Maintained by the progressive loader.
        self.loaded_chunks: set[tuple[tuple[int, int], ...]] = set()
        # Best-effort provenance per resident chunk: the scale level the
        # chunk's current content came from (this level = real data; a
        # coarser level = upsampled backdrop). Feeds debugging overlays;
        # absent entries mean unknown origin (typically still zeros).
        self.chunk_source: dict[tuple[tuple[int, int], ...], int] = {}

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        # cached in __init__: the wrapped array is fixed, and this is read
        # per-level on the interactive interval-sizing path
        return self._chunk_shape

    @property
    def interval(self) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """Current resident interval as ``(min_coord, max_coord)`` or None."""
        with self.lock:
            if self._min_coord is None:
                return None
            return tuple(self._min_coord), tuple(self._max_coord)

    def chunk_aligned_interval(
        self,
        min_coord,
        max_coord,
    ) -> tuple[list[int], list[int]]:
        """Expand ``[min_coord, max_coord)`` outward to chunk boundaries."""
        lo: list[int] = []
        hi: list[int] = []
        for dim in range(self.ndim):
            bounds = self._boundaries[dim]
            min_c = int(np.clip(min_coord[dim], 0, self.shape[dim]))
            max_c = int(np.clip(max_coord[dim], 0, self.shape[dim]))
            i = max(int(np.searchsorted(bounds, min_c, side='right')) - 1, 0)
            j = int(np.searchsorted(bounds, max_c, side='left'))
            j = min(max(j, i + 1), len(bounds) - 1)
            lo.append(int(bounds[i]))
            hi.append(int(bounds[j]))
        return lo, hi

    def covers(self, min_coord, max_coord) -> bool:
        """Return True if the resident interval covers the given region."""
        with self.lock:
            if self._min_coord is None:
                return False
            return all(
                self._min_coord[d] <= int(min_coord[d])
                and int(max_coord[d]) <= self._max_coord[d]
                for d in range(self.ndim)
            )

    def set_interval(
        self,
        min_coord,
        max_coord,
        backdrop=None,
        backdrop_source: int | None = None,
    ) -> None:
        """Set the resident interval, preserving overlapping data.

        The interval is expanded outward to chunk boundaries. Data that
        overlaps the previous interval is carried over; newly exposed
        regions are initialized from ``backdrop`` (if given) or zeros.

        Parameters
        ----------
        min_coord, max_coord : sequence of int
            Requested interval in the wrapped array's coordinates
            (half-open, like slices).
        backdrop : callable, optional
            ``backdrop(min_coord, max_coord) -> np.ndarray | None`` returning
            initial content for the new hyperslice (e.g. upsampled data from
            a coarser scale). Called with the chunk-aligned interval.
        backdrop_source : int, optional
            Scale level the backdrop content comes from; recorded in
            ``chunk_source`` (debug provenance) for chunks the backdrop
            initialized.

        """
        new_min, new_max = self.chunk_aligned_interval(min_coord, max_coord)
        with self.lock:
            prev_min = self._min_coord
            prev_hyperslice = self.hyperslice

            if (
                prev_min is not None
                and new_min == prev_min
                and new_max == self._max_coord
            ):
                return

            new_shape = [
                mx - mn for mn, mx in zip(new_min, new_max, strict=True)
            ]
            next_hyperslice = None
            if backdrop is not None:
                try:
                    content = backdrop(new_min, new_max)
                except Exception:  # pragma: no cover - backdrop is advisory
                    LOGGER.exception('backdrop initialization failed')
                    content = None
                if content is not None and tuple(content.shape) == tuple(
                    new_shape,
                ):
                    next_hyperslice = np.ascontiguousarray(
                        content,
                        dtype=self.dtype,
                    )
            used_backdrop = next_hyperslice is not None
            if next_hyperslice is None:
                next_hyperslice = np.full(new_shape, self.fill_value, dtype=self.dtype)

            # Carry over data overlapping the previous interval to avoid
            # re-fetching and visual flashing.
            if prev_min is not None and prev_hyperslice.size:
                src_key = []
                dst_key = []
                overlaps = True
                for dim in range(self.ndim):
                    lo = max(prev_min[dim], new_min[dim])
                    hi = min(
                        prev_min[dim] + prev_hyperslice.shape[dim],
                        new_max[dim],
                    )
                    if hi <= lo:
                        overlaps = False
                        break
                    src_key.append(
                        slice(lo - prev_min[dim], hi - prev_min[dim]),
                    )
                    dst_key.append(slice(lo - new_min[dim], hi - new_min[dim]))
                if overlaps:
                    next_hyperslice[tuple(dst_key)] = prev_hyperslice[
                        tuple(src_key)
                    ]

            self._min_coord = new_min
            self._max_coord = new_max
            self.translate = tuple(new_min)
            self.hyperslice = next_hyperslice

            # Drop bookkeeping for chunks that fell outside the interval.
            self.loaded_chunks = {
                key
                for key in self.loaded_chunks
                if all(
                    new_min[d] <= start and stop <= new_max[d]
                    for d, (start, stop) in enumerate(key)
                )
            }
            self.chunk_source = {
                key: src
                for key, src in self.chunk_source.items()
                if all(
                    new_min[d] <= start and stop <= new_max[d]
                    for d, (start, stop) in enumerate(key)
                )
            }
            if backdrop_source is not None and used_backdrop:
                # debug provenance: chunks not carried over start out
                # showing the backdrop source's (upsampled) content
                for chunk_id in chunk_ids_in_region(
                    self._boundaries,
                    new_min,
                    new_max,
                ):
                    self.chunk_source.setdefault(chunk_id, backdrop_source)

    def set_offset(self, key: tuple[slice, ...], value) -> None:
        """Write ``value`` at ``key`` (absolute coordinates), clipped.

        Regions of ``key`` outside the resident interval are ignored.
        """
        value = np.asarray(value)
        with self.lock:
            if self._min_coord is None:
                return
            dst_key = []
            src_key = []
            for dim, sl in enumerate(key):
                lo = max(int(sl.start), self._min_coord[dim])
                hi = min(int(sl.stop), self._max_coord[dim])
                if hi <= lo:
                    return
                dst_key.append(
                    slice(
                        lo - self._min_coord[dim], hi - self._min_coord[dim]
                    ),
                )
                src_key.append(slice(lo - int(sl.start), hi - int(sl.start)))
            expected = tuple(s.stop - s.start for s in src_key)
            if value[tuple(src_key)].shape != expected:
                LOGGER.warning(
                    'set_offset: value shape %s does not cover key %s',
                    value.shape,
                    key,
                )
                return
            self.hyperslice[tuple(dst_key)] = value[tuple(src_key)]

    def __getitem__(self, key) -> VirtualArrayView:
        full = tuple((0, s) for s in self.shape)
        return VirtualArrayView(self, full)[key]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return VirtualArrayView(
            self,
            tuple((0, s) for s in self.shape),
        ).__array__(dtype=dtype)

    def __repr__(self) -> str:
        return (
            f'<VirtualData scale_level={self.scale_level} shape={self.shape} '
            f'dtype={self.dtype} interval={self.interval}>'
        )


class MultiScaleVirtualData:
    """Coordinate a :class:`VirtualData` per level of a multiscale image.

    The list of per-level :class:`VirtualData` objects (``._data``) can be
    passed directly to ``viewer.add_image(..., multiscale=True)``. The
    object also tracks the scale factors between levels and can initialize
    a level's freshly exposed region from a coarser level's resident data
    (:meth:`backdrop_for`), which keeps the canvas filled with low-resolution
    content while high-resolution chunks stream in.

    Parameters
    ----------
    arrays : sequence of array-like
        Multiscale levels from highest resolution (index 0) to lowest.

    """

    def __init__(self, arrays: Sequence, fill_value=0):
        if len(arrays) == 0:
            raise ValueError('arrays must be a non-empty sequence')
        self.arrays = list(arrays)
        self._data = [
            VirtualData(array, scale_level=level, fill_value=fill_value)
            for level, array in enumerate(self.arrays)
        ]
        highest_res = self._data[0]
        self.dtype = highest_res.dtype
        self.shape = highest_res.shape
        self.ndim = highest_res.ndim

        # Per-level, per-dimension downsampling factor relative to level 0.
        self._scale_factors = [
            [
                hr_dim / level_dim
                for hr_dim, level_dim in zip(
                    highest_res.shape,
                    vdata.shape,
                    strict=True,
                )
            ]
            for vdata in self._data
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, level: int) -> VirtualData:
        return self._data[level]

    @property
    def levels(self) -> int:
        return len(self._data)

    def backdrop_for(self, level: int, src_level: int):
        """Return a backdrop callable that upsamples from ``src_level``.

        The returned function is suitable for
        :meth:`VirtualData.set_interval`'s ``backdrop`` parameter; it samples
        the source level's resident hyperslice with nearest-neighbor
        interpolation. Returns ``None`` when no useful backdrop exists.
        """
        if src_level == level:
            return None
        src = self._data[src_level]
        ratio = [
            self._scale_factors[level][d] / self._scale_factors[src_level][d]
            for d in range(self.ndim)
        ]

        def backdrop(min_coord, max_coord):
            with src.lock:
                if src._min_coord is None or src.hyperslice.size == 0:
                    return None
                indices = []
                for d in range(self.ndim):
                    coords = (
                        np.arange(min_coord[d], max_coord[d]) + 0.5
                    ) * ratio[d]
                    idx = coords.astype(np.int64) - src._min_coord[d]
                    indices.append(
                        np.clip(idx, 0, src.hyperslice.shape[d] - 1),
                    )
                return src.hyperslice[np.ix_(*indices)]

        return backdrop

    def set_interval(
        self,
        level: int,
        min_coord,
        max_coord,
        backdrop_level: int | None = None,
    ) -> None:
        """Set the resident interval for ``level`` (in level coordinates).

        If ``backdrop_level`` is given, newly exposed regions are initialized
        with nearest-neighbor upsampled data from that level.
        """
        backdrop = (
            self.backdrop_for(level, backdrop_level)
            if backdrop_level is not None
            else None
        )
        self._data[level].set_interval(
            min_coord,
            max_coord,
            backdrop=backdrop,
            backdrop_source=backdrop_level if backdrop is not None else None,
        )

    def fill_unloaded_from(
        self,
        level: int,
        src_level: int,
        region=None,
    ) -> bool:
        """Fill not-yet-loaded chunk regions of ``level`` from ``src_level``.

        Used to repair the backdrop when the source level finished loading
        only after the destination level's interval had been initialized
        (which would otherwise leave zeros on screen until real chunks
        arrive). Already-loaded chunks are left untouched.

        Parameters
        ----------
        region : tuple of (min_coord, max_coord), optional
            Restrict the repair to this absolute-coordinate region (e.g.
            the currently rendered tile); the upsampling gather scales
            with the region size.

        Returns True if anything was written.

        """
        backdrop = self.backdrop_for(level, src_level)
        if backdrop is None:
            return False
        dst = self._data[level]
        with dst.lock:
            if dst._min_coord is None or dst.hyperslice.size == 0:
                return False
            fill_min = list(dst._min_coord)
            fill_max = list(dst._max_coord)
            if region is not None:
                fill_min = [
                    min(max(int(r), lo), hi)
                    for r, lo, hi in zip(
                        region[0], fill_min, fill_max, strict=True
                    )
                ]
                fill_max = [
                    min(max(int(r), lo), hi)
                    for r, lo, hi in zip(
                        region[1], dst._min_coord, fill_max, strict=True
                    )
                ]
                if any(
                    mx <= mn for mn, mx in zip(fill_min, fill_max, strict=True)
                ):
                    return False
            content = backdrop(fill_min, fill_max)
            expected = tuple(
                mx - mn for mn, mx in zip(fill_min, fill_max, strict=True)
            )
            if content is None or tuple(content.shape) != expected:
                return False
            region_key = tuple(
                slice(mn - lo, mx - lo)
                for mn, mx, lo in zip(
                    fill_min, fill_max, dst._min_coord, strict=True
                )
            )
            if not dst.loaded_chunks:
                dst.hyperslice[region_key] = content
                for chunk_id in chunk_ids_in_region(
                    dst._boundaries,
                    fill_min,
                    fill_max,
                ):
                    dst.chunk_source[chunk_id] = src_level
                return True
            # per-dimension chunk extents covering the fill region, as
            # (hyperslice_start, hyperslice_stop, absolute_id) entries
            per_dim: list[list[tuple[int, int, tuple[int, int]]]] = []
            for dim in range(dst.ndim):
                bounds = dst._boundaries[dim]
                lo = dst._min_coord[dim]
                region_lo, region_hi = fill_min[dim], fill_max[dim]
                first = max(
                    int(np.searchsorted(bounds, region_lo, side='right')) - 1,
                    0,
                )
                entries = []
                start = int(bounds[first])
                for stop in bounds[first + 1 :]:
                    stop = int(stop)
                    if start >= region_hi:
                        break
                    entries.append((start - lo, stop - lo, (start, stop)))
                    start = stop
                per_dim.append(entries)
            offset = [
                mn - lo
                for mn, lo in zip(fill_min, dst._min_coord, strict=True)
            ]
            wrote = False
            for combo in itertools.product(*per_dim):
                chunk_id = tuple(absolute for *_rel, absolute in combo)
                if chunk_id in dst.loaded_chunks:
                    continue
                dst_key = tuple(
                    slice(max(rel_start, off), rel_stop)
                    for (rel_start, rel_stop, _absolute), off in zip(
                        combo,
                        offset,
                        strict=True,
                    )
                )
                src_key = tuple(
                    slice(sl.start - off, sl.stop - off)
                    for sl, off in zip(dst_key, offset, strict=True)
                )
                if any(sl.stop <= sl.start for sl in src_key):
                    continue
                src_clipped = tuple(
                    slice(sl.start, min(sl.stop, dim_len))
                    for sl, dim_len in zip(src_key, content.shape, strict=True)
                )
                if any(sl.stop <= sl.start for sl in src_clipped):
                    continue
                dst_clipped = tuple(
                    slice(d.start, d.start + (sc.stop - sc.start))
                    for d, sc in zip(dst_key, src_clipped, strict=True)
                )
                dst.hyperslice[dst_clipped] = content[src_clipped]
                dst.chunk_source[chunk_id] = src_level
                wrote = True
            return wrote
