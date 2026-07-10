"""Histogram model for Image layer data visualization."""

from __future__ import annotations

import math
import warnings
from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import PrivateAttr

from napari.utils._dask_utils import _is_dask_data
from napari.utils.events import Event, EventedModel

if TYPE_CHECKING:
    from napari.layers.image.image import Image  # noqa: TC004

__all__ = ('HistogramModel',)

# Default histogram configuration
DEFAULT_BINS: int = 256
DEFAULT_MAX_SAMPLES: int = 1_000_000
# Maximum number of elements to materialize into a numpy array when
# the data is not chunked (e.g. h5py datasets).  Used as a safety
# guard in _get_full_data() — beyond this threshold we skip full-mode
# computation and warn instead of silently pulling the full array into
# memory.  ~50M float64 elements ≈ 400 MB.
_MAX_MATERIALIZE_ELEMENTS: int = 50_000_000


class HistogramModel(EventedModel):
    """Data model for histogram computation and display.

    This model computes and stores histogram data for an Image layer,
    responding to changes in layer data, contrast limits, and gamma.

    Parameters
    ----------
    layer : Image
        The layer to compute histogram for.
    bins : int, default: 256
        Number of histogram bins (matches ``np.histogram(data, bins=...)``).
    mode : {'canvas', 'full'}, default: 'canvas'
        Whether to compute histogram from displayed data or full volume.
    log_scale : bool, default: False
        Use logarithmic scale for histogram counts.
    enabled : bool, default: False
        Whether histogram responds to data-change events automatically.

    Attributes
    ----------
    counts : np.ndarray
        Histogram counts per bin (length bins).

    Events
    ------
    counts : Event
        Fired when histogram data is recomputed.
    bins : Event
        Fired when the number of bins changes.
    max_samples : Event
        Fired when the max_samples limit changes.
    mode : Event
        Fired when histogram mode changes.
    log_scale : Event
        Fired when log scale setting changes.
    enabled : Event
        Fired when enabled state changes.
    """

    # Evented properties
    bins: int = DEFAULT_BINS
    max_samples: int = DEFAULT_MAX_SAMPLES
    mode: Literal['canvas', 'full'] = 'canvas'
    log_scale: bool = False
    enabled: bool = False

    # Private attributes — pydantic's PrivateAttr is not validated at
    # runtime, so the annotation here is only for documentation and
    # readability; the true type is ``Image`` (enforced by __init__).
    _layer: Image = PrivateAttr()
    _bin_edges: np.ndarray = PrivateAttr(
        default_factory=lambda: np.array([0.0, 1.0])
    )
    _counts: np.ndarray = PrivateAttr(default_factory=lambda: np.array([0.0]))
    _dirty: bool = PrivateAttr(default=True)
    _computing: bool = PrivateAttr(default=False)
    _compute_generation: int = PrivateAttr(default=0)
    # Set True (on the main thread) by the first view that starts a
    # background compute for this model, so the other view sharing the model
    # (e.g. the inline histogram vs. the contrast-limits popup) does not start
    # a second, competing worker: two concurrent workers would fight over
    # ``_computing`` / ``_compute_generation`` (corrupting the progressive
    # accumulation) and double the remote I/O.  A plain bool keeps Qt out of
    # this (non-Qt) model — the Qt widget layer owns the actual worker object.
    # Distinct from ``_computing`` (set inside the worker thread) because
    # scheduling must be serialized on the main thread before the thread runs.
    _compute_scheduled: bool = PrivateAttr(default=False)
    # Cached full-mode result: (bin_edges, counts, log_scale_at_compute_time).
    # Full-mode compute is costly and slice-independent, so we keep the last
    # result and restore it on switch-back instead of recomputing.  Canvas
    # mode is cheap and not cached.  Invalidated by data/param changes (see
    # ``_invalidate``); a slice change never touches it (full is unaffected).
    _full_cache: tuple[np.ndarray, np.ndarray, bool] | None = PrivateAttr(
        default=None
    )

    def __init__(
        self,
        layer: Image,
        bins: int = DEFAULT_BINS,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        mode: Literal['canvas', 'full'] = 'canvas',
        log_scale: bool = False,
        enabled: bool = False,
    ):
        """Initialize histogram model.

        Parameters
        ----------
        layer : Image
            The layer to compute histogram for.
        bins : int, default: 256
            Number of histogram bins (matches ``np.histogram(data, bins=...)``).
        max_samples : int, default: 1_000_000
            Maximum number of data points to sample from the full volume
            when ``mode='full'`` and the data exceeds this threshold.
        mode : {'canvas', 'full'}, default: 'canvas'
            Whether to compute histogram from displayed data or full volume.
        log_scale : bool, default: False
            Use logarithmic scale for histogram counts.
        enabled : bool, default: False
            Whether histogram responds to data-change events automatically.
        """
        super().__init__(
            bins=bins,
            max_samples=max_samples,
            mode=mode,
            log_scale=log_scale,
            enabled=enabled,
        )

        self._layer = layer
        self._layer_events_connected = False

        # Render-only broadcast event (psygnal, like every other event here —
        # no Qt).  The widget layer emits this after each progressive chunk so
        # that *all* views sharing this model (the inline histogram and the
        # contrast-limits popup) re-render the partial in lockstep, regardless
        # of which view owns the compute.  Unlike ``events.counts`` (which
        # views also use as a recompute trigger), this never restarts compute.
        self.events.add(partial_computed=Event)

        # Connect to our own events (these are internal to the model and
        # don't leak external callbacks on the layer).
        self.events.bins.connect(self._invalidate)
        self.events.max_samples.connect(self._invalidate)
        self.events.mode.connect(self._on_mode_change)
        self.events.log_scale.connect(self._on_log_scale_change)
        self.events.enabled.connect(self._on_enabled_change)

    def _connect_layer_events(self) -> None:
        """Connect to layer events to trigger recomputation.

        Connections are made lazily (only when the histogram is actually
        computing or enabled) so that they don't leak on the layer when
        the histogram was never used or when the layer is removed from
        the viewer.
        """
        if self._layer_events_connected:
            return
        self._layer_events_connected = True
        self._layer.events.data.connect(self._invalidate)
        self._layer.events.contrast_limits_range.connect(self._invalidate)
        self._layer.events.set_data.connect(self._on_slice_change)

    def _disconnect_layer_events(self) -> None:
        """Disconnect layer events, the symmetric counterpart to
        ``_connect_layer_events``."""
        if not self._layer_events_connected:
            return
        self._layer_events_connected = False
        self._layer.events.data.disconnect(self._invalidate)
        self._layer.events.contrast_limits_range.disconnect(self._invalidate)
        self._layer.events.set_data.disconnect(self._on_slice_change)

    @property
    def counts(self) -> np.ndarray:
        """Histogram counts per bin.

        Triggers lazy computation if the model is dirty.  The widget
        reads ``_counts`` directly to avoid accidentally triggering
        compute during visual updates.

        Returns
        -------
        np.ndarray
            Array of counts with length ``self.bins``.
        """
        if self._dirty:
            for _ in self.compute():
                pass
        return self._counts

    def _set_empty_data(self) -> None:
        """Set histogram to empty bin/edge state."""
        self._bin_edges = np.array([0.0, 1.0])
        self._counts = np.array([0.0])
        self._dirty = False

    def compute(
        self,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generator yielding ``(bin_edges, counts)`` for histogram computation.

        For chunked arrays in full mode, yields an intermediate result after
        each chunk so the caller can update the display progressively (e.g.
        via a ``GeneratorWorker`` in a background thread).  For non-chunked
        data, yields the final result once.

        Does **not** emit ``events.counts()`` — the caller (e.g.
        ``_on_async_compute_done`` on the main thread) is responsible for
        emitting after the generator completes.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            ``(bin_edges, counts)`` — bin edge values and per-bin counts.
        """
        if self._computing:
            return

        self._computing = True
        self._compute_generation += 1
        generation = self._compute_generation
        self._connect_layer_events()
        try:
            data = self._get_data()

            if data is None or data.size == 0:
                self._set_empty_data()
                return

            # For RGB(A) images convert to luminance so the histogram
            # represents perceived brightness.
            # Sample pixel positions BEFORE conversion to avoid
            # materializing the full float32 intermediate for large arrays.
            if self._layer.rgb:
                data = self._sample_rgb_and_luminance(data)
                if data.size == 0:
                    self._set_empty_data()
                    return

            if self.mode == 'full' and self._has_chunks(data):
                yield from self._compute_chunked_progressive(data, generation)
            else:
                # Always sample large data to keep the UI responsive,
                # regardless of mode.
                if data.size > self.max_samples:
                    data = self._sample_data(data, self.max_samples)
                self._finalize_histogram(data)
                yield self._bin_edges, self._counts
        finally:
            self._computing = False

    def _finalize_histogram(self, data: np.ndarray) -> None:
        """Compute histogram from a complete data array.

        Sets ``_bin_edges``, ``_counts``, and clears the dirty flag.
        Does **not** emit ``events.counts()`` — callers handle that.

        Parameters
        ----------
        data : np.ndarray
            Pre-processed data array to histogram (already sampled if needed).
        """
        range_min, range_max = self._layer.contrast_limits_range
        if range_min is None or range_max is None:
            range_min = float(np.nanmin(data))
            range_max = float(np.nanmax(data))

        bin_edges, counts = self._calc_histogram(data, range_min, range_max)
        self._bin_edges = bin_edges
        self._counts = counts
        self._dirty = False
        if self.mode == 'full':
            self._full_cache = (bin_edges, counts, self.log_scale)

    def _compute_chunked_progressive(
        self, data: Any, generation: int
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generator that yields ``(bin_edges, counts)`` after each chunk.

        Provides incremental histogram snapshots as each chunk is loaded.
        The final yield updates the model's internal ``_bin_edges`` /
        ``_counts`` and marks it not-dirty, so callers that skip
        intermediate results (e.g. the synchronous ``compute`` path)
        still see consistent state.

        Parameters
        ----------
        data : Any
            Chunked array (dask, zarr, etc.).
        generation : int
            Generation counter from the calling ``compute()`` invocation.
            If ``self._compute_generation`` differs, intermediate results
            are discarded to prevent stale async data from overwriting
            fresher computations.
        """
        n = min(self.max_samples, data.size)
        chunk_sizes = self._chunk_sizes(data)
        rng = np.random.default_rng()
        n_chunks = len(chunk_sizes)
        n_selected = min(n_chunks, max(1, n // max(1, min(chunk_sizes))))
        probs = np.asarray(chunk_sizes) / sum(chunk_sizes)
        order = rng.choice(n_chunks, size=n_selected, p=probs, replace=False)

        range_min, range_max = self._layer.contrast_limits_range
        if range_min is None or range_max is None:
            range_min = 0.0
            range_max = 1.0

        running_counts = np.zeros(self.bins, dtype=np.float64)
        for ci in order:
            # Early stale guard: check before the I/O-bound chunk load so
            # that aborted workers (e.g. after a mode switch to canvas)
            # exit quickly instead of blocking the thread pool on a chunk
            # whose results will be discarded anyway.
            if self._compute_generation != generation:
                return
            block = self._load_chunk(data, ci)
            chunk_counts, _ = np.histogram(
                block,
                bins=self.bins,
                range=(float(range_min), float(range_max)),
            )
            running_counts += chunk_counts.astype(np.float64)

            bins = np.linspace(range_min, range_max, self.bins + 1).astype(
                np.float32
            )
            if self.log_scale:
                counts = np.log10(running_counts + 1).astype(np.float32)
            else:
                counts = running_counts.astype(np.float32)

            # Post-load guard: also check after computation to prevent
            # a freshly-computed partial from overwriting state that was
            # set by a newer compute while this chunk was loading.
            if self._compute_generation != generation:
                return
            yield bins, counts

        # Stale guard: only update model state if this generation is
        # still current.  Without this, a stale async worker that
        # finishes all its chunks after a newer inline compute has
        # already set state would overwrite the fresh data.
        if self._compute_generation == generation:
            self._bin_edges = bins
            self._counts = counts
            self._dirty = False
            # This generator only runs for chunked full mode, so cache it.
            self._full_cache = (bins, counts, self.log_scale)

    def _calc_histogram(
        self,
        data: np.ndarray,
        range_min: float,
        range_max: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute bin edges and counts from data.

        Separates the pure numpy histogram computation from data fetching
        and preprocessing.

        Parameters
        ----------
        data : np.ndarray
            Preprocessed 1D data array.
        range_min : float
            Minimum value for histogram range.
        range_max : float
            Maximum value for histogram range.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (bin_edges, counts) — bin edges as float32, counts as float32.
        """
        # Handle edge case where min == max (constant data).
        # For integer types, ±0.5 places bin edges at half-integer
        # boundaries (e.g. uint8 value 42 → bin [41.5, 42.5]).
        # For float types, expand by 1 % of the value (min 0.5) to
        # keep the bin width proportional to the data magnitude.
        if range_min == range_max:
            if np.issubdtype(self._layer.dtype, np.integer):
                range_min = float(range_min) - 0.5
                range_max = float(range_max) + 0.5
            else:
                delta = (
                    max(0.5, abs(range_min) * 0.01) if range_min != 0 else 0.5
                )
                range_min = float(range_min) - delta
                range_max = float(range_max) + delta

        counts, bins = np.histogram(
            data,
            bins=self.bins,
            range=(float(range_min), float(range_max)),
        )

        bin_edges = bins.astype(np.float32)

        if self.log_scale:
            hist_counts = np.log10(counts + 1).astype(np.float32)
        else:
            hist_counts = counts.astype(np.float32)

        return bin_edges, hist_counts

    def _rgb_to_luminance(self, data: np.ndarray) -> np.ndarray:
        """Convert RGB(A) data to perceptual luminance.

        Uses ITU-R BT.709 coefficients so the result matches sRGB display
        brightness. Only the first three channels are used; alpha is ignored.
        The returned array has the same value range as the input (e.g. 0-255
        for uint8, 0-1 for float).
        """
        rgb: np.ndarray = data[..., :3].astype(np.float32)
        return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    def _sample_rgb_and_luminance(self, data: np.ndarray) -> np.ndarray:
        """Convert RGB(A) data to luminance, sampling pixels first for large data.

        For large RGB arrays, randomly samples ``max_samples`` pixel positions
        BEFORE converting to luminance to avoid materializing the full float32
        intermediate array. For small data, delegates to ``_rgb_to_luminance``.
        """
        n_pixels = data.size // data.shape[-1]
        if n_pixels <= self.max_samples:
            return self._rgb_to_luminance(data)

        rng = np.random.default_rng()
        pixel_indices = rng.choice(
            n_pixels, size=self.max_samples, replace=False
        )
        nd_indices = np.unravel_index(pixel_indices, data.shape[:-1])
        sampled_rgb = data[nd_indices + (slice(None),)]
        luminance = self._rgb_to_luminance(sampled_rgb)
        valid = np.isfinite(luminance)
        return luminance[valid]

    def _get_data(self) -> np.ndarray | None:
        """Get data from layer based on current mode."""
        if self.mode == 'canvas':
            return self._get_displayed_data()
        return self._get_full_data()

    def _get_displayed_data(self) -> np.ndarray | None:
        """Get data from currently displayed slice.

        In 'canvas' mode, the histogram is computed from the visible data
        that has already been sliced for rendering. This uses the layer's
        ``_slice.image.raw`` which contains the data being displayed.

        Returns None if the slice is not yet available (e.g. during initial
        loading) — the caller will produce empty bins/counts instead of
        silently falling back to full-volume data. Because
        ``HistogramModel`` is created lazily (only on first property
        access), slicing has typically already completed by the time this
        is called.
        """
        raw = self._get_slice_raw_data()
        if raw is not None and raw.size > 0:
            return raw
        return None

    def _get_slice_raw_data(self) -> np.ndarray | None:
        """Get the currently sliced raw image data if available."""
        layer_slice = self._layer._slice
        if layer_slice is None:
            return None
        raw = layer_slice.image.raw
        return np.asarray(raw) if raw is not None else None

    def _get_full_data(self) -> np.ndarray | None:
        """Get full volume data.

        For multiscale data, uses the lowest resolution level for
        efficiency.  For chunked arrays (dask, zarr), returns the raw
        data as-is — sampling is handled by
        ``_compute_chunked_progressive``.
        """
        data = self._layer.data

        # Unpack multiscale to the coarsest level.
        if isinstance(data, Sequence) and not isinstance(
            data, (np.ndarray, str, bytes)
        ):
            data = data[-1]

        if isinstance(data, np.ndarray):
            return data

        # Chunked arrays (dask, zarr, h5py with chunks) are returned
        # as-is for the progressive sampler in _compute_chunked_progressive.
        if self._has_chunks(data):
            return data

        # Last resort: cast to numpy.  Guard against accidentally
        # materializing a very large object (contiguous h5py) by
        # checking the estimated memory footprint first.
        data_size = data.size if hasattr(data, 'size') else 0
        if data_size > _MAX_MATERIALIZE_ELEMENTS:
            dtype_size = (
                np.dtype(data.dtype).itemsize if hasattr(data, 'dtype') else 8
            )
            est_mb = (data_size * dtype_size) / (1024 * 1024)
            warnings.warn(
                f'Skipping full-data histogram: materializing '
                f'{data_size:,} elements (~{est_mb:.0f} MB) would '
                f'exceed the safety limit of '
                f'{_MAX_MATERIALIZE_ELEMENTS:,} elements. '
                f'Use canvas mode or increase max_samples.',
                stacklevel=2,
            )
            return None
        return np.asarray(data)

    @staticmethod
    def _has_chunks(data: Any) -> bool:
        """True if *data* can be sampled chunk-by-chunk (dask, zarr, h5py).

        h5py datasets all have a ``.chunks`` attribute, but it is None for
        unchunked/contiguous datasets:
        https://docs.h5py.org/en/latest/high/dataset.html#chunked-storage
        """
        if _is_dask_data(data):
            return True
        chunks = getattr(data, 'chunks', None)
        return chunks is not None and hasattr(data, 'shape')

    @staticmethod
    def _chunk_sizes(data: Any) -> list[int]:
        """Return list of element counts for every chunk in *data*.

        Works for both dask (per-chunk tuple-of-tuples) and zarr
        (per-dimension scalar) arrays — only metadata is accessed.
        """
        import dask.array as da

        if isinstance(data, da.Array):
            nb, ch = data.numblocks, data.chunks
        else:
            # zarr: .chunks is (chunk_dim_0, chunk_dim_1, ...) not
            # per-chunk tuples. Compute block count from shape/chunks.
            nb = tuple(
                max(1, math.ceil(s / c))
                for s, c in zip(data.shape, data.chunks, strict=True)
            )
            ch = tuple(
                tuple(min(c, s - i * c) for i in range(n))
                for s, c, n in zip(data.shape, data.chunks, nb, strict=True)
            )

        sizes: list[int] = []
        for idx in np.ndindex(*nb):
            sz = 1
            for d, i in enumerate(idx):
                sz *= ch[d][i]
            sizes.append(sz)
        return sizes

    @staticmethod
    def _load_chunk(data: Any, flat_idx: int) -> np.ndarray:
        """Load a single chunk by its flat index.

        Works for both dask arrays (via ``.blocks``) and zarr arrays
        (via direct indexing with computed slice boundaries).
        """
        import dask.array as da

        if isinstance(data, da.Array):
            idx = np.unravel_index(flat_idx, data.numblocks)
            return np.asarray(data.blocks[idx]).ravel()

        # zarr path: convert flat chunk index to data-space slices
        nb = tuple(
            max(1, math.ceil(s / c))
            for s, c in zip(data.shape, data.chunks, strict=True)
        )
        idx = np.unravel_index(flat_idx, nb)
        slices: list[slice] = []
        for d, i in enumerate(idx):
            start = i * int(data.chunks[d])
            stop = min(start + int(data.chunks[d]), int(data.shape[d]))
            slices.append(slice(start, stop))
        return np.asarray(data[tuple(slices)]).ravel()

    def _sample_data(self, data: np.ndarray, max_samples: int) -> np.ndarray:
        """Randomly sample data to reduce computation."""
        flat_data = data.ravel()
        valid_mask = np.isfinite(flat_data)
        valid_data = flat_data[valid_mask]

        if valid_data.size == 0:
            return np.array([])

        if valid_data.size <= max_samples:
            return valid_data

        rng = np.random.default_rng()
        indices = rng.choice(valid_data.size, size=max_samples, replace=False)
        return valid_data[indices]

    def _on_slice_change(self) -> None:
        """Called when the displayed slice changes."""
        if self.mode == 'canvas':
            self._mark_dirty()

    def _on_log_scale_change(self) -> None:
        """Called when log_scale changes. Transforms counts without recomputing.

        When the model is clean, applies log10 (or inverse) to existing counts
        in-place.  When dirty, triggers compute which applies log_scale via
        ``_calc_histogram``.  For chunked full-mode data, defers to the
        widget's async worker to avoid blocking on I/O.
        """
        if self._dirty or len(self._counts) <= 1:
            # Defer chunked full-mode to the widget's async worker.
            if self.mode == 'full' and self._has_chunks(self._layer.data):
                return
            for _ in self.compute():
                pass
            self.events.counts()
            return

        self._apply_log_scale()

    def _apply_log_scale(self) -> None:
        """Transform ``_counts`` in place to match ``self.log_scale``, then emit.

        The caller is responsible for only invoking this on an actual flip.
        Shared by ``_on_log_scale_change`` (live toggle) and ``_on_mode_change``
        (restoring a cached full histogram computed in the other log state).
        """
        if self.log_scale:
            self._counts = np.log10(np.maximum(self._counts, 0) + 1).astype(
                np.float32
            )
        else:
            # Approximate inverse of log10(counts + 1)
            self._counts = np.maximum(10**self._counts - 1, 0).astype(
                np.float32
            )
        self.events.counts()

    def _on_enabled_change(self) -> None:
        """When enabled flips to True, compute if dirty (non-chunked) or
        connect layer events and defer to widget (chunked full-mode)."""
        if self.enabled:
            self._connect_layer_events()
            if self._dirty:
                self._mark_dirty()
        else:
            self._disconnect_layer_events()

    def _invalidate(self, event: Event | None = None) -> None:
        """Drop the cached full histogram, then mark dirty.

        Cache-clearing is kept out of ``_mark_dirty`` so the mode-restore path
        can call ``_mark_dirty`` without discarding the cache it may reuse.
        Connected to the events that actually change histogram values: layer
        ``data``/``contrast_limits_range`` and the ``bins``/``max_samples``
        params.  (A slice change goes through ``_on_slice_change`` instead and
        leaves the full cache intact, since full mode is slice-independent.)
        """
        self._full_cache = None
        self._mark_dirty(event)

    def _on_mode_change(self) -> None:
        """Handle a mode switch, restoring the cached full histogram if present.

        Full-mode compute is costly and slice-independent, so switching back to
        ``full`` restores the cached result (adjusting only for a log-scale
        difference) instead of recomputing.  Canvas mode is cheap and always
        recomputes via ``_mark_dirty``.
        """
        if self.mode == 'full' and self._full_cache is not None:
            bin_edges, counts, cached_log = self._full_cache
            self._bin_edges = bin_edges
            self._counts = counts
            self._dirty = False
            if cached_log != self.log_scale:
                # Counts were computed in the other log state; reuse the live
                # toggle transform to bring them to the current setting.
                self._apply_log_scale()
            else:
                self.events.counts()
            return
        self._mark_dirty()

    def _mark_dirty(self, event: Event | None = None) -> None:
        """Mark histogram as needing recomputation and compute if possible.

        For chunked arrays (dask, zarr) in full mode, defers computation
        to the widget's async ``GeneratorWorker`` to avoid blocking the
        main thread on I/O.  For all other cases, computes synchronously
        so the histogram updates immediately.
        """
        self._dirty = True
        if not self._computing and self.enabled:
            if self.mode == 'full' and self._has_chunks(self._layer.data):
                return
            for _ in self.compute():
                pass
            self.events.counts()

    def reset(self) -> None:
        """Reset histogram to default settings and disable.

        Clears cached data, resets all parameters to defaults,
        and sets enabled=False so no computation occurs until
        explicitly requested.
        """
        # Disable first to avoid wasteful intermediate compute() calls
        # from the parameter-change event handlers.
        self.enabled = False
        self.bins = DEFAULT_BINS
        self.max_samples = DEFAULT_MAX_SAMPLES
        self.log_scale = False
        self.mode = 'canvas'
        self._bin_edges = np.array([0.0, 1.0])
        self._counts = np.array([0.0])
        self._dirty = True
        self._full_cache = None
