"""Histogram model for Image layer data visualization."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import PrivateAttr

from napari.utils._dask_utils import _is_dask_data
from napari.utils.events import EventedModel

if TYPE_CHECKING:
    from napari.layers.image.image import Image  # noqa: TC004

__all__ = ('HistogramModel',)

# Default histogram configuration
DEFAULT_N_BINS: int = 256
DEFAULT_MAX_SAMPLES: int = 1_000_000


class HistogramModel(EventedModel):
    """Data model for histogram computation and display.

    This model computes and stores histogram data for an Image layer,
    responding to changes in layer data, contrast limits, and gamma.

    Parameters
    ----------
    layer : Image
        The layer to compute histogram for.
    n_bins : int, default: 256
        Number of histogram bins.
    mode : {'canvas', 'full'}, default: 'canvas'
        Whether to compute histogram from displayed data or full volume.
    log_scale : bool, default: False
        Use logarithmic scale for histogram counts.
    enabled : bool, default: False
        Whether histogram responds to data-change events automatically.
        When False the model is still computed on explicit access to
        ``bins`` or ``counts`` and when flipped to True.

    Attributes
    ----------
    bins : np.ndarray
        Histogram bin edges (length n_bins + 1).
    counts : np.ndarray
        Histogram counts per bin (length n_bins).

    Events
    ------
    bins : Event
        Fired when bin edges change.
    counts : Event
        Fired when histogram counts change.
    max_samples : Event
        Fired when the max_samples limit changes.
    n_bins : Event
        Fired when number of bins changes.
    mode : Event
        Fired when histogram mode changes.
    log_scale : Event
        Fired when log scale setting changes.
    enabled : Event
        Fired when enabled state changes.
    """

    # Evented properties
    n_bins: int = DEFAULT_N_BINS
    max_samples: int = DEFAULT_MAX_SAMPLES
    mode: Literal['canvas', 'full'] = 'canvas'
    log_scale: bool = False
    enabled: bool = False

    # Private attributes — pydantic's PrivateAttr is not validated at
    # runtime, so the annotation here is only for documentation and
    # readability; the true type is ``Image`` (enforced by __init__).
    _layer: Image = PrivateAttr()
    _bins: np.ndarray = PrivateAttr(
        default_factory=lambda: np.array([0.0, 1.0])
    )
    _counts: np.ndarray = PrivateAttr(default_factory=lambda: np.array([0.0]))
    _dirty: bool = PrivateAttr(default=True)
    _computing: bool = PrivateAttr(default=False)

    def __init__(
        self,
        layer: Image,
        n_bins: int = DEFAULT_N_BINS,
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
        n_bins : int, default: 256
            Number of histogram bins.
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
            n_bins=n_bins,
            max_samples=max_samples,
            mode=mode,
            log_scale=log_scale,
            enabled=enabled,
        )

        self._layer = layer

        # Connect to layer events to trigger recomputation
        layer.events.data.connect(self._on_data_change)
        layer.events.contrast_limits_range.connect(self._on_range_change)

        # set_data fires from Layer._refresh_sync() at the end of both
        # synchronous and async slice updates
        layer.events.set_data.connect(self._on_slice_change)

        # Connect to our own events
        self.events.n_bins.connect(self._on_params_change)
        self.events.max_samples.connect(self._on_params_change)
        self.events.mode.connect(self._on_params_change)
        self.events.log_scale.connect(self._on_log_scale_change)
        # When enabled flips True and data is dirty, compute immediately so
        # any widget that just subscribed to bins/counts events gets a result.
        self.events.enabled.connect(self._on_enabled_change)

    @property
    def bins(self) -> np.ndarray:
        """Histogram bin edges.

        Returns
        -------
        np.ndarray
            Array of bin edges with length n_bins + 1.
        """
        if self._dirty:
            self.compute()
        return self._bins

    @property
    def counts(self) -> np.ndarray:
        """Histogram counts per bin.

        Returns
        -------
        np.ndarray
            Array of counts with length n_bins.
        """
        if self._dirty:
            self.compute()
        return self._counts

    def compute(self) -> None:
        """Compute histogram from layer data.

        This method extracts data from the layer based on the current mode
        (displayed or full), samples if necessary, and computes the histogram.
        For chunked arrays (dask, zarr) in full mode, a random subset of
        chunks is sampled to avoid full materialization.
        """
        if self._computing:
            return

        self._computing = True
        try:
            data = self._get_data()

            if data is None or data.size == 0:
                self._bins = np.array([0.0, 1.0])
                self._counts = np.array([0.0])
                self._dirty = False
                self.events.bins()
                self.events.counts()
                return

            # For RGB(A) images convert to luminance so the histogram
            # represents perceived brightness.
            if self._layer.rgb:
                data = self._rgb_to_luminance(data)

            if self.mode == 'full' and self._has_chunks(data):
                # Chunked arrays: sample chunks, compute all at once
                self._compute_chunked(data)
            elif self.mode == 'full' and data.size > self.max_samples:
                # Random subsample for large in-memory arrays
                data = self._sample_data(data, self.max_samples)
                self._finalize_histogram(data)
            else:
                # Direct path — data is already manageable
                self._finalize_histogram(data)
        finally:
            self._computing = False

    def _finalize_histogram(self, data: np.ndarray) -> None:
        """Compute histogram from a complete data array and emit events."""
        range_min, range_max = self._layer.contrast_limits_range
        if range_min is None or range_max is None:
            range_min = float(np.nanmin(data))
            range_max = float(np.nanmax(data))

        bins, counts = self._calc_histogram(data, range_min, range_max)
        self._bins = bins
        self._counts = counts
        self._dirty = False
        self.events.bins()
        self.events.counts()

    def _compute_chunked(self, data: Any) -> None:
        """Compute histogram from a chunked array via random chunk sampling.

        Builds chunk-size metadata (cheap — no data access), randomly
        selects a subset of chunks proportional to their size, then
        loads and histogram-counts them in a single pass.  This avoids
        full materialization of remote or disk-backed arrays.

        .. note::
            For remote/HTTP data sources with high per-chunk latency,
            the Qt layer can wrap this in a ``thread_worker`` via
            :mod:`napari.qt.threading` to keep the UI responsive.
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

        running_counts = np.zeros(self.n_bins, dtype=np.float64)
        for ci in order:
            block = self._load_chunk(data, ci)
            chunk_counts, _ = np.histogram(
                block,
                bins=self.n_bins,
                range=(float(range_min), float(range_max)),
            )
            running_counts += chunk_counts.astype(np.float64)

        self._bins = np.linspace(range_min, range_max, self.n_bins + 1).astype(
            np.float32
        )
        if self.log_scale:
            self._counts = np.log10(running_counts + 1).astype(np.float32)
        else:
            self._counts = running_counts.astype(np.float32)

        self._dirty = False
        self.events.bins()
        self.events.counts()

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
            bins=self.n_bins,
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
        return (
            0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        )

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
        data as-is — progressive sampling is handled by ``_compute_sampled``.
        """
        data = self._layer.data

        # Unpack multiscale to the coarsest level.
        if isinstance(data, Sequence) and not isinstance(
            data, (np.ndarray, str, bytes)
        ):
            data = data[-1]

        if isinstance(data, np.ndarray):
            return data

        # Chunked arrays (dask, zarr) are returned as-is for the
        # progressive sampler in _compute_sampled.
        if self._has_chunks(data):
            return data

        return np.asarray(data)

    @staticmethod
    def _has_chunks(data: Any) -> bool:
        """True if *data* can be sampled chunk-by-chunk (dask or zarr).

        Both dask ``Array`` and zarr ``Array`` expose ``.chunks`` and
        ``.shape``, and support tuple-indexing to load individual chunks.
        """
        if _is_dask_data(data):
            return True
        return hasattr(data, 'chunks') and hasattr(data, 'shape')

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

    def _on_data_change(self) -> None:
        """Called when layer data changes."""
        self._mark_dirty()

    def _on_range_change(self) -> None:
        """Called when contrast limits range changes."""
        self._mark_dirty()

    def _on_slice_change(self) -> None:
        """Called when the displayed slice changes."""
        if self.mode == 'canvas':
            self._mark_dirty()

    def _on_params_change(self) -> None:
        """Called when n_bins or mode changes."""
        self._mark_dirty()

    def _on_log_scale_change(self) -> None:
        """Called when log_scale changes. Triggers full recomputation."""
        self._mark_dirty()

    def _on_enabled_change(self) -> None:
        """When enabled flips to True, compute if there is pending dirty data."""
        if self.enabled and self._dirty:
            self.compute()

    def _mark_dirty(self) -> None:
        """Mark histogram as needing recomputation.

        If already computing, defer recomputation (the ``_dirty`` flag
        persists so the next access will recompute).  If not computing
        and enabled, compute immediately so connected widgets stay live.
        If disabled, defer — the next explicit access or enabled=True
        will trigger the compute.
        """
        self._dirty = True
        if not self._computing and self.enabled:
            self.compute()

    def disconnect(self) -> None:
        """Disconnect all event listeners to prevent memory leaks.

        Call this when the layer is removed or the histogram is no
        longer needed to break psygnal event connections.
        """
        self._layer.events.data.disconnect(self._on_data_change)
        self._layer.events.contrast_limits_range.disconnect(
            self._on_range_change
        )
        self._layer.events.set_data.disconnect(self._on_slice_change)

        # Disconnect from our own events
        self.events.n_bins.disconnect(self._on_params_change)
        self.events.max_samples.disconnect(self._on_params_change)
        self.events.mode.disconnect(self._on_params_change)
        self.events.log_scale.disconnect(self._on_log_scale_change)
        self.events.enabled.disconnect(self._on_enabled_change)

    def reset(self) -> None:
        """Reset histogram to default settings and disable.

        Clears cached data, resets all parameters to defaults,
        and sets enabled=False so no computation occurs until
        explicitly requested.
        """
        # Disable first to avoid wasteful intermediate compute() calls
        # from the parameter-change event handlers.
        self.enabled = False
        self.n_bins = DEFAULT_N_BINS
        self.max_samples = DEFAULT_MAX_SAMPLES
        self.log_scale = False
        self.mode = 'canvas'
        self._bins = np.array([0.0, 1.0])
        self._counts = np.array([0.0])
        self._dirty = True
