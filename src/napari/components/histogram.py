"""Histogram model for Image layer data visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import PrivateAttr

from napari.utils._dask_utils import _is_dask_data
from napari.utils.events import EventedModel

if TYPE_CHECKING:
    from napari.layers.image.image import Image  # noqa: TC004

__all__ = ('HistogramModel',)

# Default histogram configuration
_DEFAULT_N_BINS: int = 256
_DEFAULT_MAX_SAMPLES: int = 1_000_000
_DEFAULT_CANVAS_WIDTH: int = 300
_DEFAULT_CANVAS_HEIGHT: int = 150


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
    log_scale : Event
        Fired when log scale setting changes.
    n_bins : Event
        Fired when number of bins changes.
    mode : Event
        Fired when histogram mode changes.
    enabled : Event
        Fired when enabled state changes.
    """

    # Evented properties
    n_bins: int = _DEFAULT_N_BINS
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
        n_bins: int = _DEFAULT_N_BINS,
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
        mode : {'canvas', 'full'}, default: 'canvas'
            Whether to compute histogram from displayed data or full volume.
        log_scale : bool, default: False
            Use logarithmic scale for histogram counts.
        enabled : bool, default: False
            Whether histogram responds to data-change events automatically.
        """
        super().__init__(
            n_bins=n_bins, mode=mode, log_scale=log_scale, enabled=enabled
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

            # Sample if data is too large.
            # This also covers dask arrays that were already reduced by
            # _get_full_data / _sample_dask_safe to avoid full
            # materialization.
            if data.size > _DEFAULT_MAX_SAMPLES:
                data = self._sample_data(data, _DEFAULT_MAX_SAMPLES)

            # Get histogram range from contrast limits range
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
        finally:
            self._computing = False

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
            data.ravel(),
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
        efficiency.  For dask or other lazy arrays, avoids full
        materialization by sampling across chunks so that ``compute()``
        never loads the entire array into memory.
        """
        data = self._layer.data
        if isinstance(data, list | tuple):
            data = data[-1]

        if isinstance(data, np.ndarray):
            return data

        if _is_dask_data(data):
            return self._sample_dask_safe(data)

        # For other lazy array types (tensorstore, zarr, etc.) that
        # haven't been pre-sliced by the canvas pipeline, try to sample
        # rather than materializing the full volume.
        if hasattr(data, 'shape') and data.size > _DEFAULT_MAX_SAMPLES:
            return self._sample_dask_safe(data)

        return np.asarray(data)

    @staticmethod
    def _sample_dask_safe(data: object) -> np.ndarray:
        """Sample from a dask array across chunks without full materialization.

        Iterates over available chunks and draws proportional samples
        from each, avoiding ``.ravel()`` which would build a large task
        graph for very large arrays. Only the chunks that are sampled
        are computed, never the full array.

        Parameters
        ----------
        data : object
            A dask array.

        Returns
        -------
        np.ndarray
            Flat array of sampled finite values.
        """
        import dask.array as da

        data_arr: da.Array = data  # type: ignore[assignment]
        n_total = data_arr.size
        n_samples = min(_DEFAULT_MAX_SAMPLES, n_total)

        # Build list of (slice, chunk_size) for every chunk in the array
        chunk_slices_and_sizes: list[tuple[tuple[slice, ...], int]] = []
        for idx in np.ndindex(*data_arr.numblocks):
            slices: list[slice] = []
            chunk_size = 1
            for d, i in enumerate(idx):
                start = sum(data_arr.chunks[d][:i])
                stop = start + data_arr.chunks[d][i]
                slices.append(slice(start, stop))
                chunk_size *= data_arr.chunks[d][i]
            chunk_slices_and_sizes.append((tuple(slices), chunk_size))

        total_chunk_size = sum(sz for _, sz in chunk_slices_and_sizes)
        rng = np.random.default_rng(0)
        sampled_parts: list[np.ndarray] = []

        for chunk_slc, chunk_sz in chunk_slices_and_sizes:
            n_samp = max(1, int(n_samples * chunk_sz / total_chunk_size))
            chunk_data = da.asarray(data_arr[chunk_slc])  # type: ignore[no-untyped-call]
            chunk_computed = chunk_data.compute().ravel()
            if chunk_computed.size <= n_samp:
                sampled_parts.append(chunk_computed)
            else:
                chosen = rng.choice(
                    chunk_computed.size, size=n_samp, replace=False
                )
                sampled_parts.append(chunk_computed[chosen])

        combined = np.concatenate(sampled_parts)
        # Trim down to the target sample count if we overshot
        if combined.size > n_samples:
            chosen = rng.choice(combined.size, size=n_samples, replace=False)
            combined = combined[chosen]

        valid = combined[np.isfinite(combined)]
        return valid if len(valid) > 0 else combined[:1]

    def _sample_data(self, data: np.ndarray, max_samples: int) -> np.ndarray:
        """Randomly sample data to reduce computation."""
        flat_data = data.ravel()
        valid_mask = np.isfinite(flat_data)
        valid_data = flat_data[valid_mask]

        if valid_data.size == 0:
            return np.array([])

        if valid_data.size <= max_samples:
            return valid_data

        rng = np.random.default_rng(0)
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
        self.n_bins = _DEFAULT_N_BINS
        self.log_scale = False
        self.mode = 'canvas'
        self._bins = np.array([0.0, 1.0])
        self._counts = np.array([0.0])
        self._dirty = True
