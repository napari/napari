"""Histogram model for layer data visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from pydantic import PrivateAttr

from napari.utils.events import EventedModel

if TYPE_CHECKING:
    from napari.layers import Image

__all__ = ('HistogramModel',)


class HistogramModel(EventedModel):
    """Data model for histogram computation and display.

    This model computes and stores histogram data for a layer,
    responding to changes in layer data, contrast limits, and gamma.

    Parameters
    ----------
    layer : Image
        The layer to compute histogram for.
    n_bins : int, default: 256
        Number of histogram bins.
    mode : {'displayed', 'full'}, default: 'displayed'
        Whether to compute histogram from displayed data or full volume.
    log_scale : bool, default: False
        Use logarithmic scale for histogram counts.
    enabled : bool, default: True
        Whether histogram is enabled (computes on data changes).

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
    n_bins: int = 256
    mode: Literal['displayed', 'full'] = 'displayed'
    log_scale: bool = False
    enabled: bool = True

    # Private attributes (use PrivateAttr for pydantic)
    _layer: object = PrivateAttr()
    _bins: np.ndarray = PrivateAttr(
        default_factory=lambda: np.array([0.0, 1.0])
    )
    _counts: np.ndarray = PrivateAttr(default_factory=lambda: np.array([0.0]))
    _dirty: bool = PrivateAttr(default=True)

    def __init__(
        self,
        layer: Image,
        n_bins: int = 256,
        mode: Literal['displayed', 'full'] = 'displayed',
        log_scale: bool = False,
        enabled: bool = True,
    ):
        """Initialize histogram model.

        Parameters
        ----------
        layer : Image
            The layer to compute histogram for.
        n_bins : int, default: 256
            Number of histogram bins.
        mode : {'displayed', 'full'}, default: 'displayed'
            Whether to compute histogram from displayed data or full volume.
        log_scale : bool, default: False
            Use logarithmic scale for histogram counts.
        enabled : bool, default: True
            Whether histogram is enabled (computes on data changes).
        """
        super().__init__(
            n_bins=n_bins, mode=mode, log_scale=log_scale, enabled=enabled
        )

        # Set private attributes using object.__setattr__ to bypass pydantic
        object.__setattr__(self, '_layer', layer)
        object.__setattr__(self, '_bins', np.array([0.0, 1.0]))
        object.__setattr__(self, '_counts', np.array([0.0]))
        object.__setattr__(self, '_dirty', True)

        # Connect to layer events to trigger recomputation
        layer.events.data.connect(self._on_data_change)
        layer.events.contrast_limits_range.connect(self._on_range_change)

        # Connect to set_data event which fires on slice changes
        if hasattr(layer.events, 'set_data'):
            layer.events.set_data.connect(self._on_slice_change)

        # Connect to loaded event for async slicing support
        # This fires when async slice loading completes
        if hasattr(layer.events, 'loaded'):
            layer.events.loaded.connect(self._on_loaded_change)

        # Connect to our own events to trigger recomputation
        self.events.n_bins.connect(self._on_params_change)
        self.events.mode.connect(self._on_params_change)
        self.events.log_scale.connect(self._on_log_scale_change)

        # Initial computation
        if self.enabled:
            self.compute()

    @property
    def bins(self) -> np.ndarray:
        """Histogram bin edges.

        Returns
        -------
        np.ndarray
            Array of bin edges with length n_bins + 1.
        """
        if self._dirty and self.enabled:
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
        if self._dirty and self.enabled:
            self.compute()
        return self._counts

    def compute(self) -> None:
        """Compute histogram from layer data.

        This method extracts data from the layer based on the current mode
        (displayed or full), samples if necessary, and computes the histogram.
        """
        # Get data based on mode
        data = self._get_data()

        if data is None or data.size == 0:
            self._bins = np.array([0.0, 1.0])
            self._counts = np.array([0.0])
            self._dirty = False
            self.events.bins()
            self.events.counts()
            return

        # Sample if data is too large (>1M points)
        if data.size > 1_000_000:
            data = self._sample_data(data, max_samples=1_000_000)

        # Get histogram range from contrast limits range
        range_min, range_max = self._layer.contrast_limits_range

        # Handle edge case where min == max
        if range_min == range_max:
            range_min = float(range_min) - 0.5
            range_max = float(range_max) + 0.5

        # Compute histogram
        counts, bins = np.histogram(
            data.ravel(),
            bins=self.n_bins,
            range=(float(range_min), float(range_max)),
        )

        # Store original counts (before log transform)
        self._bins = bins.astype(np.float32)

        # Apply log scale if needed
        if self.log_scale:
            self._counts = np.log10(counts + 1).astype(np.float32)
        else:
            self._counts = counts.astype(np.float32)

        self._dirty = False
        self.events.bins()
        self.events.counts()

    def _get_data(self) -> Optional[np.ndarray]:
        """Get data from layer based on current mode.

        Returns
        -------
        np.ndarray | None
            Data array to compute histogram from.
        """
        if self.mode == 'displayed':
            return self._get_displayed_data()
        return self._get_full_data()

    def _get_displayed_data(self) -> Optional[np.ndarray]:
        """Get data from currently displayed slice.

        In 'displayed' mode, the histogram is computed from the visible data
        that has already been sliced for rendering. This uses the layer's
        internal _slice.image.raw which contains the data being displayed.

        This provides a histogram of what the user is actually seeing,
        which is most useful for adjusting contrast limits. It also
        correctly handles multiscale data by using the appropriate
        resolution level that is currently being rendered.

        Returns
        -------
        np.ndarray | None
            Data from displayed slice only.
        """
        # Use the already-sliced data from the rendering pipeline
        # This correctly handles multiscale, dask, and async loading
        if (
            hasattr(self._layer, '_slice')
            and self._layer._slice is not None
            and hasattr(self._layer._slice, 'image')
            and self._layer._slice.image is not None
            and hasattr(self._layer._slice.image, 'raw')
        ):
            raw = self._layer._slice.image.raw
            if raw is not None:
                return np.asarray(raw)

        # Fallback: if slice not available, use full data
        # This can happen before first render
        return self._get_full_data()

    def _get_full_data(self) -> Optional[np.ndarray]:
        """Get full volume data.

        For multiscale data, uses the lowest resolution level (like
        contrast limit calculations) for efficiency.

        Returns
        -------
        np.ndarray | None
            Full volume data.
        """
        data = self._layer.data
        # Handle multiscale - use lowest resolution level for efficiency
        # This matches the pattern in _calc_data_range
        if isinstance(data, list | tuple):
            data = data[-1]
        return np.asarray(data)

    def _sample_data(self, data: np.ndarray, max_samples: int) -> np.ndarray:
        """Randomly sample data to reduce computation.

        Parameters
        ----------
        data : np.ndarray
            Data array to sample from.
        max_samples : int
            Maximum number of samples to take.

        Returns
        -------
        np.ndarray
            Sampled data array.
        """
        flat_data = data.ravel()
        # Remove NaN and inf values before sampling
        valid_mask = np.isfinite(flat_data)
        valid_data = flat_data[valid_mask]

        if valid_data.size == 0:
            return np.array([])

        if valid_data.size <= max_samples:
            return valid_data

        # Random sampling without replacement
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
        """Called when the displayed slice changes (slice navigation or 2D/3D toggle).

        This is important for 'displayed' mode where we only compute histogram
        on the currently visible data.
        """
        if self.mode == 'displayed':
            self._mark_dirty()

    def _on_loaded_change(self) -> None:
        """Called when async slice loading completes.

        For async slicing, the slice data may not be available immediately.
        This event fires when loading completes, so we can update the histogram
        with the newly loaded slice data.
        """
        if self.mode == 'displayed' and getattr(self._layer, 'loaded', True):
            self._mark_dirty()

    def _on_params_change(self) -> None:
        """Called when n_bins or mode changes."""
        self._mark_dirty()

    def _on_log_scale_change(self) -> None:
        """Called when log_scale changes.

        For log scale, we can just retransform the counts without
        recomputing the histogram.
        """
        if not self._dirty:
            # Re-apply log transform to existing counts
            # We need to recompute to get original counts
            self._mark_dirty()

    def _mark_dirty(self) -> None:
        """Mark histogram as needing recomputation."""
        self._dirty = True
        if self.enabled:
            self.compute()

    def reset(self) -> None:
        """Reset histogram to default settings."""
        self.n_bins = 256
        self.log_scale = False
        self.mode = 'displayed'
        self.compute()
