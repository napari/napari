"""Histogram model for Image layer data visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
from pydantic import PrivateAttr

from napari.utils.events import EventedModel

if TYPE_CHECKING:
    from napari.layers.image.image import Image

__all__ = ('HistogramModel',)


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
    n_bins: int = 256
    mode: Literal['canvas', 'full'] = 'canvas'
    log_scale: bool = False
    enabled: bool = False

    # Private attributes (use PrivateAttr for pydantic).
    # Annotated as Any here because pydantic does not validate PrivateAttr
    # types at runtime; the actual type is Image (enforced by the __init__
    # signature and checked by static type-checkers via TYPE_CHECKING).
    _layer: Any = PrivateAttr()
    _bins: np.ndarray = PrivateAttr(
        default_factory=lambda: np.array([0.0, 1.0])
    )
    _counts: np.ndarray = PrivateAttr(default_factory=lambda: np.array([0.0]))
    _dirty: bool = PrivateAttr(default=True)

    def __init__(
        self,
        layer: Image,
        n_bins: int = 256,
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
        data = self._get_data()

        if data is None or data.size == 0:
            self._bins = np.array([0.0, 1.0])
            self._counts = np.array([0.0])
            self._dirty = False
            self.events.bins()
            self.events.counts()
            return

        # For RGB(A) images convert to luminance so the histogram represents
        # perceived brightness. For napari rgb=True images the channel axis is
        # always the last axis regardless of the number of leading dimensions
        # (e.g. TxHxWxC), so `data[..., :3]` is the correct slice.
        # There is no dedicated napari utility for this; calc_data_range uses
        # the same last-axis convention (via `offset = 2 + int(rgb)`).
        if self._layer.rgb:
            data = self._rgb_to_luminance(data)

        # Sample if data is too large (>1M points)
        if data.size > 1_000_000:
            data = self._sample_data(data, max_samples=1_000_000)

        # Get histogram range from contrast limits range
        range_min, range_max = self._layer.contrast_limits_range
        if range_min is None or range_max is None:
            range_min = float(np.nanmin(data))
            range_max = float(np.nanmax(data))

        # Handle edge case where min == max
        if range_min == range_max:
            range_min = float(range_min) - 0.5
            range_max = float(range_max) + 0.5

        counts, bins = np.histogram(
            data.ravel(),
            bins=self.n_bins,
            range=(float(range_min), float(range_max)),
        )

        self._bins = bins.astype(np.float32)

        if self.log_scale:
            self._counts = np.log10(counts + 1).astype(np.float32)
        else:
            self._counts = counts.astype(np.float32)

        self._dirty = False
        self.events.bins()
        self.events.counts()

    def _rgb_to_luminance(self, data: np.ndarray) -> np.ndarray:
        """Convert RGB(A) data to perceptual luminance.

        Uses ITU-R BT.709 coefficients so the result matches sRGB display
        brightness. Only the first three channels are used; alpha is ignored.
        The returned array has the same value range as the input (e.g. 0-255
        for uint8, 0-1 for float).
        """
        rgb = data[..., :3].astype(np.float32)
        return (
            0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        )

    def _get_data(self) -> Optional[np.ndarray]:
        """Get data from layer based on current mode."""
        if self.mode == 'canvas':
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
        """
        raw = self._get_slice_raw_data()
        if raw is not None and self._has_real_displayed_data(raw):
            return raw

        # Fallback: if slice not available, use full data
        return self._get_full_data()

    def _get_slice_raw_data(self) -> Optional[np.ndarray]:
        """Get the currently sliced raw image data if available."""
        if self._layer._slice is None:
            return None
        raw = self._layer._slice.image.raw
        return np.asarray(raw) if raw is not None else None

    def _has_real_displayed_data(self, raw: np.ndarray) -> bool:
        """Return True when sliced data is more than the placeholder sample."""
        if raw.size == 0:
            return False

        if self._layer.multiscale:
            return True

        displayed_shape = raw.shape[:-1] if self._layer.rgb else raw.shape
        if any(size != 1 for size in displayed_shape):
            return True

        full_data = self._get_full_data()
        if full_data is None:
            return True

        full_shape = (
            full_data.shape[:-1] if self._layer.rgb else full_data.shape
        )
        return all(size == 1 for size in full_shape)

    def _get_full_data(self) -> Optional[np.ndarray]:
        """Get full volume data.

        For multiscale data, uses the lowest resolution level for efficiency.
        """
        data = self._layer.data
        if isinstance(data, list | tuple):
            data = data[-1]
        return np.asarray(data)

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
        if not self._dirty:
            self._mark_dirty()

    def _on_enabled_change(self) -> None:
        """When enabled flips to True, compute if there is pending dirty data."""
        if self.enabled and self._dirty:
            self.compute()

    def _mark_dirty(self) -> None:
        """Mark histogram as needing recomputation.

        If enabled, compute immediately so connected widgets stay live.
        If disabled, defer — the next explicit access or enabled=True will
        trigger the compute.
        """
        self._dirty = True
        if self.enabled:
            self.compute()

    def reset(self) -> None:
        """Reset histogram to default settings."""
        self.n_bins = 256
        self.log_scale = False
        self.mode = 'canvas'
        self.compute()
