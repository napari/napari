"""Histogram model for Image/Surface layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from napari.utils.events import Event, EventedModel
from napari.utils.histogram import (
    _make_empty,
    compute_histogram,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np

    from napari.layers import Layer


class HistogramModel(EventedModel):
    """Histogram model with controls for histogram generation."""

    bins: int = 256
    max_samples: int = 1_000_000
    mode: Literal['canvas', 'full'] = 'canvas'
    log_scale: bool = False

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.events.add(completed=Event, updated=Event)

    def compute_async(
        self, layer: Layer
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(bin_edges, counts)`` for ``layer`` using the model settings.

        This is a generator that can be used to update a representation asyncronously.
        At each step, it writes the updated data into layer.metadata['_computed_histogram'].

        Do not use in a separate thread (use _compute_async_no_events instead).
        """
        for bin_edges, counts in self._compute_async_no_events(layer):
            self.events.updated()
            yield bin_edges, counts
        self.events.completed()

    def compute(self, layer: Any) -> tuple[np.ndarray, np.ndarray]:
        """Compute the full histogram for ``layer`` synchronously.

        Emits ``updated`` (once per chunk) and ``completed`` on the calling
        thread, so listeners (e.g. Qt controls) update accordingly.
        """
        bin_edges, counts = _make_empty().values()
        for res in self.compute_async(layer):
            bin_edges, counts = res
        return bin_edges, counts

    def _compute_async_no_events(
        self, layer: Layer
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """Non-evented generator to use in separate threads.

        Responsibility to fire events in the main thread lies on the caller.
        """
        for bin_edges, counts in compute_histogram(
            layer,
            bins=self.bins,
            max_samples=self.max_samples,
            mode=self.mode,
            log_scale=self.log_scale,
        ):
            layer.metadata['_computed_histogram'] = {
                'bin_edges': bin_edges,
                'counts': counts,
            }
            yield bin_edges, counts
