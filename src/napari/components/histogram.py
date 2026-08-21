"""Histogram model for Image/Surface layers."""

from __future__ import annotations

from typing import Any, Literal

from napari.utils.events import Event, EventedModel
from napari.utils.histogram import (
    compute_histogram,
)


class HistogramModel(EventedModel):
    """Histogram model with controls for histogram generation."""

    bins: int = 256
    max_samples: int = 1_000_000
    mode: Literal['canvas', 'full'] = 'canvas'
    log_scale: bool = False

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.events.add(completed=Event, updated=Event)

    def compute_async(self, layer: Any) -> Any:
        """Yield ``(bin_edges, counts)`` for ``layer`` using the model settings.

        This is a generator that can be used to update a representation asyncronously.
        At each step, it writes the updated data into layer.metadata['_computed_histogram'].
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
            self.events.updated()
            yield bin_edges, counts
        self.events.completed()

    def compute(self, layer: Any) -> None:
        """Compute the full histogram for ``layer`` synchronously."""
        with self.events.updated.blocker():
            for _ in self.compute_async(layer):
                pass
