"""Histogram model for Image/Surface layers.

Stores only histogram settings. Computation is delegated to
``napari.utils.histogram.compute_histogram`` and results are written to
``layer.metadata['_computed_histogram']`` by the caller/Qt controls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from napari.utils.events import Event, EventedModel
from napari.utils.histogram import (
    DEFAULT_BINS,
    DEFAULT_MAX_SAMPLES,
    compute_histogram,
    get_computed,
)

if TYPE_CHECKING:
    import numpy as np


__all__ = ('HistogramModel',)


class HistogramModel(EventedModel):
    bins: int = DEFAULT_BINS
    max_samples: int = DEFAULT_MAX_SAMPLES
    mode: Literal['canvas', 'full'] = 'canvas'
    log_scale: bool = False

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.events.add(completed=Event, updated=Event)

    def compute_async(self, layer: Any) -> Any:
        """Yield ``(bin_edges, counts)`` for ``layer`` using the model settings.

        This is a generator; Qt drives it step-by-step (e.g. via a worker)
        so progressive results can be displayed. Each step writes the
        latest result to ``layer.metadata['_computed_histogram']``.
        """
        yield from compute_histogram(
            layer,
            bins=self.bins,
            max_samples=self.max_samples,
            mode=self.mode,
            log_scale=self.log_scale,
        )

    def compute(self, layer: Any) -> None:
        """Compute the histogram for ``layer`` synchronously.

        Iterates ``compute_async`` to completion; results end up in
        ``layer.metadata['_computed_histogram']``.
        """
        for _ in self.compute_async(layer):
            pass

    def get_computed(self, layer: Any) -> dict[str, np.ndarray]:
        return get_computed(layer)
