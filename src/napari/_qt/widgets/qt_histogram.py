"""Reusable histogram content widget for histogram hosts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QVBoxLayout, QWidget

from napari._qt.widgets.qt_histogram_canvas import QtHistogramCanvasWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
from napari.qt import create_worker
from napari.settings import get_settings
from napari.utils.events import disconnect_events
from napari.utils.histogram import _make_empty
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np

    from napari._qt.qthreading import FunctionWorker, GeneratorWorker
    from napari.layers.intensity_mixin import IntensityVisualizationMixin


class QtHistogramWidget(QWidget):
    """Shared histogram visualization and settings content.

    Parameters
    ----------
    layer : Image
        The napari Image layer to visualize.
    parent : QWidget, optional
        Parent widget.

    Attributes
    ----------
    histogram_widget : QtHistogramWidget
        The histogram visualization widget.
    settings_widget : QtHistogramSettingsWidget
        Widget for mode and log scale controls.
    """

    def __init__(
        self,
        layer: IntensityVisualizationMixin,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.layer = layer
        self.bin_edges, self.counts = _make_empty().values()
        self._constructed = False
        self._histogram_worker: GeneratorWorker | FunctionWorker | None = None

        for ev in (
            self.layer.histogram.events.bins,
            self.layer.histogram.events.max_samples,
            self.layer.histogram.events.mode,
            self.layer.histogram.events.log_scale,
            self.layer.events.data,
            self.layer.events.contrast_limits_range,
            self.layer.events.set_data,
        ):
            ev.connect(self._schedule_histogram_compute)

        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.contrast_limits.connect(self.update_histogram)
        self.layer.events.gamma.connect(self.update_histogram)

        self.destroyed.connect(self._abort_histogram_worker)

    def _ensure_histogram_content(self) -> None:
        """Lazily create the histogram content widget (vispy canvas).

        Sometimes necessary because opengl requires a preexisting context.
        """
        if self._constructed:
            return

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.setLayout(layout)
        self.histogram_canvas = QtHistogramCanvasWidget(
            parent=self,
        )
        layout.addWidget(self.histogram_canvas)
        self.histogram_canvas.update_colormap(self.layer.colormap)
        theme = get_theme(get_settings().appearance.theme)
        self.histogram_canvas._on_theme_change(theme=theme)

        self.settings_widget = QtHistogramSettingsWidget(
            self.layer.histogram,
            parent=self,
        )
        layout.addWidget(self.settings_widget)

        self._constructed = True

    def _yield_from_histogram(
        self,
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
        yield from self.layer.histogram._compute_async_no_events(self.layer)

    def _schedule_histogram_compute(self) -> None:
        """Run the async histogram compute."""
        self._ensure_histogram_content()
        self._abort_histogram_worker()
        worker = create_worker(self._yield_from_histogram)  # pyrefly: ignore [bad-argument-type]
        self._histogram_worker = worker

        worker.yielded.connect(self._on_histogram_yield)
        worker.finished.connect(self._on_histogram_done)
        worker.start()

    def _on_histogram_yield(self, args) -> None:
        """Emit ``yielded`` on the main thread once the worker finishes."""
        self.bin_edges, self.counts = args
        self.layer.histogram.events.updated(
            bin_edges=self.bin_edges, counts=self.counts
        )
        self.update_histogram()

    def _on_histogram_done(self) -> None:
        """Emit ``completed`` on the main thread once the worker finishes."""
        self.layer.histogram.events.completed(
            bin_edges=self.bin_edges, counts=self.counts
        )
        self.update_histogram()
        self._abort_histogram_worker()

    def _abort_histogram_worker(self) -> None:
        """Stop any in-flight compute worker."""
        worker = self._histogram_worker
        self._histogram_worker = None
        if worker is None:
            return
        worker.yielded.disconnect()
        worker.finished.disconnect()
        # progress bar should be stopped if present
        pbar = getattr(worker, 'pbar', None)
        if pbar is not None:
            pbar.close()
        worker.quit()

    def _on_colormap_change(self):
        if self._constructed:
            self.histogram_canvas.update_colormap(self.layer.colormap)

    def update_histogram(self):
        if self._constructed:
            self.histogram_canvas.histogram_visual.set_data(
                bin_edges=self.bin_edges,
                counts=self.counts,
                gamma=self.layer.gamma,
                clims=self.layer.contrast_limits,
                data_range=self.layer.contrast_limits_range,
            )
            self.histogram_canvas.canvas.update()

    def disconnect_widget_controls(self) -> None:
        disconnect_events(self.layer.histogram.events, self)
        disconnect_events(self.layer.events, self)

    def cleanup(self) -> None:
        """Disconnect event handlers and clean up child widgets."""
        if self._constructed:
            self._abort_histogram_worker()
            self.settings_widget.cleanup()
            self.histogram_canvas.cleanup()
