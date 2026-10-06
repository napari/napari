"""Reusable histogram content widget for histogram hosts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QVBoxLayout, QWidget

from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
from napari.qt import create_worker
from napari.utils.events import disconnect_events

if TYPE_CHECKING:
    from napari._qt.qthreading import GeneratorWorker
    from napari.layers.intensity_mixin import IntensityVisualizationMixin


class QtHistogramContentWidget(QWidget):
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
        self._constructed = False
        self._histogram_worker: GeneratorWorker | None = None

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
        self.histogram_widget = QtHistogramWidget(
            self.layer,
            parent=self,
        )
        layout.addWidget(self.histogram_widget)

        self.settings_widget = QtHistogramSettingsWidget(
            self.layer.histogram,
            parent=self,
        )
        layout.addWidget(self.settings_widget)

        self._constructed = True

    def _yield_histogram(self):
        yield from self.layer.histogram._compute_async_no_events(self.layer)

    def _schedule_histogram_compute(self, event=None) -> None:
        """Run the async histogram compute."""
        self._ensure_histogram_content()
        self._abort_histogram_worker()
        worker = create_worker(self._yield_histogram)
        self._histogram_worker = worker

        worker.yielded.connect(lambda _: self.layer.histogram.events.updated)
        worker.finished.connect(self._on_histogram_done)
        worker.start()

    def _on_histogram_done(self) -> None:
        """Emit ``completed`` on the main thread once the worker finishes."""
        self.layer.histogram.events.completed()
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

    def disconnect_widget_controls(self) -> None:
        disconnect_events(self.layer.histogram.events, self)

    def cleanup(self) -> None:
        """Disconnect event handlers and clean up child widgets."""
        self._abort_histogram_worker()
        self.settings_widget.cleanup()
        self.histogram_widget.cleanup()
