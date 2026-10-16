"""Reusable histogram content widget for histogram hosts."""

from __future__ import annotations

import contextlib
import logging
import threading
import weakref
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget

from napari._qt.widgets.qt_histogram_canvas import QtHistogramCanvasWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
from napari.settings import get_settings
from napari.utils.events import disconnect_events
from napari.utils.histogram import _make_empty
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    import numpy as np

    from napari.layers.intensity_mixin import IntensityVisualizationMixin
    from napari.utils.events import Event


logger = logging.getLogger(__name__)


class _HistogramWorker(QObject):
    yielded = Signal(object)
    finished = Signal(object)

    def __init__(self, layer: IntensityVisualizationMixin) -> None:
        super().__init__()
        self._layer_ref = weakref.ref(layer)
        self._request_event = threading.Event()
        self._terminate = False
        self._generation = 0
        self._gen_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def request(self) -> None:
        with self._gen_lock:
            self._generation += 1
        self._request_event.set()
        thread = self._thread
        if thread is None or not thread.is_alive():
            self._terminate = False
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def cancel(self) -> None:
        with self._gen_lock:
            self._generation += 1

    def stop(self) -> None:
        self._terminate = True
        with self._gen_lock:
            self._generation += 1
        self._request_event.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._terminate:
            self._request_event.wait()
            if self._terminate:
                return
            self._request_event.clear()
            while not self._terminate:
                with self._gen_lock:
                    generation = self._generation
                result = self._compute_once(generation)
                with self._gen_lock:
                    fresh = generation == self._generation
                if result is not None and fresh and not self._terminate:
                    self.finished.emit(result)
                if self._request_event.is_set():
                    self._request_event.clear()
                    continue
                break

    def _compute_once(
        self, generation: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        layer = self._layer_ref()
        if layer is None:
            return None
        try:
            gen = layer.histogram._compute_async_no_events(layer)
        except Exception:
            logger.warning('Histogram compute setup failed', exc_info=True)
            return None

        last: tuple[np.ndarray, np.ndarray] | None = None
        try:
            for bin_edges, counts in gen:
                with self._gen_lock:
                    stale = generation != self._generation
                if stale or self._terminate:
                    return None
                last = (bin_edges, counts)
                self.yielded.emit((bin_edges, counts))
        except Exception:
            logger.warning('Histogram compute failed', exc_info=True)
            return None
        return last


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
    histogram_canvas : QtHistogramCanvasWidget
        The histogram visualization widget.
    settings_widget : QtHistogramSettingsWidget
        Widget for mode and log scale controls.
    """

    def __init__(
        self,
        layer: IntensityVisualizationMixin,
        parent: QWidget | None = None,
        mirror: bool = False,
    ) -> None:
        super().__init__(parent)
        self.layer = layer
        self.bin_edges, self.counts = _make_empty().values()
        self._constructed = False
        self._mirror = mirror
        self._histogram_worker: _HistogramWorker | None = None
        self._appearance = get_settings().appearance

        if mirror:
            self.layer.histogram.events.updated.connect(
                self._on_mirrored_histogram
            )
            self.layer.histogram.events.completed.connect(
                self._on_mirrored_histogram
            )
        else:
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
        self._appearance.events.theme.connect(
            self._on_appearance_theme_changed
        )

        self.destroyed.connect(self._stop_histogram_worker)

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

        self.settings_widget = QtHistogramSettingsWidget(
            self.layer.histogram,
            parent=self,
        )
        layout.addWidget(self.settings_widget)

        self._constructed = True

    def _ensure_histogram_worker(self) -> _HistogramWorker:
        worker = self._histogram_worker
        if worker is None:
            worker = _HistogramWorker(self.layer)
            worker.yielded.connect(self._on_histogram_yield)
            worker.finished.connect(self._on_histogram_done)
            self._histogram_worker = worker
        return worker

    def _schedule_histogram_compute(self) -> None:
        self._ensure_histogram_content()
        self._ensure_histogram_worker().request()

    def _on_mirrored_histogram(self, event: Event) -> None:
        self.bin_edges = event.bin_edges
        self.counts = event.counts
        self.update_histogram()

    def _on_histogram_yield(self, args) -> None:
        self.bin_edges, self.counts = args
        self.layer.histogram.events.updated(
            bin_edges=self.bin_edges, counts=self.counts
        )
        self.update_histogram()

    def _on_histogram_done(self, args=None) -> None:
        if args is not None:
            self.bin_edges, self.counts = args
        self.layer.histogram.events.completed(
            bin_edges=self.bin_edges, counts=self.counts
        )
        self.update_histogram()

    def cancel_histogram_compute(self) -> None:
        if self._histogram_worker is not None:
            self._histogram_worker.cancel()

    def _stop_histogram_worker(self) -> None:
        worker = self._histogram_worker
        self._histogram_worker = None
        if worker is None:
            return
        with contextlib.suppress(RuntimeError, TypeError):
            worker.yielded.disconnect()
            worker.finished.disconnect()
        worker.stop()

    def _on_colormap_change(self):
        if self._constructed:
            self.histogram_canvas.update_colormap(self.layer.colormap)

    def _on_appearance_theme_changed(self, event: Event) -> None:
        if not self._constructed:
            return
        self.histogram_canvas._on_theme_change(get_theme(event.value))

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
        disconnect_events(self._appearance.events, self)
        self._stop_histogram_worker()

    def cleanup(self) -> None:
        """Disconnect event handlers and clean up child widgets."""
        disconnect_events(self.layer.histogram.events, self)
        disconnect_events(self.layer.events, self)
        disconnect_events(self._appearance.events, self)
        self._stop_histogram_worker()
        if self._constructed:
            self.settings_widget.cleanup()
            self.histogram_canvas.cleanup()
