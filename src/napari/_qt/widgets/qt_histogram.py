"""Qt widget for displaying histogram using vispy visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from qtpy.QtWidgets import QVBoxLayout, QWidget
from vispy.scene import SceneCanvas

from napari._qt.qthreading import GeneratorWorker, create_worker
from napari._vispy.visuals.histogram import HistogramVisual
from napari.settings import get_settings
from napari.utils.events.event_utils import disconnect_events
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from pydantic_extra_types.color import Color

    from napari.layers import Image
    from napari.utils.events import Event

# Default histogram canvas dimensions
_DEFAULT_CANVAS_WIDTH = 300
_DEFAULT_CANVAS_HEIGHT = 150
_DEFAULT_CANVAS_MIN_HEIGHT = 100


class QtHistogramWidget(QWidget):
    """
    Qt widget that embeds a vispy histogram visualization.

    This widget wraps a HistogramVisual in a vispy SceneCanvas,
    providing a Qt-compatible widget that can be embedded in
    layer controls.

    For chunked arrays in full mode, computation runs in a background
    thread via :func:`napari._qt.qthreading.create_worker` to keep
    the UI responsive during I/O-bound loads (e.g. remote S3 zarr).

    Parameters
    ----------
    layer : Image
        The napari Image layer to visualize.
    parent : QWidget, optional
        Parent widget.

    Attributes
    ----------
    canvas : SceneCanvas
        Vispy canvas containing the histogram visual.
    view : ViewBox
        Vispy view for the histogram.
    histogram_visual : HistogramVisual
        The vispy visual that renders the histogram.
    """

    def __init__(
        self,
        layer: Image,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.layer = layer
        self._histogram = layer.histogram
        self._appearance = get_settings().appearance
        self._updating = False
        self._cleaned_up: bool = False
        self._compute_worker: GeneratorWorker | None = None

        theme = get_theme(self._appearance.theme)

        # Create vispy canvas
        self.canvas = SceneCanvas(
            size=(_DEFAULT_CANVAS_WIDTH, _DEFAULT_CANVAS_HEIGHT),
            bgcolor=theme.canvas.as_hex(),
            keys=None,
        )
        self.canvas.native.setParent(self)
        self.canvas.native.setMinimumHeight(_DEFAULT_CANVAS_MIN_HEIGHT)

        from vispy.scene import ViewBox

        self.view = ViewBox(parent=self.canvas.scene)
        self.canvas.central_widget.add_widget(self.view)

        self.histogram_visual = HistogramVisual()
        self.histogram_visual.parent = self.view.scene

        self.view.camera = 'panzoom'
        self.view.camera.set_range(x=(0, 1), y=(0, 1), margin=0.01)
        # Disable viewbox interaction to prevent accidental pan/zoom
        self.view.interactive = False

        # Layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.canvas.native)
        self.setLayout(main_layout)

        # Connect model events to a single handler that decides whether
        # to re-render or trigger async compute, depending on whether the
        # model computed synchronously or deferred for chunked full data.
        self._histogram.events.counts.connect(self._on_model_event)
        self._histogram.events.enabled.connect(self._on_model_event)
        self._histogram.events.log_scale.connect(self._on_model_event)
        self._histogram.events.mode.connect(self._on_model_event)
        self._histogram.events.bins.connect(self._on_model_event)
        self._histogram.events.max_samples.connect(self._on_model_event)

        # Connect to layer events that affect visualization
        layer.events.gamma.connect(self._update_histogram)
        layer.events.contrast_limits.connect(self._update_histogram)
        layer.events.colormap.connect(self._on_colormap_change)
        self._appearance.events.theme.connect(self._on_theme_change)

        # Initial update
        self._apply_visual_style()
        self._update_histogram()

    def _on_colormap_change(self, event: Event | None = None) -> None:
        """Update histogram colors when the layer colormap changes."""
        self._apply_visual_style()
        self._update_histogram()

    def _on_model_event(self) -> None:
        """Respond to model property changes — re-render or trigger async compute.

        The model's own event handlers (``_mark_dirty``,
        ``_on_log_scale_change``) either compute synchronously (clearing
        ``_dirty`` and emitting ``events.counts()``), or defer for chunked
        full-mode data (leaving ``_dirty=True``).  This handler picks up
        deferred work via async compute, and otherwise just re-renders the
        fresh data already in the model.
        """
        if self._histogram._dirty and self._histogram.enabled:
            self._ensure_histogram_computed()
        else:
            self._update_histogram()

    def _on_theme_change(self, event: Event | None = None) -> None:
        """Update canvas and plot styling when the application theme changes.

        Uses ``event.value`` (the new theme) when available, falling back
        to ``settings.appearance.theme``.  This handles both the settings
        path (``settings.appearance.events.theme``) and the viewer path
        (``viewer.events.theme`` from ``toggle_theme`` keybinding, where
        settings are intentionally not updated).
        """
        theme_name = (
            event.value if event is not None else self._appearance.theme
        )
        self._apply_visual_style(theme_name=theme_name)
        self.canvas.update()

    def _ensure_histogram_computed(self, event: Event | None = None) -> None:
        """Trigger histogram computation in a background thread.

        Uses a ``GeneratorWorker`` so intermediate results are yielded
        progressively for chunked data (e.g. remote zarr), while
        non-chunked data completes with a single yield.
        """
        # Cancel any in-flight worker
        if self._compute_worker is not None:
            # Disconnect both signals so no stale callbacks fire.
            self._compute_worker.finished.disconnect(
                self._on_async_compute_done
            )
            self._compute_worker.yielded.disconnect(self._on_partial_histogram)

            self._compute_worker.quit()
            self._compute_worker = None
            # The old worker's generator holds _computing=True, which would
            # block both _mark_dirty's inline compute and any new worker's
            # compute via the reentrancy guard.  Reset the flag so the new
            # worker can proceed.  The old worker's finally block will
            # redundantly set _computing=False on exit — harmless.
            self._histogram._computing = False

        worker = create_worker(self._histogram.compute)  # type: ignore[arg-type]
        worker.yielded.connect(self._on_partial_histogram)
        worker.finished.connect(self._on_async_compute_done)
        worker.start()
        self._compute_worker = cast(GeneratorWorker, worker)

    def _on_partial_histogram(
        self, bins_counts: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Update the vispy canvas with partial histogram data from a chunk.

        Called on the main thread each time the background generator yields
        ``(bins, counts)``.  We skip the re-entrancy guard since this path
        is never triggered by model-event handlers (they are disconnected
        during the background compute).
        """
        bins, counts = bins_counts
        gamma = self.layer.gamma
        clims = self.layer.contrast_limits
        clims_range = self.layer.contrast_limits_range

        self.histogram_visual.set_data(
            bin_edges=bins,
            counts=counts,
            gamma=gamma,
            clims=clims,
            data_range=clims_range,
        )
        self.canvas.update()

    def _on_async_compute_done(self, _: Any = None) -> None:
        """Called on the main thread when background compute finishes.

        Emits ``events.counts()`` so the histogram visual re-reads the
        freshly computed histogram data from the model.
        """
        if self._cleaned_up:
            return
        self._compute_worker = None

        self._histogram.events.counts()

    def _theme_rgba(
        self, color: Color, alpha: float = 1.0
    ) -> tuple[float, float, float, float]:
        """Convert a napari theme color to a vispy RGBA tuple."""
        # color is returned as a 4-tuple even when alpha=False,
        # so to satisfy mypy we need to explicity build a 3-tuple
        rgb = color.as_rgb_tuple(alpha=False)
        red, green, blue = rgb[0], rgb[1], rgb[2]
        return (red / 255, green / 255, blue / 255, alpha)

    def _layer_bar_color(self) -> tuple[float, float, float, float]:
        """Use end of colormap for histogram bars.

        Picking the almost highest end (``map([0.8])``) avoids invisibility on
        dark canvas with reversed colormaps like ``gray_r``.
        """
        rgba = np.atleast_2d(self.layer.colormap.map([0.8]))[0].astype(float)
        alpha = max(float(rgba[3]), 0.8)
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]), alpha)

    def _apply_visual_style(self, theme_name: str | None = None) -> None:
        """Apply theme-aware and layer-aware styling to the histogram plot.

        Parameters
        ----------
        theme_name : str, optional
            Theme name to apply. If not provided, reads from
            ``settings.appearance.theme``.
        """
        if theme_name is None:
            theme_name = self._appearance.theme
        theme = get_theme(theme_name)
        self.canvas.bgcolor = theme.canvas.as_hex()
        self.histogram_visual.set_style(
            bar_color=self._layer_bar_color(),
            lut_color=self._theme_rgba(theme.highlight, 0.95),
            axes_color=self._theme_rgba(theme.text, 0.7),
        )

    def _update_histogram(self, event: Event | None = None) -> None:
        """Update the histogram visual with current data.

        Accepts an optional event argument so it can be connected directly
        to psygnal events without a wrapper.

        Reads ``_bin_edges`` and ``_counts`` directly (the private
        attributes) to guarantee this method never triggers a synchronous
        ``compute()``.  The private attributes always hold the last
        computed (or default) values, and callers always invoke this
        method after a compute has completed.
        """
        if self._updating:
            return

        self._updating = True
        try:
            if not self._histogram.enabled:
                self.histogram_visual.set_data()
                self.canvas.update()
                return

            bin_edges = self._histogram._bin_edges
            counts = self._histogram._counts

            gamma = self.layer.gamma
            clims = self.layer.contrast_limits
            clims_range = self.layer.contrast_limits_range

            self.histogram_visual.set_data(
                bin_edges=bin_edges,
                counts=counts,
                gamma=gamma,
                clims=clims,
                data_range=clims_range,
            )
            self.canvas.update()
        finally:
            self._updating = False

    def cleanup(self) -> None:
        """Disconnect event handlers and clean up resources."""
        self._cleaned_up = True

        # Disconnect events first to prevent new computation triggers
        # during teardown.
        disconnect_events(self._histogram.events, self)
        disconnect_events(self.layer.events, self)
        disconnect_events(self._appearance.events, self)

        # Request abort from the worker.  If the generator is between
        # chunks, it will exit on the next loop iteration.  If it's
        # mid-chunk (blocking on I/O) the thread pool will terminate
        # it at shutdown — the _cleaned_up guard in
        # _on_async_compute_done prevents stale callbacks.
        if self._compute_worker is not None:
            self._compute_worker.finished.disconnect(
                self._on_async_compute_done
            )
            self._compute_worker.quit()
            self._compute_worker = None

        self.histogram_visual.destroy()
        self.canvas.close()
