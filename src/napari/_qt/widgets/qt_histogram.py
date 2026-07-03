"""Qt widget for displaying histogram using vispy visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from qtpy.QtWidgets import QVBoxLayout, QWidget
from vispy.scene import SceneCanvas

from napari._qt.qthreading import FunctionWorker, create_worker
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
        self._compute_worker: FunctionWorker | None = None

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

        # Connect to model events for live updates from sync compute
        self._histogram.events.bins.connect(self._on_histogram_change)
        self._histogram.events.counts.connect(self._on_histogram_change)
        self._histogram.events.log_scale.connect(self._on_histogram_change)
        self._histogram.events.enabled.connect(self._on_histogram_change)

        # Connect to layer events that affect visualization
        layer.events.gamma.connect(self._on_gamma_change)
        layer.events.contrast_limits.connect(self._on_clims_change)
        layer.events.colormap.connect(self._on_colormap_change)
        self._appearance.events.theme.connect(self._on_theme_change)

        self._apply_visual_style()

        # Initial update
        self._update_histogram()

    def _on_histogram_change(self, event: Event | None = None) -> None:
        """Update visualization when histogram data changes."""
        self._update_histogram()

    def _on_gamma_change(self, event: Event | None = None) -> None:
        """Update gamma curve when layer gamma changes."""
        self._update_histogram()

    def _on_clims_change(self, event: Event | None = None) -> None:
        """Update contrast limit indicators when they change."""
        self._update_histogram()

    def _on_colormap_change(self, event: Event | None = None) -> None:
        """Update histogram colors when the layer colormap changes."""
        self._apply_visual_style()
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

    def _ensure_histogram_computed(self) -> None:
        """Trigger histogram computation, using a thread for chunked data.

        For in-memory (numpy / small) data, ``compute()`` runs inline
        and events fire on the main thread as usual.

        For chunked arrays (dask / zarr) in full mode, the actual chunk
        I/O runs in a background thread so the UI stays responsive.
        While the thread runs, the widget's event connections are
        temporarily suspended to prevent vispy updates from the
        background thread; once the thread finishes, the widget reads
        the fresh results and reconnects.
        """
        # Cancel any in-flight worker
        if self._compute_worker is not None:
            self._compute_worker.quit()
            self._compute_worker = None

        if self._histogram.mode == 'full' and self._histogram._has_chunks(
            self.layer.data
        ):
            self._start_async_compute()
        else:
            # Sync path — events fire on the main thread as usual
            self._histogram.compute()

    def _start_async_compute(self) -> None:
        """Run histogram compute in a background thread.

        Disconnects event-driven vispy updates during the thread to
        avoid calling vispy from the background thread.  After the
        thread finishes, reconnects and reads the fresh results.
        """
        # Disconnect event-driven updates during thread
        self._histogram.events.bins.disconnect(self._on_histogram_change)
        self._histogram.events.counts.disconnect(self._on_histogram_change)

        def _work() -> None:
            self._histogram.compute()

        # mypy: create_worker expects FunctionType but accepts Callable at runtime
        worker = create_worker(_work)  # type: ignore[arg-type]
        worker.finished.connect(self._on_async_compute_done)
        worker.start()
        self._compute_worker = cast(FunctionWorker, worker)

    def _on_async_compute_done(self, _: Any = None) -> None:
        """Called on the main thread when background compute finishes.

        Reconnects event listeners and reads the freshly computed
        histogram data to update the vispy canvas.
        """
        self._compute_worker = None

        # Reconnect event-driven updates
        self._histogram.events.bins.connect(self._on_histogram_change)
        self._histogram.events.counts.connect(self._on_histogram_change)

        # Read fresh data and update vispy (safe on main thread)
        self._update_histogram()

    def _theme_rgba(
        self, color: Color, alpha: float = 1.0
    ) -> tuple[float, float, float, float]:
        """Convert a napari theme color to a vispy RGBA tuple."""
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

    def _update_histogram(self) -> None:
        """Update the histogram visual with current data."""
        if self._updating:
            return

        self._updating = True
        try:
            hist = self._histogram
            if not hist.enabled:
                self.histogram_visual.set_data()
                self.canvas.update()
                return

            bins = hist.bins
            counts = hist.counts

            gamma = self.layer.gamma
            clims = self.layer.contrast_limits
            clims_range = self.layer.contrast_limits_range

            self.histogram_visual.set_data(
                bins=bins,
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
        if self._compute_worker is not None:
            self._compute_worker.quit()
            self._compute_worker = None
        disconnect_events(self._histogram.events, self)
        disconnect_events(self.layer.events, self)
        disconnect_events(self._appearance.events, self)

        self.histogram_visual.destroy()
        self.canvas.close()
