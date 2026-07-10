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
        # Progressive partials are broadcast on a render-only channel so this
        # view animates chunk-by-chunk even when another view owns the worker.
        self._histogram.events.partial_computed.connect(self._update_histogram)

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

        At most one worker runs per model.  The inline histogram and the
        popup share one model, so the first view to reach here claims the
        compute — flipping the model's Qt-free ``_compute_scheduled`` flag on
        the main thread — and owns the worker.  The other view returns and
        instead animates from ``events.partial_computed`` broadcasts,
        rendering the final result on ``events.counts()``.  The owner may
        abort and restart its own worker (e.g. on a parameter change) to
        recompute with fresh settings.
        """
        if self._compute_worker is not None:
            # We own the in-flight worker: abort and restart so a parameter
            # change recomputes with fresh settings (we stay the owner, so
            # ``_compute_scheduled`` stays True throughout).
            self._abort_worker(self._compute_worker)
            self._compute_worker = None
        elif self._histogram._compute_scheduled:
            # Another view drives the shared compute; it reacts to the same
            # model event and (re)starts as needed.  We animate from its
            # partial_computed broadcasts and render on events.counts().
            return

        worker = cast(
            GeneratorWorker,
            create_worker(
                self._histogram.compute,  # type: ignore[arg-type]
                _progress={'desc': 'Computing histogram'},
            ),
        )
        worker.yielded.connect(self._on_partial_histogram)
        worker.finished.connect(self._on_async_compute_done)
        self._compute_worker = worker
        self._histogram._compute_scheduled = True
        worker.start()

    def _abort_worker(self, worker: GeneratorWorker) -> None:
        """Disconnect our slots from *worker* and ask its generator to stop.

        Only disconnects **our** specific slots (``_on_async_compute_done``,
        ``_on_partial_histogram``), leaving napari's built-in handlers
        (task status cleanup, progress bar close) attached so they fire
        normally when the aborted worker's thread pool thread finishes its
        current chunk and exits.

        Resets the model's ``_computing`` re-entrancy guard and
        ``_compute_scheduled`` flag so a replacement compute can proceed.
        The aborted generator's ``finally`` also clears ``_computing``
        (harmless double reset).

        ``_compute_scheduled`` is cleared here because we disconnected
        ``_on_async_compute_done``, so that callback (which normally
        clears ``_compute_scheduled``) will never fire for this worker.
        Without this clear, the flag would leak as ``True`` forever,
        preventing any future async compute for the model.
        """
        worker.finished.disconnect(self._on_async_compute_done)
        worker.yielded.disconnect(self._on_partial_histogram)
        pbar = getattr(worker, 'pbar', None)
        if pbar is not None:
            pbar.close()
        worker.quit()
        self._histogram._computing = False
        self._histogram._compute_scheduled = False

    def _on_partial_histogram(
        self, bins_counts: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Publish a partial histogram result and broadcast it to all views.

        Called on the main thread each time the background generator yields
        ``(bins, counts)`` — only on the view that owns the worker.  We write
        the partial into the shared model and emit ``partial_computed`` so
        that every view (this one and, e.g., the inline histogram or the
        popup) re-renders the same snapshot in lockstep.  This decouples
        *which view drives the worker* from *which views animate*, so the
        visible view always updates even when the other one owns the worker.

        Yields from a stale worker that was aborted mid-compute are
        discarded when the model already has clean data (e.g. after a
        canvas-mode sync compute that bumped the generation).  Without
        this guard, a full-mode chunk that completed after the mode
        switch would overwrite the canvas histogram with stale data.
        """
        if self._cleaned_up or not self._histogram._dirty:
            return
        bins, counts = bins_counts
        self._histogram._bin_edges = bins
        self._histogram._counts = counts
        self._histogram.events.partial_computed()

    def _on_async_compute_done(self, _: Any = None) -> None:
        """Called on the main thread when our background compute finishes.

        Releases the shared compute slot and emits ``events.counts()`` so
        every view (this widget and any other sharing the model, e.g. the
        popup vs the inline histogram) re-reads the freshly computed data.
        Only the owning view connects this slot, so reaching here means we
        held the compute.

        ``finished`` fires whether ``compute()`` succeeded, raised (e.g. a
        remote chunk failed to load), or was superseded by a newer
        generation — in every case the model clears ``_dirty`` only on a
        clean result.  So a still-dirty model here means there is nothing
        new to publish; emitting ``events.counts()`` anyway would re-enter
        ``_on_model_event`` (still dirty + enabled) and spawn a replacement
        worker, i.e. retry a persistent failure in a tight loop.  The error
        itself is already surfaced to the user by the worker's notification
        mixin (see ``napari._qt.qthreading``), so we simply stop here.
        """
        if self._cleaned_up:
            return
        self._compute_worker = None
        self._histogram._compute_scheduled = False

        if self._histogram._dirty:
            return

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
        # during teardown.  This also stops *this* widget from reacting to
        # the counts() nudge emitted below, so only surviving views do.
        disconnect_events(self._histogram.events, self)
        disconnect_events(self.layer.events, self)
        disconnect_events(self._appearance.events, self)

        # Abort the worker if we own one.  If the generator is between
        # chunks it exits on the next iteration; if it's mid-chunk (blocking
        # on I/O) the thread pool terminates it at shutdown — the
        # _cleaned_up guard in _on_async_compute_done and
        # _on_partial_histogram prevents stale callbacks either way.  Each
        # view only ever holds its own worker, so there's no shared handle
        # to reconcile.
        owned_unfinished = False
        worker = self._compute_worker
        self._compute_worker = None
        if worker is not None:
            self._abort_worker(worker)
            self._histogram._compute_scheduled = False
            # If the compute had not finished (still dirty), our aborted
            # worker will never emit counts(), so a surviving view must
            # restart the load.
            owned_unfinished = self._histogram._dirty

        self.histogram_visual.destroy()
        self.canvas.close()

        # Nudge any surviving view (e.g. the inline histogram when the popup
        # closes mid-load) to take over the compute we were driving.  Our
        # own model-event subscriptions are already gone, so this reaches
        # only other views — or no-one, harmlessly.
        if owned_unfinished and self._histogram.enabled:
            self._histogram.events.counts()
