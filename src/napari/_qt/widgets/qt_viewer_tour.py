from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, TypeAlias, cast

from qt_tour import GuidedTour, TourAnchor, TourStep
from qtpy.QtWidgets import QWidget

from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from collections.abc import Callable

    from napari._qt.qt_main_window import Window, _QtMainWindow

    TourViewerWindow: TypeAlias = Window | _QtMainWindow


def _as_main_window(window: TourViewerWindow) -> _QtMainWindow:
    qt_window = getattr(window, '_qt_window', window)
    if not hasattr(qt_window, '_qt_viewer'):
        raise TypeError('Tour needs a napari window.')
    return cast('_QtMainWindow', qt_window)


_BUILTIN_TOUR_TARGETS: dict[str, Callable[[_QtMainWindow], QWidget | None]] = {
    'canvas': lambda qt_window: qt_window._qt_viewer.canvas.native,
    'layer_list': lambda qt_window: qt_window._qt_viewer.dockLayerList,
    'layer_buttons': lambda qt_window: qt_window._qt_viewer.layerButtons,
    'layer_controls': lambda qt_window: qt_window._qt_viewer.dockLayerControls,
    'viewer_buttons': lambda qt_window: qt_window._qt_viewer.viewerButtons,
    'dims': lambda qt_window: qt_window._qt_viewer.dims,
    'status_bar': lambda qt_window: qt_window.statusBar(),
}


def resolve_tour_target(
    qt_window: TourViewerWindow, name: str
) -> QWidget | None:
    """Resolve a tour step target by name.

    Checks napari's built-in viewer regions first (``'canvas'``,
    ``'layer_list'``, ``'layer_buttons'``, ``'layer_controls'``,
    ``'viewer_buttons'``, ``'dims'``, ``'status_bar'``), then falls back to
    ``window.dock_widgets``, so a plugin can target a widget it docked by
    the name it registered it under, without reaching into ``qt_viewer``.
    """
    qt_window = _as_main_window(qt_window)
    builtin = _BUILTIN_TOUR_TARGETS.get(name)
    if builtin is not None:
        return builtin(qt_window)
    widget = qt_window._window.dock_widgets.get(name)
    if widget is None or isinstance(widget, QWidget):
        return widget
    return widget.native


def build_viewer_tour(
    window: TourViewerWindow,
    *,
    sample: tuple[str, str] | None = ('napari', 'balls_3d'),
) -> GuidedTour:
    """Build napari's built-in guided viewer tour.

    Parameters
    ----------
    window : napari._qt.qt_main_window.Window or qtpy.QtWidgets.QWidget
        The napari window to build the tour for.
    sample : tuple[str, str] | None, default: ('napari', 'balls_3d')
        Plugin and sample names to load via ``viewer.open_sample`` if the
        viewer has no layers yet, so the dims-slider step has something to
        show. Pass a different ``(plugin, sample)`` pair to demo the tour
        against other data, or ``None`` to skip loading sample data
        entirely (e.g. when the viewer already has data loaded).
    """
    # Deferred to avoid a circular import: _help.py imports build_viewer_tour
    # at module level, so importing HELP_URLS from it at this module's own
    # top level would form a cycle.
    from napari._qt._qapp_model.qactions._help import HELP_URLS

    window = _as_main_window(window)
    qt_viewer = window._qt_viewer
    viewer = qt_viewer.viewer
    if sample is not None and not viewer.layers:
        viewer.open_sample(*sample)

    link_color = get_theme(viewer.theme).to_rgb_dict()['current']

    def target(name: str) -> Callable[[], QWidget | None]:
        return partial(resolve_tour_target, window, name)

    # A step whose target lives in a hidden dock would otherwise freeze the
    # tour: Next/Back silently no-op once a step's target isn't visible.
    # Reveal the relevant dock right as its step is about to be shown
    # (rather than up front), so a dock hidden *during* the tour is caught
    # too, and restore whichever docks we ended up revealing once the tour
    # finishes.
    shown_docks: list[QWidget] = []

    def reveal(dock: Callable[[], QWidget]) -> Callable[[], bool]:
        def _ensure_visible() -> bool:
            widget = dock()
            if widget.isVisible():
                return False
            shown_docks.append(widget)
            widget.show()
            return True

        return _ensure_visible

    reveal_layer_list = reveal(lambda: qt_viewer.dockLayerList)
    reveal_layer_controls = reveal(lambda: qt_viewer.dockLayerControls)

    tour = GuidedTour(
        [
            TourStep(
                target=target('canvas'),
                title='Welcome to napari',
                body=(
                    "This quick tour walks you through the viewer's main pieces. "
                    'You can interact with the viewer the whole time, and reopen '
                    'this tour from Help any time.'
                ),
                anchor=TourAnchor.CENTER,
            ),
            TourStep(
                target=target('canvas'),
                title='Explore the canvas',
                body=(
                    'The viewer canvas shows your layers. Drag to pan and scroll to zoom.'
                ),
                anchor=TourAnchor.BELOW,
            ),
            TourStep(
                target=target('layer_list'),
                title='Layer list',
                body=(
                    'Layers live here. Select one to edit it, rename it inline, change visibility, or reorder by dragging.'
                ),
                ensure_visible=reveal_layer_list,
            ),
            TourStep(
                target=target('layer_buttons'),
                title='Layer buttons',
                body=(
                    'These create or delete layers. They are the quickest way to add points, shapes, or labels on top of your data.'
                ),
                ensure_visible=reveal_layer_list,
            ),
            TourStep(
                target=target('layer_controls'),
                title='Layer controls',
                body=(
                    'The active layer decides what appears here. You will always find opacity and '
                    'blending controls, plus extra options specific to that layer type, for example '
                    'contrast limits for images or colors for points.'
                ),
                ensure_visible=reveal_layer_controls,
            ),
            TourStep(
                target=target('viewer_buttons'),
                title='Viewer buttons',
                body=(
                    'Use these for grid mode, 2D/3D display, axis order, and resetting the camera with the home button.'
                    ' Many UI elements have an indicator in the lower-right meaning that they can be right-clicked'
                    ' for advanced functionality.'
                ),
                ensure_visible=reveal_layer_list,
                anchor=TourAnchor.ABOVE,
            ),
            TourStep(
                target=target('dims'),
                title='Dimension sliders',
                body=(
                    'Extra dimensions show up here. Move through slices, or press play on a slider to animate along that axis.'
                ),
                anchor=TourAnchor.ABOVE,
                skip=lambda: viewer.dims.ndim <= viewer.dims.ndisplay,
            ),
            TourStep(
                target=target('status_bar'),
                title='Status bar',
                body=(
                    'The status bar reports cursor position, values under the mouse, and small context-sensitive hints while you interact.'
                ),
                anchor=TourAnchor.ABOVE,
            ),
            TourStep(
                target=target('canvas'),
                title="That's the tour",
                body=(
                    'Explore at your own pace, and check out the '
                    f'<a href="{HELP_URLS["getting_started"]}" style="color: {link_color};">'
                    'napari user guide</a> '
                    'whenever you want to go deeper.'
                ),
                anchor=TourAnchor.CENTER,
            ),
        ],
        window,
    )

    def _restore_docks() -> None:
        for dock in shown_docks:
            dock.hide()

    tour.finished.connect(_restore_docks)
    return tour
