from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, QObject, QPoint, QRect, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFont, QKeyEvent, QPainter
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtGui import QPaintEvent

    from napari._qt.qt_main_window import _QtMainWindow


class TourAnchor(Enum):
    LEFT = 'left'
    RIGHT = 'right'
    ABOVE = 'above'
    BELOW = 'below'
    CENTER = 'center'


@dataclass(frozen=True)
class TourStep:
    target: Callable[[], QWidget | None]
    title: str
    body: str
    anchor: TourAnchor = TourAnchor.RIGHT
    skip: Callable[[], bool] = lambda: False
    # Returns whether it actually revealed something new, so callers can
    # wait for layout to settle before reading the target's geometry.
    ensure_visible: Callable[[], bool] = lambda: False


_TOOLTIP_WIDTH = 340


class _TourTooltip(QFrame):
    next_clicked = Signal()
    back_clicked = Signal()
    skip_clicked = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName('qt_viewer_tour_tooltip')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        self._title = QLabel()
        self._title.setWordWrap(True)
        title_font = QFont(self.font())
        title_font.setBold(True)
        self._title.setFont(title_font)
        layout.addWidget(self._title)

        self._body = QLabel()
        self._body.setWordWrap(True)
        self._body.setOpenExternalLinks(True)
        layout.addWidget(self._body)

        self._nav = QHBoxLayout()
        self._nav.setSpacing(8)
        self._counter = QLabel()
        self._nav.addWidget(self._counter)
        self._nav.addStretch()
        self._back = QPushButton()
        self._back.clicked.connect(self.back_clicked)
        self._nav.addWidget(self._back)
        self._next = QPushButton()
        self._next.clicked.connect(self.next_clicked)
        self._nav.addWidget(self._next)
        self._skip = QPushButton()
        self._skip.clicked.connect(self.skip_clicked)
        self._nav.addWidget(self._skip)
        layout.addLayout(self._nav)

    def _update_size(self) -> None:
        layout = self.layout()
        if layout is None:
            return
        # sizeHint() can be stale after a previous fixed-width pass;
        # invalidate() forces a fresh measurement of the current button text.
        self._nav.invalidate()
        margins = layout.contentsMargins()
        nav_width = (
            self._nav.sizeHint().width() + margins.left() + margins.right()
        )
        width = max(_TOOLTIP_WIDTH, nav_width)
        self.setFixedWidth(width)
        self.setFixedHeight(layout.heightForWidth(width))
        layout.activate()

    def set_content(
        self, title: str, body: str, step: int, total: int
    ) -> None:
        self._title.setText(title)
        self._body.setText(body)
        self._counter.setText(f'{step}/{total}')
        self._back.setText('Previous')
        self._skip.setText('(Esc) Skip')
        self._next.setText('Finish' if step == total else 'Next')
        self._back.setVisible(step > 1)
        self._skip.setVisible(step < total)
        self._update_size()

    def place(
        self, target_rect: QRect, anchor: TourAnchor, bounds: QRect
    ) -> None:
        gap = 12
        w, h = self.width(), self.height()
        if anchor == TourAnchor.LEFT:
            x, y = target_rect.left() - w - gap, target_rect.top()
        elif anchor == TourAnchor.ABOVE:
            x, y = (
                target_rect.center().x() - w // 2,
                target_rect.top() - h - gap,
            )
        elif anchor == TourAnchor.BELOW:
            x, y = (
                target_rect.center().x() - w // 2,
                target_rect.bottom() + gap,
            )
        elif anchor == TourAnchor.CENTER:
            x, y = (
                target_rect.center().x() - w // 2,
                target_rect.center().y() - h // 2,
            )
        else:
            x, y = target_rect.right() + gap, target_rect.top()
        x = max(bounds.left() + 8, min(x, bounds.right() - w - 8))
        y = max(bounds.top() + 8, min(y, bounds.bottom() - h - 8))
        self.move(x, y)


class _TourOverlay(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._spotlight: QRect | None = None

    def set_spotlight(self, rect: QRect | None) -> None:
        self._spotlight = rect
        self.update()

    def paintEvent(self, _event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        overlay = QColor(0, 0, 0, 150)
        if self._spotlight is None:
            painter.fillRect(self.rect(), overlay)
            return

        rect = self._spotlight.adjusted(-6, -6, 6, 6)
        painter.fillRect(0, 0, self.width(), rect.top(), overlay)
        painter.fillRect(
            0,
            rect.bottom() + 1,
            self.width(),
            self.height() - rect.bottom() - 1,
            overlay,
        )
        painter.fillRect(0, rect.top(), rect.left(), rect.height(), overlay)
        painter.fillRect(
            rect.right() + 1,
            rect.top(),
            self.width() - rect.right() - 1,
            rect.height(),
            overlay,
        )


class GuidedTour(QObject):
    finished = Signal()

    def __init__(self, steps: list[TourStep], parent_window: QWidget) -> None:
        super().__init__(parent_window)
        self._steps = steps
        self._window: QWidget | None = parent_window
        self._current = 0
        self._active = False
        self._overlay = _TourOverlay(parent_window)
        self._tooltip = _TourTooltip(parent_window)
        self._tooltip.next_clicked.connect(self._on_next)
        self._tooltip.back_clicked.connect(self._on_back)
        self._tooltip.skip_clicked.connect(self.close_tour)

    def start(self) -> None:
        if self._active or self._window is None:
            return
        self._active = True
        self._overlay.setGeometry(self._window.rect())
        self._overlay.show()
        self._overlay.raise_()
        self._tooltip.show()
        self._tooltip.raise_()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        start_index = self._seek(0, 1)
        if start_index is None:
            self.close_tour()
            return
        QTimer.singleShot(0, lambda: self._show_step(start_index))

    def close_tour(self) -> None:
        if not self._active or self._window is None:
            return
        self._active = False
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._tooltip.next_clicked.disconnect(self._on_next)
        self._tooltip.back_clicked.disconnect(self._on_back)
        self._tooltip.skip_clicked.disconnect(self.close_tour)
        self._overlay.hide()
        self._tooltip.hide()
        self._overlay.setParent(None)
        self._tooltip.setParent(None)
        self._overlay.deleteLater()
        self._tooltip.deleteLater()
        self.finished.emit()
        self.setParent(None)
        self._window = None
        self.deleteLater()

    def eventFilter(
        self, watched: QObject | None, event: QEvent | None
    ) -> bool:
        if event is None:
            return super().eventFilter(watched, event)
        if watched is self._window and event.type() == QEvent.Type.Resize:
            self._show_step(self._current)
        elif event.type() == QEvent.Type.KeyPress:
            # Escape always closes the tour, regardless of which widget has
            # focus: viewer widgets like the canvas unconditionally accept
            # key events, so Escape never reaches the tooltip's own
            # keyPressEvent unless the tooltip itself is focused. Other
            # nav shortcuts (N/P/Backspace/...) stay tooltip-focus-only
            # since, unlike Escape, they collide with real viewer shortcuts.
            key_event = event
            if (
                isinstance(key_event, QKeyEvent)
                and key_event.key() == Qt.Key.Key_Escape
            ):
                self.close_tour()
                return True
        return super().eventFilter(watched, event)

    @staticmethod
    def _is_available(step: TourStep) -> bool:
        return not step.skip() and step.target() is not None

    def _seek(self, index: int, direction: int) -> int | None:
        while 0 <= index < len(self._steps):
            if self._is_available(self._steps[index]):
                return index
            index += direction
        return None

    def _on_next(self) -> None:
        next_index = self._seek(self._current + 1, 1)
        if next_index is None:
            self.close_tour()
            return
        self._show_step(next_index)

    def _on_back(self) -> None:
        prev_index = self._seek(self._current - 1, -1)
        if prev_index is not None:
            self._show_step(prev_index)

    def _show_step(self, index: int) -> None:
        if self._window is None:
            return
        step = self._steps[index]
        if step.ensure_visible():
            # Revealing a hidden dock doesn't settle its final layout
            # geometry synchronously (QMainWindowLayout needs an event-loop
            # pass to reflow neighboring docks), so mapTo()/size() read
            # right now could be stale. Defer by one tick, same as the
            # initial _show_step call in start().
            QTimer.singleShot(0, lambda: self._show_step(index))
            return
        target = step.target()
        if target is None or not target.isVisible():
            return
        self._current = index
        visible = [
            i for i, s in enumerate(self._steps) if self._is_available(s)
        ]
        top_left = target.mapTo(self._window, QPoint(0, 0))
        rect = QRect(top_left, target.size())
        self._overlay.setGeometry(self._window.rect())
        self._overlay.set_spotlight(rect)
        self._tooltip.set_content(
            step.title,
            step.body,
            visible.index(index) + 1,
            len(visible),
        )
        self._tooltip.place(rect, step.anchor, self._window.rect())
        self._tooltip.raise_()


_BUILTIN_TOUR_TARGETS: dict[str, Callable[[_QtMainWindow], QWidget | None]] = {
    'canvas': lambda qt_window: qt_window._qt_viewer.canvas.native,
    'layer_list': lambda qt_window: qt_window._qt_viewer.dockLayerList,
    'layer_buttons': lambda qt_window: qt_window._qt_viewer.layerButtons,
    'layer_controls': lambda qt_window: qt_window._qt_viewer.dockLayerControls,
    'viewer_buttons': lambda qt_window: qt_window._qt_viewer.viewerButtons,
    'dims': lambda qt_window: qt_window._qt_viewer.dims,
    'status_bar': lambda qt_window: qt_window.statusBar(),
}


def resolve_tour_target(qt_window: _QtMainWindow, name: str) -> QWidget | None:
    """Resolve a tour step target by name.

    Checks napari's built-in viewer regions first (``'canvas'``,
    ``'layer_list'``, ``'layer_buttons'``, ``'layer_controls'``,
    ``'viewer_buttons'``, ``'dims'``, ``'status_bar'``), then falls back to
    ``window.dock_widgets``, so a plugin can target a widget it docked by
    the name it registered it under, without reaching into ``qt_viewer``.
    """
    builtin = _BUILTIN_TOUR_TARGETS.get(name)
    if builtin is not None:
        return builtin(qt_window)
    widget = qt_window._window.dock_widgets.get(name)
    if widget is None or isinstance(widget, QWidget):
        return widget
    return widget.native


def build_viewer_tour(
    window: _QtMainWindow,
    *,
    sample: tuple[str, str] | None = ('napari', 'balls_3d'),
) -> GuidedTour:
    """Build napari's built-in guided viewer tour.

    Parameters
    ----------
    window : _QtMainWindow
        The main window to build the tour for.
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
