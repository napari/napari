from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, QObject, QPoint, QRect, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFont, QKeyEvent, QPainter
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari.utils.translations import trans

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtGui import QPaintEvent

    from napari._qt.qt_main_window import _QtMainWindow


class TourAnchor(Enum):
    LEFT = 'left'
    RIGHT = 'right'
    ABOVE = 'above'
    BELOW = 'below'


@dataclass(frozen=True)
class TourStep:
    target: Callable[[], QWidget | None]
    title: str
    body: str
    anchor: TourAnchor = TourAnchor.RIGHT
    skip: Callable[[], bool] = lambda: False


_TOOLTIP_MAX_WIDTH = 420


class _TourTooltip(QFrame):
    next_clicked = Signal()
    back_clicked = Signal()
    skip_clicked = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName('qt_viewer_tour_tooltip')
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

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
        layout.addWidget(self._body)

        nav = QHBoxLayout()
        nav.setSpacing(8)
        self._counter = QLabel()
        nav.addWidget(self._counter)
        nav.addStretch()
        self._back = QPushButton()
        self._back.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._back.clicked.connect(self.back_clicked)
        nav.addWidget(self._back)
        self._next = QPushButton()
        self._next.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._next.clicked.connect(self.next_clicked)
        nav.addWidget(self._next)
        self._skip = QPushButton()
        self._skip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._skip.clicked.connect(self.skip_clicked)
        nav.addWidget(self._skip)
        layout.addLayout(nav)

    def _update_size(self) -> None:
        layout = self.layout()
        if layout is None:
            return
        parent = self.parentWidget()
        window_width = parent.width() if parent is not None else 900
        width = min(_TOOLTIP_MAX_WIDTH, max(280, window_width // 3))
        self.setFixedWidth(width)
        self.setFixedHeight(layout.heightForWidth(width))
        layout.activate()

    def set_content(
        self, title: str, body: str, step: int, total: int
    ) -> None:
        self._title.setText(title)
        self._body.setText(body)
        self._counter.setText(f'{step}/{total}')
        self._back.setText(trans._('(P)revious'))
        self._skip.setText(trans._('(Esc) Skip'))
        self._next.setText(
            trans._('(N) Finish') if step == total else trans._('(N)ext')
        )
        self._back.setVisible(step > 1)
        self._skip.setVisible(step < total)
        self._update_size()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is None:
            return
        if event.key() == Qt.Key.Key_Escape:
            self.skip_clicked.emit()
            return
        if event.key() in (
            Qt.Key.Key_N,
            Qt.Key.Key_Right,
            Qt.Key.Key_Enter,
            Qt.Key.Key_Return,
        ):
            self.next_clicked.emit()
            return
        if event.key() in (
            Qt.Key.Key_P,
            Qt.Key.Key_Left,
            Qt.Key.Key_Backspace,
        ):
            self.back_clicked.emit()
            return
        super().keyPressEvent(event)

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
        self._tooltip.setFocus()
        self._window.installEventFilter(self)
        start_index = self._seek(0, 1)
        if start_index is None:
            self.close_tour()
            return
        QTimer.singleShot(0, lambda: self._show_step(start_index))

    def close_tour(self) -> None:
        if not self._active or self._window is None:
            return
        self._active = False
        self._window.removeEventFilter(self)
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


def resolve_tour_target(
    qt_window: _QtMainWindow, name: str
) -> QWidget | None:
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


def build_viewer_tour(window: _QtMainWindow) -> GuidedTour:
    viewer = window._qt_viewer.viewer

    def target(name: str) -> Callable[[], QWidget | None]:
        return partial(resolve_tour_target, window, name)

    return GuidedTour(
        [
            TourStep(
                target=target('canvas'),
                title=trans._('Welcome to the viewer'),
                body=trans._(
                    'The viewer canvas shows your layers. Drag to pan, scroll to zoom, '
                    'and reopen this tour from Help any time.'
                ),
                anchor=TourAnchor.BELOW,
            ),
            TourStep(
                target=target('layer_list'),
                title=trans._('Layer list'),
                body=trans._(
                    'Layers live here. Select one to edit it, rename it inline, change visibility, or reorder by dragging.'
                ),
            ),
            TourStep(
                target=target('layer_buttons'),
                title=trans._('Layer buttons'),
                body=trans._(
                    'These create or delete layers. They are the quickest way to add points, shapes, or labels on top of your data.'
                ),
            ),
            TourStep(
                target=target('layer_controls'),
                title=trans._('Layer controls'),
                body=trans._(
                    'The active layer decides what appears here. Different layer types expose different controls for appearance and editing.'
                ),
            ),
            TourStep(
                target=target('viewer_buttons'),
                title=trans._('Viewer buttons'),
                body=trans._(
                    'Use these for grid mode, 2D/3D display, axis order, and resetting the camera with the home button.'
                ),
                anchor=TourAnchor.ABOVE,
            ),
            TourStep(
                target=target('dims'),
                title=trans._('Dimension sliders'),
                body=trans._(
                    'Extra dimensions show up here. Move through slices, or press play on a slider to animate along that axis.'
                ),
                anchor=TourAnchor.ABOVE,
                skip=lambda: viewer.dims.ndim <= viewer.dims.ndisplay,
            ),
            TourStep(
                target=target('status_bar'),
                title=trans._('Status bar'),
                body=trans._(
                    'The status bar reports cursor position, values under the mouse, and small context-sensitive hints while you interact.'
                ),
                anchor=TourAnchor.ABOVE,
            ),
        ],
        window,
    )
