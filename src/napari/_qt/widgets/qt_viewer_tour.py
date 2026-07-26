from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, QObject, QPoint, QRect, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFont, QKeyEvent, QPainter, QPen
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari.utils.theme import get_theme
from napari.utils.translations import trans

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtGui import QMouseEvent, QPaintEvent

    from napari._qt.qt_main_window import _QtMainWindow


@dataclass(frozen=True)
class TourStep:
    target: Callable[[], QWidget | None]
    title: str
    body: str
    anchor: str = 'right'


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
        self._back.clicked.connect(self.back_clicked)
        nav.addWidget(self._back)
        self._next = QPushButton()
        self._next.clicked.connect(self.next_clicked)
        nav.addWidget(self._next)
        self._skip = QPushButton()
        self._skip.clicked.connect(self.skip_clicked)
        nav.addWidget(self._skip)
        layout.addLayout(nav)

    def apply_theme(self, theme_name: str) -> None:
        theme = get_theme(theme_name).to_rgb_dict()
        accent = theme['primary']
        bg = theme['background']
        fg = theme['text']
        border = theme['foreground']
        btn_bg = theme['secondary']
        self.setStyleSheet(
            f"""
            QFrame#qt_viewer_tour_tooltip {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 0;
            }}
            QFrame#qt_viewer_tour_tooltip QLabel {{
                color: {fg};
                background: transparent;
                border: none;
            }}
            QFrame#qt_viewer_tour_tooltip QPushButton {{
                background: {btn_bg};
                color: {fg};
                border: none;
                border-radius: 0;
                padding: 5px 12px;
            }}
            QFrame#qt_viewer_tour_tooltip QPushButton:hover {{
                background: {accent};
            }}
            """
        )
        self._title.setStyleSheet('color: #ffffff;')

    def _update_size(self) -> None:
        window_width = (
            self.parentWidget().width() if self.parentWidget() else 900
        )
        width = min(_TOOLTIP_MAX_WIDTH, max(280, window_width // 3))
        self.setFixedWidth(width)
        layout = self.layout()
        if layout is None:
            return
        margins = layout.contentsMargins()
        content_width = width - margins.left() - margins.right()
        self._title.setFixedWidth(content_width)
        self._body.setFixedWidth(content_width)
        self._title.setFixedHeight(self._title.heightForWidth(content_width))
        self._body.setFixedHeight(self._body.heightForWidth(content_width))
        layout.activate()
        self.setFixedHeight(layout.sizeHint().height())

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

    def place(self, target_rect: QRect, anchor: str, bounds: QRect) -> None:
        gap = 12
        w, h = self.width(), self.height()
        if anchor == 'left':
            x, y = target_rect.left() - w - gap, target_rect.top()
        elif anchor == 'above':
            x, y = (
                target_rect.center().x() - w // 2,
                target_rect.top() - h - gap,
            )
        elif anchor == 'below':
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
    def __init__(self, parent: QWidget, *, accent_color: str) -> None:
        super().__init__(parent)
        self._spotlight: QRect | None = None
        self._accent_color = accent_color

    def set_spotlight(self, rect: QRect | None) -> None:
        self._spotlight = rect
        self.update()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:  # type: ignore[override]
        if event is not None:
            event.accept()

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
        pen = QPen(QColor(self._accent_color), 2)
        painter.setPen(pen)
        painter.drawRect(rect)


class GuidedTour(QObject):
    finished = Signal()

    def __init__(
        self, steps: list[TourStep], parent_window: QWidget, *, theme_name: str
    ) -> None:
        super().__init__(parent_window)
        self._steps = steps
        self._window: QWidget | None = parent_window
        self._current = 0
        self._active = False
        theme = get_theme(theme_name).to_rgb_dict()
        self._overlay = _TourOverlay(
            parent_window, accent_color=theme['primary']
        )
        self._tooltip = _TourTooltip(parent_window)
        self._tooltip.apply_theme(theme_name)
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
        self._window.installEventFilter(self)
        QTimer.singleShot(0, lambda: self._show_step(0))

    def close_tour(self) -> None:
        if not self._active or self._window is None:
            return
        self._active = False
        window = self._window
        window.removeEventFilter(self)
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
    ) -> bool:  # type: ignore[override]
        if watched is not self._window or event is None:
            return super().eventFilter(watched, event)
        if event.type() == QEvent.Type.Resize:
            self._show_step(self._current)
        elif event.type() == QEvent.Type.KeyPress:
            key_event = event
            if not isinstance(key_event, QKeyEvent):
                return super().eventFilter(watched, event)
            if key_event.key() in (
                Qt.Key.Key_N,
                Qt.Key.Key_Right,
                Qt.Key.Key_Enter,
                Qt.Key.Key_Return,
            ):
                self._on_next()
                return True
            if key_event.key() in (
                Qt.Key.Key_P,
                Qt.Key.Key_Left,
                Qt.Key.Key_Backspace,
            ):
                self._on_back()
                return True
            if key_event.key() == Qt.Key.Key_Escape:
                self.close_tour()
                return True
        return super().eventFilter(watched, event)

    def _on_next(self) -> None:
        if self._current >= len(self._steps) - 1:
            self.close_tour()
            return
        self._show_step(self._current + 1)

    def _on_back(self) -> None:
        if self._current > 0:
            self._show_step(self._current - 1)

    def _show_step(self, index: int) -> None:
        if self._window is None:
            return
        self._current = index
        step = self._steps[index]
        target = step.target()
        if target is None or not target.isVisible():
            return
        top_left = target.mapTo(self._window, QPoint(0, 0))
        rect = QRect(top_left, target.size())
        self._overlay.setGeometry(self._window.rect())
        self._overlay.set_spotlight(rect)
        self._tooltip.set_content(
            step.title, step.body, index + 1, len(self._steps)
        )
        self._tooltip.place(rect, step.anchor, self._window.rect())
        self._tooltip.raise_()


def build_viewer_tour(window: _QtMainWindow) -> GuidedTour:
    qt_viewer = window._qt_viewer
    status_bar = window.statusBar()

    return GuidedTour(
        [
            TourStep(
                target=lambda: qt_viewer.canvas.native,
                title=trans._('Welcome to the viewer'),
                body=trans._(
                    'This PoC loads the built-in Balls (3D) sample so the tour has something to show. '
                    'Drag to pan, scroll to zoom, and reopen this tour from Help any time.'
                ),
                anchor='below',
            ),
            TourStep(
                target=lambda: qt_viewer.dockLayerList,
                title=trans._('Layer list'),
                body=trans._(
                    'Layers live here. Select one to edit it, rename it inline, change visibility, or reorder by dragging.'
                ),
            ),
            TourStep(
                target=lambda: qt_viewer.layerButtons,
                title=trans._('Layer buttons'),
                body=trans._(
                    'These create or delete layers. They are the quickest way to add points, shapes, or labels on top of your data.'
                ),
            ),
            TourStep(
                target=lambda: qt_viewer.dockLayerControls,
                title=trans._('Layer controls'),
                body=trans._(
                    'The active layer decides what appears here. Different layer types expose different controls for appearance and editing.'
                ),
            ),
            TourStep(
                target=lambda: qt_viewer.viewerButtons,
                title=trans._('Viewer buttons'),
                body=trans._(
                    'Use these for grid mode, 2D/3D display, axis order, and resetting the camera with the home button.'
                ),
                anchor='above',
            ),
            TourStep(
                target=lambda: qt_viewer.dims,
                title=trans._('Dimension sliders'),
                body=trans._(
                    'Extra dimensions show up here. Move through slices, or press play on a slider to animate along that axis.'
                ),
                anchor='above',
            ),
            TourStep(
                target=lambda: status_bar,
                title=trans._('Status bar'),
                body=trans._(
                    'The status bar reports cursor position, values under the mouse, and small context-sensitive hints while you interact.'
                ),
                anchor='above',
            ),
        ],
        window,
        theme_name=qt_viewer.viewer.theme,
    )
