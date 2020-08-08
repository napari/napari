from enum import auto
from typing import Callable, Optional, Sequence, Tuple, Union

from qtpy.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    QTimer,
)
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..utils.misc import StringEnum
from .qt_eliding_label import MultilineElidedLabel


ActionSequence = Sequence[Tuple[str, Callable[[], None]]]


class NotificationSeverity(StringEnum):
    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    NONE = auto()

    def as_icon(self):
        return {
            self.ERROR: "ⓧ",
            self.WARNING: "⚠️",
            self.INFO: "ⓘ",
            self.NONE: "",
        }[self]


class NapariNotification(QDialog):
    MAX_OPACITY = 0.9
    FADE_IN_RATE = 220
    FADE_OUT_RATE = 120
    DISMISS_AFTER = 5000
    MIN_WIDTH = 400

    def __init__(
        self,
        message: str,
        severity: Union[
            str, NotificationSeverity
        ] = NotificationSeverity.ERROR,
        source: Optional[str] = None,
        actions: ActionSequence = (),
    ):
        # FIXME: this works with command line, but not with IPython...
        # and may not work well with multiple viewers.
        parent = None
        for wdg in QApplication.topLevelWidgets():
            if isinstance(wdg, QMainWindow):
                parent = wdg
                break
        super().__init__(parent)
        if self.parent():
            self.parent().resized.connect(self.move_to_bottom_right)
        self.setWindowFlags(
            Qt.SubWindow | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setupUi()
        self.setup_buttons(actions)
        self.setMouseTracking(True)

        self.severity_icon.setText(NotificationSeverity(severity).as_icon())
        self.message.setText(message)
        if source:
            self.source_label.setText(f'source: {source}')
        self.close_button.clicked.connect(self.close)
        self.expand_button.clicked.connect(self.toggle_expansion)

        self.opacity = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity)
        self.opacity_anim = QPropertyAnimation(self.opacity, b"opacity", self)
        self.geom_anim = QPropertyAnimation(self, b"geometry", self)
        self.move_to_bottom_right()

    def move_to_bottom_right(self, size=None, from_edge=(22, 50)):
        # move to the bottom right of parent
        sz = (size or self.parent().size()) - self.size() - QSize(*from_edge)
        self.move(QPoint(sz.width(), sz.height()))

    def slide_in(self):
        # slide in
        geom = self.geometry()
        self.geom_anim.setDuration(self.FADE_IN_RATE)
        self.geom_anim.setStartValue(geom.translated(0, 20))
        self.geom_anim.setEndValue(geom)
        self.geom_anim.setEasingCurve(QEasingCurve.OutQuad)
        # fade in
        self.opacity_anim.setDuration(self.FADE_IN_RATE)
        self.opacity_anim.setStartValue(0)
        self.opacity_anim.setEndValue(self.MAX_OPACITY)
        self.geom_anim.start()
        self.opacity_anim.start()

    def show(self):
        """Show the message with a fade and slight slide in from the bottom.
        """
        super().show()
        self.slide_in()
        self.timer = QTimer()
        self.timer.setInterval(self.DISMISS_AFTER)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.close)
        self.timer.start()

    def mouseMoveEvent(self, event):
        self.close_button.show()
        self.timer.stop()

    def mouseDoubleClickEvent(self, event):
        self.toggle_expansion()

    def close(self):
        """Fade out then close."""
        self.opacity_anim.setDuration(self.FADE_OUT_RATE)
        self.opacity_anim.setStartValue(self.MAX_OPACITY)
        self.opacity_anim.setEndValue(0)
        self.opacity_anim.start()
        self.opacity_anim.finished.connect(super().close)

    def toggle_expansion(self):
        self.contract() if self.property('expanded') else self.expand()
        self.timer.stop()

    def expand(self):
        curr = self.geometry()
        self.geom_anim.setDuration(100)
        self.geom_anim.setStartValue(curr)
        new_height = self.sizeHint().height()
        delta = new_height - curr.height()
        self.geom_anim.setEndValue(
            QRect(curr.x(), curr.y() - delta, curr.width(), new_height)
        )
        self.geom_anim.setEasingCurve(QEasingCurve.OutQuad)
        self.geom_anim.start()
        self.setProperty('expanded', True)
        self.style().unpolish(self.expand_button)
        self.style().polish(self.expand_button)

    def contract(self):
        geom = self.geometry()
        self.geom_anim.setDuration(100)
        self.geom_anim.setStartValue(geom)
        dlt = geom.height() - self.minimumHeight()
        self.geom_anim.setEndValue(
            QRect(geom.x(), geom.y() + dlt, geom.width(), geom.height() - dlt)
        )
        self.geom_anim.setEasingCurve(QEasingCurve.OutQuad)
        self.geom_anim.start()
        self.setProperty('expanded', False)
        self.style().unpolish(self.expand_button)
        self.style().polish(self.expand_button)

    def setupUi(self):
        self.setMinimumWidth(self.MIN_WIDTH)
        self.setMinimumHeight(40)
        self.setSizeGripEnabled(False)
        self.setModal(False)
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout.setSpacing(0)

        self.row1_widget = QWidget(self)
        self.row1 = QHBoxLayout(self.row1_widget)
        self.row1.setContentsMargins(12, 12, 12, 8)
        self.row1.setSpacing(4)
        self.severity_icon = QLabel(self.row1_widget)
        self.severity_icon.setObjectName("severity_icon")
        self.severity_icon.setMinimumWidth(30)
        self.severity_icon.setMaximumWidth(30)
        self.row1.addWidget(self.severity_icon, alignment=Qt.AlignTop)
        self.message = MultilineElidedLabel(self.row1_widget)
        self.message.setMinimumWidth(self.MIN_WIDTH - 200)
        self.message.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.row1.addWidget(self.message, alignment=Qt.AlignTop)
        self.expand_button = QPushButton(self.row1_widget)
        self.expand_button.setObjectName("expand_button")
        self.expand_button.setMaximumWidth(20)
        self.expand_button.setFlat(True)

        self.row1.addWidget(self.expand_button, alignment=Qt.AlignTop)
        self.close_button = QPushButton(self.row1_widget)
        self.close_button.setObjectName("close_button")
        self.close_button.setMaximumWidth(20)
        self.close_button.setFlat(True)

        self.row1.addWidget(self.close_button, alignment=Qt.AlignTop)
        self.verticalLayout.addWidget(self.row1_widget, 1)
        self.row2_widget = QWidget(self)
        self.row2_widget.hide()
        self.row2 = QHBoxLayout(self.row2_widget)
        self.source_label = QLabel(self.row2_widget)
        self.source_label.setObjectName("source_label")
        self.row2.addWidget(self.source_label, alignment=Qt.AlignBottom)
        self.row2.addStretch()
        self.row2.setContentsMargins(12, 2, 16, 12)
        self.row2_widget.setMaximumHeight(34)
        self.row2_widget.setStyleSheet(
            'QPushButton{'
            'padding: 4px 12px 4px 12px; '
            'font-size: 11px;'
            'min-height: 18px; border-radius: 0;}'
        )
        self.verticalLayout.addWidget(self.row2_widget, 0)
        self.setProperty('expanded', False)

    def setup_buttons(self, actions=()):
        for text, callback in actions:
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            btn.clicked.connect(self.close)
            self.row2.addWidget(btn)
        if actions:
            self.row2_widget.show()
            self.setMinimumHeight(
                self.row2_widget.maximumHeight() + self.minimumHeight()
            )

    def sizeHint(self):
        return QSize(
            super().sizeHint().width(),
            self.row2_widget.height() + self.message.sizeHint().height(),
        )
