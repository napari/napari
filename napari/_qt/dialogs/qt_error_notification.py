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

from ...utils.misc import StringEnum
from ..widgets.qt_eliding_label import MultilineElidedLabel

ActionSequence = Sequence[Tuple[str, Callable[[], None]]]


class NotificationSeverity(StringEnum):
    """Severity levels for the notification dialog.  Along with icons for each."""

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
    """Notification dialog frame, appears at the bottom right of the canvas.

    By default, only the first line of the notification is shown, and the text
    is elided.  Double-clicking on the text (or clicking the chevron icon) will
    expand to show the full notification.  The dialog will autmatically
    disappear in ``DISMISS_AFTER`` milliseconds, unless hovered or clicked.

    Parameters
    ----------
    message : str
        The message that will appear in the notification
    severity : str or NotificationSeverity, optional
        Severity level {'error', 'warning', 'info', 'none'}.  Will determine
        the icon associated with the message.
        by default NotificationSeverity.WARNING.
    source : str, optional
        A source string for the notifcation (intended to show the module and
        or package responsible for the notification), by default None
    actions : list of tuple, optional
        A sequence of 2-tuples, where each tuple is a string and a callable.
        Each tuple will be used to create button in the dialog, where the text
        on the button is determine by the first item in the tuple, and a
        callback function to call when the button is pressed is the second item
        in the tuple. by default ()
    """

    MAX_OPACITY = 0.9
    FADE_IN_RATE = 220
    FADE_OUT_RATE = 120
    DISMISS_AFTER = 4000
    MIN_WIDTH = 400

    def __init__(
        self,
        message: str,
        severity: Union[
            str, NotificationSeverity
        ] = NotificationSeverity.WARNING,
        source: Optional[str] = None,
        actions: ActionSequence = (),
    ):
        """[summary]"""
        super().__init__(None)
        # FIXME: this does not work with multiple viewers.
        # we need a way to detect the viewer in which the error occured.
        for wdg in QApplication.topLevelWidgets():
            if isinstance(wdg, QMainWindow):
                try:
                    # TODO: making the canvas the parent makes it easier to
                    # move/resize, but also means that the notification can get
                    # clipped on the left if the canvas is too small.
                    canvas = wdg.centralWidget().children()[1].canvas.native
                    self.setParent(canvas)
                    canvas.resized.connect(self.move_to_bottom_right)
                    break
                except Exception:
                    pass
        self.setupUi()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setup_buttons(actions)
        self.setMouseTracking(True)

        self.severity_icon.setText(NotificationSeverity(severity).as_icon())
        self.message.setText(message)
        if source:
            self.source_label.setText(f'Source: {source}')

        self.close_button.clicked.connect(self.close)
        self.expand_button.clicked.connect(self.toggle_expansion)

        self.timer = None
        self.opacity = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity)
        self.opacity_anim = QPropertyAnimation(self.opacity, b"opacity", self)
        self.geom_anim = QPropertyAnimation(self, b"geometry", self)
        self.move_to_bottom_right()

    def move_to_bottom_right(self, offset=(8, 8)):
        """Position widget at the bottom right edge of the parent."""
        if not self.parent():
            return
        sz = self.parent().size() - self.size() - QSize(*offset)
        self.move(QPoint(sz.width(), sz.height()))

    def slide_in(self):
        """Run animation that fades in the dialog with a slight slide up."""
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
        """Show the message with a fade and slight slide in from the bottom."""
        super().show()
        self.slide_in()
        self.timer = QTimer()
        self.timer.setInterval(self.DISMISS_AFTER)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.close)
        self.timer.start()

    def mouseMoveEvent(self, event):
        """On hover, stop the self-destruct timer"""
        self.timer.stop()

    def mouseDoubleClickEvent(self, event):
        """Expand the notification on double click."""
        self.toggle_expansion()

    def close(self):
        """Fade out then close."""
        self.opacity_anim.setDuration(self.FADE_OUT_RATE)
        self.opacity_anim.setStartValue(self.MAX_OPACITY)
        self.opacity_anim.setEndValue(0)
        self.opacity_anim.start()
        self.opacity_anim.finished.connect(super().close)

    def toggle_expansion(self):
        """Toggle the expanded state of the notification frame."""
        self.contract() if self.property('expanded') else self.expand()
        self.timer.stop()

    def expand(self):
        """Expanded the notification so that the full message is visible."""
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
        """Contract notification to a single elided line of the message."""
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
        """Set up the UI during initialization."""
        self.setWindowFlags(Qt.SubWindow)
        self.setMinimumWidth(self.MIN_WIDTH)
        self.setMaximumWidth(self.MIN_WIDTH)
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
        self.expand_button.setCursor(Qt.PointingHandCursor)
        self.expand_button.setMaximumWidth(20)
        self.expand_button.setFlat(True)

        self.row1.addWidget(self.expand_button, alignment=Qt.AlignTop)
        self.close_button = QPushButton(self.row1_widget)
        self.close_button.setObjectName("close_button")
        self.close_button.setCursor(Qt.PointingHandCursor)
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
        self.resize(self.MIN_WIDTH, 40)

    def setup_buttons(self, actions: ActionSequence = ()):
        """Add buttons to the dialog.


        Parameters
        ----------
        actions : tuple, optional
            A sequence of 2-tuples, where each tuple is a string and a
            callable. Each tuple will be used to create button in the dialog,
            where the text on the button is determine by the first item in the
            tuple, and a callback function to call when the button is pressed
            is the second item in the tuple. by default ()
        """
        if isinstance(actions, dict):
            actions = list(actions.items())
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
        """Return the size required to show the entire message."""
        return QSize(
            super().sizeHint().width(),
            self.row2_widget.height() + self.message.sizeHint().height(),
        )

    @classmethod
    def from_exception(cls, exception: BaseException) -> 'NapariNotification':
        """Create a NapariNotifcation dialog from an exception object."""
        # TODO: this method could be used to recognize various exception
        # subclasses and populate the dialog accordingly.
        msg = getattr(exception, 'message', str(exception))
        severity = getattr(exception, 'severity', 'WARNING')
        source = None
        actions = getattr(exception, 'actions', ())
        return cls(msg, severity, source, actions)
