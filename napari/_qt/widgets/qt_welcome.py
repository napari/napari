from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QKeySequence, QPainter
from qtpy.QtWidgets import (
    QFormLayout,
    QLabel,
    QStackedWidget,
    QStyle,
    QStyleOption,
    QVBoxLayout,
    QWidget,
)

from ...utils.settings import SETTINGS
from ...utils.translations import trans
from ..qt_resources import QColoredSVGIcon


class QtWelcomeLabel(QLabel):
    """Labels used for main message in welcome page."""


class QtShortcutLabel(QLabel):
    """Labels used for displaying shortcu information in welcome page."""


class QtWelcomeWidget(QWidget):
    """Welcome widget to display initial information and shortcuts to user."""

    sig_dropped = Signal(object)

    def __init__(self, parent):
        super().__init__(parent)

        # Create colored icon using theme
        icon = QColoredSVGIcon.from_resources('logo_silhouette')
        self._image = QLabel()
        self._image.setPixmap(
            icon.colored(theme=SETTINGS.appearance.theme).pixmap(300, 300)
        )
        self._label = QtWelcomeLabel(
            trans._(
                "Drag image(s) here to open\nor\nUse the menu shortcuts below:"
            )
        )

        # Widget setup
        self.setAutoFillBackground(True)
        self.setAcceptDrops(True)
        self._image.setAlignment(Qt.AlignCenter)
        self._label.setAlignment(Qt.AlignCenter)

        # Layout
        text_layout = QVBoxLayout()
        text_layout.addWidget(self._label)

        # TODO: Use action manager for shortcut query and handling
        shortcut_layout = QFormLayout()
        sc = QKeySequence('Ctrl+O').toString(QKeySequence.NativeText)
        shortcut_layout.addRow(
            QtShortcutLabel(sc),
            QtShortcutLabel(trans._("open image(s)")),
        )
        sc = QKeySequence('Ctrl+Alt+/').toString(QKeySequence.NativeText)
        shortcut_layout.addRow(
            QtShortcutLabel(sc),
            QtShortcutLabel(trans._("show all key bindings")),
        )
        shortcut_layout.setSpacing(0)

        layout = QVBoxLayout()
        layout.addStretch()
        layout.setSpacing(30)
        layout.addWidget(self._image)
        layout.addLayout(text_layout)
        layout.addLayout(shortcut_layout)
        layout.addStretch()

        self.setLayout(layout)

    def paintEvent(self, event):
        """Override Qt method.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        option = QStyleOption()
        option.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, option, p, self)

    def _update_property(self, prop, value):
        """Update properties of widget to update style.

        Parameters
        ----------
        prop : str
            Property name to update.
        value : bool
            Property value to update.
        """
        self.setProperty(prop, value)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event):
        """Override Qt method.

        Provide style updates on event.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self._update_property("drag", True)
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Override Qt method.

        Provide style updates on event.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self._update_property("drag", False)

    def dropEvent(self, event):
        """Override Qt method.

        Provide style updates on event and emit the drop event.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self._update_property("drag", False)
        self.sig_dropped.emit(event)


class QtCanvasOverlay(QWidget):
    """
    Stacked widget providing switching between the canvas and the welcome page.
    """

    sig_dropped = Signal(object)

    def __init__(self, parent, canvas):
        super().__init__(parent)

        # Widgets
        self._stack = QStackedWidget(self)
        self._canvas = canvas
        self._overlay = QtWelcomeWidget(self)

        # Widget setup
        self._stack.addWidget(self._canvas.native)
        self._stack.addWidget(self._overlay)
        self._stack.setCurrentIndex(0)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self._stack)
        self.setLayout(layout)

        # Signals
        self._overlay.sig_dropped.connect(self.sig_dropped)

    def show_welcome(self):
        """Show welcome screen widget on stack."""
        self._stack.setCurrentIndex(0)

    def hide_welcome(self):
        """Hide welcome screen widget and display Canvas."""
        self._stack.setCurrentIndex(1)
