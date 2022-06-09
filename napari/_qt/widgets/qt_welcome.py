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

from ...utils.action_manager import action_manager
from ...utils.interactions import Shortcut
from ...utils.translations import trans


class QtWelcomeLabel(QLabel):
    """Labels used for main message in welcome page."""


class QtShortcutLabel(QLabel):
    """Labels used for displaying shortcu information in welcome page."""


class QtWelcomeWidget(QWidget):
    """Welcome widget to display initial information and shortcuts to user."""

    sig_dropped = Signal("QEvent")

    def __init__(self, parent):
        super().__init__(parent)

        # Create colored icon using theme
        self._image = QLabel()
        self._image.setObjectName("logo_silhouette")
        self._image.setMinimumSize(300, 300)
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
        sc = QKeySequence('Ctrl+O', QKeySequence.PortableText).toString(
            QKeySequence.NativeText
        )
        shortcut_layout.addRow(
            QtShortcutLabel(sc),
            QtShortcutLabel(trans._("open image(s)")),
        )
        self._shortcut_label = QtShortcutLabel("")
        shortcut_layout.addRow(
            self._shortcut_label,
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
        self._show_shortcuts_updated()
        action_manager.events.shorcut_changed.connect(
            self._show_shortcuts_updated
        )

    def _show_shortcuts_updated(self):
        shortcut_list = list(
            action_manager._shortcuts["napari:show_shortcuts"]
        )
        if not shortcut_list:
            return
        self._shortcut_label.setText(Shortcut(shortcut_list[0]).platform)

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


class QtWidgetOverlay(QStackedWidget):
    """
    Stacked widget providing switching between the widget and a welcome page.
    """

    sig_dropped = Signal("QEvent")
    resized = Signal()
    leave = Signal()
    enter = Signal()

    def __init__(self, parent, widget):
        super().__init__(parent)

        self._overlay = QtWelcomeWidget(self)

        # Widget setup
        self.addWidget(widget)
        self.addWidget(self._overlay)
        self.setCurrentIndex(0)

        # Signals
        self._overlay.sig_dropped.connect(self.sig_dropped)

    def set_welcome_visible(self, visible=True):
        """Show welcome screen widget on stack."""
        self.setCurrentIndex(int(visible))

    def resizeEvent(self, event):
        """Emit our own event when canvas was resized."""
        self.resized.emit()
        return super().resizeEvent(event)

    def enterEvent(self, event):
        """Emit our own event when mouse enters the canvas."""
        self.enter.emit()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Emit our own event when mouse leaves the canvas."""
        self.leave.emit()
        super().leaveEvent(event)
