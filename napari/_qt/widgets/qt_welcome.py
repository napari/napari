from __future__ import annotations

from qtpy.QtCore import Qt
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

from ...utils.translations import trans


class QtWelcomeLabel(QLabel):
    """Labels used for main message in welcome page."""


class QtShortcutLabel(QLabel):
    """Labels used for displaying shortcu information in welcome page."""


class QtWelcomeWidget(QWidget):
    """Welcome widget to display initial information and shortcuts to user."""

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

        self._update()

    def _update(self, event=None):
        icon = QColoredSVGIcon.from_resources('logo_silhouette')
        self._image.setPixmap(
            icon.colored(
                theme=SETTINGS.appearance.theme, theme_key='background'
            ).pixmap(300, 300)
        )

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
        # QtCanvasOverlay > QWidget > QtViewer
        self.parent().parent().parent().dropEvent(event)


class QtCanvasOverlay(QStackedWidget):
    """
    Stacked widget providing switching between the canvas and the welcome page.
    """

    def __init__(self, parent, canvas):
        super().__init__(parent)

        # Widget setup
        self.addWidget(canvas.native)
        self.addWidget(QtWelcomeWidget(self))
        self.setCurrentIndex(0)

    def set_welcome_visible(self, visible=True):
        """Show welcome screen widget on stack."""
        self.setCurrentIndex(int(visible))
