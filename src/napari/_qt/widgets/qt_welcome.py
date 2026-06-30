from __future__ import annotations

from random import choice
from typing import TYPE_CHECKING

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QPainter
from qtpy.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QLabel,
    QStyle,
    QStyleOption,
    QVBoxLayout,
    QWidget,
)

from napari import __version__
from napari.utils.action_manager import action_manager
from napari.utils.tips import (
    NAPARI_TIPS,
    _get_command_shortcut_and_description,
    format_tip,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


WELCOME_SHORTCUTS = (
    'napari.window.file._image_from_clipboard',
    'napari.window.file.open_files_dialog',
    'napari.window.view.toggle_command_palette',
    'napari:show_shortcuts',
)


class QtWelcomeLabel(QLabel):
    """Labels used for main message in welcome page."""


class QtShortcutLabel(QLabel):
    """Labels used for displaying shortcut information in welcome page."""


class QtVersionLabel(QLabel):
    """Label used for displaying version information."""


class QtWelcomeWidget(QWidget):
    """Welcome widget to display initial information, shortcuts, and tips."""

    urls_drag_entered = Signal()
    urls_dropped = Signal(object)

    def __init__(
        self, parent: QWidget | None, tips: Sequence[str] | None = None
    ) -> None:
        super().__init__(parent)

        # Widget setup
        self.setAutoFillBackground(True)
        self.setAcceptDrops(True)
        self.setProperty('drag', False)

        # Create colored icon using theme
        self._image = QLabel()
        self._image.setObjectName('logo_silhouette')
        self._image.setMinimumSize(300, 300)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # setup text elements
        self._version_label = QtVersionLabel(f'napari {__version__}')
        self._version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._version_label.setWordWrap(True)

        self._label = QtWelcomeLabel(
            'Drag file(s) here to open, or use shortcuts below:'
        )
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)

        self._tip_label = QtShortcutLabel('')
        self._tip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tip_label.setWordWrap(True)
        self._tip_label.setContentsMargins(64, 0, 64, 0)

        self.set_tips(tips)

        # setup the shortcuts "table"
        shortcut_layout = QFormLayout()
        self._shortcut_rows: list[
            tuple[str, QtShortcutLabel, QtShortcutLabel]
        ] = []
        for command_id in WELCOME_SHORTCUTS:
            shortcut_label = QtShortcutLabel('')
            description_label = QtShortcutLabel('')
            self._shortcut_rows.append(
                (command_id, shortcut_label, description_label)
            )
            shortcut_layout.addRow(shortcut_label, description_label)
        shortcut_layout.setVerticalSpacing(0)
        shortcut_layout.setHorizontalSpacing(10)
        # override default form styles for consistent appearance across platforms
        shortcut_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        shortcut_layout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        shortcut_layout.setFormAlignment(Qt.AlignHCenter | Qt.AlignTop)
        shortcut_layout.setLabelAlignment(Qt.AlignRight)

        self.refresh_shortcuts()

        #  Widget layout of logo and text elements
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # setup logo layout
        logo_container = QWidget()
        logo_container.setStyleSheet('background: transparent;')
        logo_layout = QVBoxLayout(logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.addStretch(1)
        logo_layout.addWidget(self._image, alignment=Qt.AlignCenter)
        logo_layout.addStretch(4)  # Push logo up towards the top
        layout.addWidget(logo_container, 0, 0)

        # setup text blocks layout
        text_container = QWidget()
        text_container.setStyleSheet('background: transparent;')
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        text_layout.addStretch(10)  # push text down
        text_layout.addWidget(self._version_label)
        text_layout.addStretch(1)
        text_layout.addWidget(self._label)
        text_layout.addLayout(shortcut_layout)
        text_layout.addStretch(1)
        text_layout.addWidget(self._tip_label)
        text_layout.addStretch(1)  # Leave some space at bottom

        # stack text container over the logo container
        layout.addWidget(text_container, 0, 0)

        self.setLayout(layout)
        # The welcome screen is a manual overlay on top of the canvas
        # Need to keep its geometry synced explicitly
        parent_resized = getattr(self.parentWidget(), 'resized', None)
        if parent_resized is not None:
            parent_resized.connect(self._sync_to_parent)
        self._sync_to_parent()
        self.hide()
        action_manager.events.shortcut_changed.connect(self.refresh)

    def minimumSizeHint(self) -> QSize:
        """Overwrite minimum size to allow creating small viewer instance."""
        return QSize(100, 100)

    def refresh(self, _event=None) -> None:
        self.refresh_shortcuts()
        self._update_tip_label()

    def refresh_shortcuts(self, _event=None) -> None:
        """Update the shortcut table using the current runtime bindings."""
        for (
            command_id,
            shortcut_label,
            description_label,
        ) in self._shortcut_rows:
            shortcut, description = _get_command_shortcut_and_description(
                command_id
            )
            shortcut_label.setText(shortcut or '')
            description_label.setText(description or '')

    def set_tips(self, tips: Sequence[str] | None) -> None:
        """Replace the tip pool and show a random one."""
        self._tips = (
            NAPARI_TIPS if tips is None else (tips or ("You're awesome!",))
        )
        self.show_random_tip()

    def show_random_tip(self) -> None:
        self._current_tip = choice(self._tips)
        self._update_tip_label()

    def set_welcome_visible(self, visible: bool = True) -> None:
        if visible:
            self.refresh_shortcuts()
            self.show_random_tip()
            self._sync_to_parent()
            self.show()
            self.raise_()
        else:
            self.hide()

    def _update_tip_label(self) -> None:
        """Render the current tip after expanding any shortcut placeholders."""
        self._tip_label.setText(
            f'Did you know?\n{format_tip(self._current_tip)}'
        )

    def paintEvent(self, event):
        """Override Qt method.

        Paint using the stylesheet.
        This is needed for the drag-and-drop highlighting.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        option = QStyleOption()
        option.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, option, p, self)

    def _set_drag_highlight(self, value: bool) -> None:
        """Update the QSS state used to highlight drag-and-drop.

        The welcome widget QSS uses the dynamic ``drag`` property
        e.g. ``QtWelcomeWidget[drag=true]`` to switch backgrounds
        while a file is dragged over the canvas. Here we repolish
        the widget after updating it, so the dynamic-property
        selectors above take effect immediately when the drag state changes
        """
        self.setProperty('drag', value)
        self.style().unpolish(self)
        self.style().polish(self)

    def _sync_to_parent(self, *_args) -> None:
        """Resize this overlay to cover the full parent canvas.

        The welcome widget is parented directly to the VisPy
        canvas so it can sit above the rendered scene, but
        it is not managed by a layout.
        This helper ensures its geometry is matched whenever
        the canvas resizes.
        """
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())

    def dragEnterEvent(self, event):
        """Override Qt method.

        Update the drag highlight and forward valid URL drags upstream.

        Parameters
        ----------
        event : qtpy.QtCore.QDragEnterEvent
            Event from the Qt context.
        """
        if not event.mimeData().hasUrls():
            self._set_drag_highlight(False)
            event.ignore()
            return

        self._set_drag_highlight(True)
        self.urls_drag_entered.emit()
        event.accept()

    def dragLeaveEvent(self, event):
        """Override Qt method.

        Clear the drag highlight when a drag leaves the overlay.

        Parameters
        ----------
        event : qtpy.QtCore.QDragLeaveEvent
            Event from the Qt context.
        """
        self._set_drag_highlight(False)

    def dropEvent(self, event):
        """Override Qt method.

        Clear the drag highlight and forward the drop to the viewer.

        Parameters
        ----------
        event : qtpy.QtCore.QDropEvent
            Event from the Qt context.
        """
        self._set_drag_highlight(False)
        self.urls_dropped.emit(event)
