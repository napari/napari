from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from random import choice

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QPainter
from qtpy.QtWidgets import (
    QFormLayout,
    QLabel,
    QStackedWidget,
    QStyle,
    QStyleOption,
    QVBoxLayout,
    QWidget,
)

from napari import __version__
from napari._app_model import get_app_model
from napari.settings import get_settings
from napari.utils.action_manager import action_manager
from napari.utils.interactions import Shortcut
from napari.utils.translations import trans

WELCOME_SHORTCUTS = (
    'napari.window.file._image_from_clipboard',
    'napari.window.file.open_files_dialog',
    'napari.window.view.toggle_command_palette',
    'napari:show_shortcuts',
)

WELCOME_TIPS = (
    'You can take a screenshot of the canvas and copy it to your clipboard by pressing {napari.window.file.copy_canvas_screenshot}.',
    'You can change most shortcuts from the File -> Preferences -> Shortcuts menu.',
    'You can right click many components of the graphical interface to access advanced controls.',
    'If you select multiple layers in the layer list, then right click and select "Link Layers", their parameters will be synced.',
    'You can press {Ctrl} and scroll the mouse wheel to move the dimension sliders.',
    'To zoom in on a specific area, hold {Alt} and draw a rectangle around it.',
    'Hold {napari:hold_for_pan_zoom} to pan/zoom in any mode (e.g. while painting).',
    'While painting labels, hold {Alt} and move the cursor left/right to quickly decrease/increase the brush size.',
    'If you have questions, you can reach out on our community chat at napari.zulipchat.com!',
    'The community at forum.image.sc is full of imaging experts sharing knowledge and tools for napari and much, much more!',
)


class QtWelcomeLabel(QLabel):
    """Labels used for main message in welcome page."""


class QtShortcutLabel(QLabel):
    """Labels used for displaying shortcut information in welcome page."""


class QtVersionLabel(QLabel):
    """Label used for displaying version information."""


class QtWelcomeWidget(QWidget):
    """Welcome widget to display initial information, shortcuts, and tips."""

    def __init__(
        self, parent: QWidget | None, tips: Sequence[str] | None = None
    ) -> None:
        super().__init__(parent)

        self._tips = tuple(tips) if tips is not None else WELCOME_TIPS
        self._current_tip = ''

        self._image = QLabel()
        self._image.setObjectName('logo_silhouette')
        self._image.setMinimumSize(300, 300)
        self._version_label = QtVersionLabel(f'napari {__version__}')
        self._label = QtWelcomeLabel(
            trans._('Drag file(s) here to open, or use shortcuts below:')
        )
        self._tip_label = QtShortcutLabel('')

        self.setAutoFillBackground(True)
        self.setAcceptDrops(True)
        self.setProperty('drag', False)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tip_label.setWordWrap(True)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(30)
        text_layout.addWidget(self._version_label)
        text_layout.addWidget(self._label)

        shortcut_layout = QFormLayout()
        self._shortcut_key_labels: list[QtShortcutLabel] = []
        self._shortcut_description_labels: list[QtShortcutLabel] = []
        for _ in WELCOME_SHORTCUTS:
            shortcut_label = QtShortcutLabel('')
            description_label = QtShortcutLabel('')
            self._shortcut_key_labels.append(shortcut_label)
            self._shortcut_description_labels.append(description_label)
            shortcut_layout.addRow(shortcut_label, description_label)
        shortcut_layout.setSpacing(0)

        layout = QVBoxLayout()
        layout.addStretch()
        layout.setSpacing(10)
        layout.addWidget(self._image)
        layout.addLayout(text_layout)
        layout.addLayout(shortcut_layout)
        layout.addSpacing(30)
        layout.addWidget(self._tip_label)
        layout.addStretch()

        self.setLayout(layout)
        self.refresh_shortcuts()
        self.show_random_tip()
        action_manager.events.shorcut_changed.connect(self.refresh_shortcuts)

    def minimumSizeHint(self):
        """Overwrite minimum size to allow creating small viewer instance."""
        return QSize(100, 100)

    def refresh(self, _event=None) -> None:
        self.refresh_shortcuts()
        self._update_tip_label()

    def refresh_shortcuts(self, _event=None) -> None:
        for shortcut_label, description_label, command_id in zip(
            self._shortcut_key_labels,
            self._shortcut_description_labels,
            WELCOME_SHORTCUTS,
            strict=False,
        ):
            shortcut, description = self._command_shortcut_and_description(
                command_id
            )
            shortcut_label.setText(shortcut or '')
            description_label.setText(description or '')

    def set_tips(self, tips: Sequence[str] | None) -> None:
        self._tips = tuple(tips) if tips is not None else WELCOME_TIPS
        self.show_random_tip()

    def show_random_tip(self) -> None:
        tips = self._tips or ("You're awesome!",)
        self._current_tip = choice(tips)
        self._update_tip_label()

    def _update_tip_label(self) -> None:
        if not self._current_tip:
            return
        self._tip_label.setText(
            trans._(
                'Did you know?\n{tip}',
                tip=self._format_tip(self._current_tip),
            )
        )

    def _format_tip(self, tip: str) -> str:
        for match in re.finditer(r'{(.*?)}', tip):
            command_id = match.group(1)
            shortcut, _ = self._command_shortcut_and_description(command_id)
            if shortcut is None:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    shortcut = Shortcut(command_id).platform
            if shortcut:
                tip = tip.replace(match.group(), str(shortcut))
        return tip

    @staticmethod
    def _command_shortcut_and_description(
        command_id: str,
    ) -> tuple[str | None, str | None]:
        app = get_app_model()
        all_shortcuts = get_settings().shortcuts.shortcuts
        keybinding = app.keybindings.get_keybinding(command_id)

        shortcut = command = None
        if keybinding is not None and command_id in app.commands:
            shortcut = Shortcut(keybinding.keybinding).platform
            command = app.commands[command_id].title
        else:
            keybinding = all_shortcuts.get(command_id, [None])[0]
            action = action_manager._actions.get(command_id)
            if keybinding is not None and action is not None:
                shortcut = Shortcut(keybinding).platform
                command = action.description

        return shortcut, command

    def paintEvent(self, event):
        """Override Qt method."""
        option = QStyleOption()
        option.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, option, p, self)

    def _update_property(self, prop, value):
        """Update properties of widget to update style."""
        self.setProperty(prop, value)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event):
        """Provide style updates on drag enter."""
        self._update_property('drag', True)
        if event.mimeData().hasUrls():
            viewer = self.parentWidget().nativeParentWidget()._qt_viewer
            viewer._set_drag_status()
            event.accept()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Provide style updates on drag leave."""
        self._update_property('drag', False)

    def dropEvent(self, event):
        """Provide style updates on drop and emit the drop event."""
        self._update_property('drag', False)
        if (
            self.parent() is not None
            and self.parent().parent() is not None
            and self.parent().parent().parent() is not None
        ):
            self.parent().parent().parent().dropEvent(event)


class QtWidgetOverlay(QStackedWidget):
    """Stacked widget providing switching between the canvas and welcome page."""

    resized = Signal()
    leave = Signal()
    enter = Signal()

    def __init__(
        self,
        parent: QWidget | None,
        widget: QWidget,
        tips: Sequence[str] | None = None,
    ) -> None:
        super().__init__(parent)

        self._overlay = QtWelcomeWidget(self, tips=tips)

        self.addWidget(widget)
        self.addWidget(self._overlay)
        self.setCurrentIndex(0)

    def refresh(self) -> None:
        self._overlay.refresh()

    def set_welcome_visible(self, visible=True):
        """Show welcome screen widget on stack."""
        if visible:
            self._overlay.refresh_shortcuts()
            self._overlay.show_random_tip()
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
