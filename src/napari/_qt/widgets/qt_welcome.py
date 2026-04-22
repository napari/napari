from __future__ import annotations

import re
import warnings
from random import choice
from typing import TYPE_CHECKING

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QPainter
from qtpy.QtWidgets import (
    QFormLayout,
    QLabel,
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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from napari._qt.qt_viewer import QtViewer


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

TIP_PLACEHOLDER_PATTERN = re.compile(r'{(.*?)}')
SHORTCUTS_SETTINGS_KEY = 'shortcuts.shortcuts'


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

        # Create colored icon using theme
        self._image = QLabel()
        self._image.setObjectName('logo_silhouette')
        self._image.setMinimumSize(300, 300)
        self._version_label = QtVersionLabel(f'napari {__version__}')
        self._label = QtWelcomeLabel(
            trans._('Drag file(s) here to open, or use shortcuts below:')
        )
        self._tip_label = QtShortcutLabel('')

        # Widget setup
        self.setAutoFillBackground(True)
        self.setAcceptDrops(True)
        self.setProperty('drag', False)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tip_label.setWordWrap(True)

        # Layout
        text_layout = QVBoxLayout()
        text_layout.setSpacing(30)
        text_layout.addWidget(self._version_label)
        text_layout.addWidget(self._label)

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
        parent_resized = getattr(self.parentWidget(), 'resized', None)
        if parent_resized is not None:
            parent_resized.connect(self._sync_to_parent)
        self._sync_to_parent()
        self.hide()
        get_settings().events.changed.connect(self._refresh_on_settings_change)
        action_manager.events.shortcut_changed.connect(self.refresh)

    def minimumSizeHint(self) -> QSize:
        """Overwrite minimum size to allow creating small viewer instance."""
        return QSize(100, 100)

    def refresh(self, _event=None) -> None:
        self.refresh_shortcuts()
        self._update_tip_label()

    def _refresh_on_settings_change(self, event) -> None:
        if getattr(event, 'key', '').startswith(SHORTCUTS_SETTINGS_KEY):
            self.refresh(event)

    def refresh_shortcuts(self, _event=None) -> None:
        """Update the shortcut table using the current settings."""
        for (
            command_id,
            shortcut_label,
            description_label,
        ) in self._shortcut_rows:
            shortcut, description = self._command_shortcut_and_description(
                command_id
            )
            shortcut_label.setText(shortcut or '')
            description_label.setText(description or '')

    def set_tips(self, tips: Sequence[str] | None) -> None:
        """Replace the tip pool and show a random one."""
        self._tips = tuple(tips) if tips is not None else WELCOME_TIPS
        self.show_random_tip()

    def set_welcome_visible(self, visible: bool = True) -> None:
        if visible:
            self.refresh_shortcuts()
            self.show_random_tip()
            self._sync_to_parent()
            self.show()
            self.raise_()
        else:
            self.hide()

    def show_random_tip(self) -> None:
        tips = self._tips or ("You're awesome!",)
        self._current_tip = choice(tips)
        self._update_tip_label()

    def _update_tip_label(self) -> None:
        """Render the current tip after expanding any shortcut placeholders."""
        if not self._current_tip:
            return
        self._tip_label.setText(
            trans._(
                'Did you know?\n{tip}',
                tip=self._format_tip(self._current_tip),
            )
        )

    def _format_tip(self, tip: str) -> str:
        """Replace keybinding ``{...}`` placeholders with text glyphs."""

        def replace_placeholder(match: re.Match[str]) -> str:
            command_id = match.group(1)
            shortcut, _ = self._command_shortcut_and_description(command_id)
            if shortcut is None:
                # Some placeholders are literal keys such as ``{Ctrl}`` rather
                # than command ids, so normalize them as standalone shortcuts.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    shortcut = Shortcut(command_id).platform
            return str(shortcut) if shortcut else match.group(0)

        return TIP_PLACEHOLDER_PATTERN.sub(replace_placeholder, tip)

    @staticmethod
    def _command_shortcut_and_description(
        command_id: str,
    ) -> tuple[str | None, str | None]:
        """Return the current shortcut text and label for a command id."""
        app = get_app_model()
        all_shortcuts = get_settings().shortcuts.shortcuts
        keybinding = app.keybindings.get_keybinding(command_id)

        shortcut = command = None
        if keybinding is not None and command_id in app.commands:
            shortcut = Shortcut(keybinding.keybinding).platform
            command = app.commands[command_id].title
        else:
            # Some welcome entries still come from the legacy action manager,
            # so we fall back to the settings-backed shortcut registry here.
            keybinding = all_shortcuts.get(command_id, [None])[0]
            action = action_manager._actions.get(command_id)
            if keybinding is not None and action is not None:
                shortcut = Shortcut(keybinding).platform
                command = action.description

        return shortcut, command

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
        # Qt only reapplies selector rules that depend on dynamic properties
        # after the widget is polished again.
        self.style().unpolish(self)
        self.style().polish(self)

    def _sync_to_parent(self, *_args) -> None:
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())

    def _qt_viewer(self) -> QtViewer | None:
        return getattr(self.window(), '_qt_viewer', None)

    def dragEnterEvent(self, event):
        """Override Qt method.

        Provide style updates on event.

        Parameters
        ----------
        event : qtpy.QtCore.QDragEnterEvent
            Event from the Qt context.
        """
        if not event.mimeData().hasUrls():
            self._update_property('drag', False)
            event.ignore()
            return

        qt_viewer = self._qt_viewer()
        if qt_viewer is None:
            self._update_property('drag', False)
            event.ignore()
            return

        self._update_property('drag', True)
        qt_viewer._set_drag_status()
        event.accept()

    def dragLeaveEvent(self, event):
        """Override Qt method.

        Provide style updates on event.

        Parameters
        ----------
        event : qtpy.QtCore.QDragLeaveEvent
            Event from the Qt context.
        """
        self._update_property('drag', False)

    def dropEvent(self, event):
        """Override Qt method.

        Provide style updates on event and emit the drop event.

        Parameters
        ----------
        event : qtpy.QtCore.QDropEvent
            Event from the Qt context.
        """
        self._update_property('drag', False)
        if qt_viewer := self._qt_viewer():
            qt_viewer.dropEvent(event)
        else:
            event.ignore()
