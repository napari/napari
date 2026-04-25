from __future__ import annotations

import re
import warnings
from random import choice
from typing import TYPE_CHECKING

from qtpy.QtCore import QSize, Qt, Signal
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
from napari.utils.action_manager import action_manager
from napari.utils.interactions import Shortcut

if TYPE_CHECKING:
    from collections.abc import Sequence


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

        self._tips = tuple(tips) if tips is not None else WELCOME_TIPS
        self._current_tip = ''

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

        #  Widget layout of logo and text elements
        layout = QVBoxLayout()
        layout.addStretch()
        layout.setSpacing(10)
        layout.addWidget(self._image)
        layout.addSpacing(30)
        layout.addWidget(self._version_label)
        layout.addSpacing(30)
        layout.addWidget(self._label)
        layout.addLayout(shortcut_layout)
        layout.addSpacing(30)
        layout.addWidget(self._tip_label)
        layout.addStretch()

        self.setLayout(layout)
        self.refresh_shortcuts()
        self.show_random_tip()
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
        self._tip_label.setText(
            f'Did you know?\n{self._format_tip(self._current_tip)}'
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
        """Return the active shortcut text and label for a command.

        NOte: The shortcut table has to support both app-model commands
        (e.g. ``napari.window.*``) and ``action_manager`` commands
        (e.g. ``napari:*``).
        """
        app = get_app_model()

        if command_id in app.commands:
            keybinding = app.keybindings.get_keybinding(command_id)
            shortcut = (
                Shortcut(keybinding.keybinding).platform
                if keybinding is not None
                else None
            )
            return shortcut, app.commands[command_id].title

        action = action_manager._actions.get(command_id)
        if action is None:
            return None, None

        shortcuts = action_manager._shortcuts.get(command_id, [])
        shortcut = Shortcut(shortcuts[0]).platform if shortcuts else None
        return shortcut, action.description

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
