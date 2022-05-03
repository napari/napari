from enum import Enum

from qtpy.QtCore import Qt
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from ... import settings
from ...utils.theme import get_theme
from ...utils.translations import trans
from ...utils.updates import InstallerTypes
from ..qt_resources import QColoredSVGIcon


class UpdateAction(Enum):
    update = 'update'
    update_on_quit = 'update_on_quit'


class UpdateOptionsDialog(QDialog):
    """Qt dialog window for displaying Update options.

    Parameters
    ----------
    parent : QWidget, optional
        Parent of the dialog, to correctly inherit and apply theme.
        Default is ``None``.
    version : str, optional
        The new napari version available for update. Default is ``None``.
    installer : str, optional
        The new napari version available for update. Default is ``None``.
    """

    def __init__(
        self, parent=None, version=None, installer: InstallerTypes = None
    ):
        super().__init__(parent)
        self._version = version
        self._action = None
        self._installer = installer

        if version:
            msg = trans._(
                # TODO: Check if link in docs is stable
                # https://napari.org/stable/release/release_{version_underscores}.html
                'A new version of napari is available!<br><br>Install <a href="https://napari.org/release/release_{version_underscores}.html">napari {version}</a> to stay up to date.<br>',
                version=self._version,
                version_underscores=self._version.replace(".", "_"),
            )
        else:
            msg = trans._(
                'A new version of napari is available!<br><br>Update to stay up to date.<br>',
            )

        # Widgets
        self._icon = QLabel()
        self._text = QLabel(msg)
        self._button_dismiss = QPushButton(trans._("Dismiss"))
        self._button_skip = QPushButton(trans._("Skip this version"))
        self._button_update = QPushButton(trans._("Update"))
        self._button_update_on_quit = QPushButton(trans._("Update on quit"))

        # Setup
        theme_name = settings.get_settings().appearance.theme
        theme = get_theme(theme_name, as_dict=True)
        icon = QColoredSVGIcon.from_resources("update_available")
        self._icon.setPixmap(icon.colored(color=theme['icon']).pixmap(60, 60))
        self._text.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self._text.setOpenExternalLinks(True)
        self._button_update.setObjectName("primary")
        self._button_update_on_quit.setObjectName("primary")
        self.setWindowTitle(trans._("Update napari"))
        self.setMinimumWidth(500)

        if self._installer == "pip":
            self._button_skip.setVisible(False)
            self._button_update.setVisible(False)
            self._button_update_on_quit.setVisible(False)

        # Signals
        self._button_dismiss.clicked.connect(self.accept)
        self._button_skip.clicked.connect(self.skip)
        self._button_update.clicked.connect(
            lambda _: self._set_action(UpdateAction.update)
        )
        self._button_update_on_quit.clicked.connect(
            lambda _: self._set_action(UpdateAction.update_on_quit)
        )

        # Layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        buttons_layout.addWidget(self._button_dismiss)
        buttons_layout.addWidget(self._button_skip)
        buttons_layout.addWidget(self._button_update)
        buttons_layout.addWidget(self._button_update_on_quit)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self._text)
        vertical_layout.addStretch()
        vertical_layout.addLayout(buttons_layout)

        layout = QHBoxLayout()
        layout.addWidget(self._icon, alignment=Qt.AlignTop)
        layout.addLayout(vertical_layout)

        self.setLayout(layout)

    def _set_action(self, action: UpdateAction):
        """Set the action to perform when the dialog is accepted.

        Parameters
        ----------
        action : UpdateAction
            The action set for the dialog.
        """
        self._action = action
        self.accept()

    def skip(self):
        """Skip this release so it is not taken into account for updates."""
        _settings = settings.get_settings()
        versions = _settings.updates.update_version_skip
        if self._version and self._version not in versions:
            versions.append(self._version)
            _settings.updates.update_version_skip = versions

        self.accept()

    def is_update_on_quit(self):
        """Check if option selected is to update on quit."""
        return self._action == UpdateAction.update_on_quit

    def is_update(self):
        """Check if option selected is to update right away."""
        return self._action == UpdateAction.update


class UpdateStatusDialog(QDialog):
    """Qt dialog window for displaying Update information.

    Parameters
    ----------
    parent : QWidget, optional
        Parent of the dialog, to correctly inherit and apply theme.
        Default is None.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        _settings = settings.get_settings()

        # Widgets
        self._icon = QLabel()
        self._text = QLabel(
            trans._(
                'You are using the latest napari!',
            )
        )
        self._check = QCheckBox(trans._("Automatically check for updates"))
        self._button_dismiss = QPushButton(trans._("Dismiss"))

        # Setup
        self.setMinimumWidth(500)
        self.setWindowTitle(trans._("Update napari"))
        theme_name = _settings.appearance.theme
        theme = get_theme(theme_name, as_dict=True)
        icon = QColoredSVGIcon.from_resources("update_ready")
        self._icon.setPixmap(icon.colored(color=theme['icon']).pixmap(60, 60))
        self._check.setChecked(_settings.updates.check_for_updates)

        # Signals
        self._check.clicked.connect(self._update_settings)
        self._button_dismiss.clicked.connect(self.close)

        # Layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        buttons_layout.addWidget(self._button_dismiss)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self._text)
        vertical_layout.addWidget(self._check)
        vertical_layout.addLayout(buttons_layout)

        layout = QHBoxLayout()
        layout.addWidget(self._icon)
        layout.addLayout(vertical_layout)

        self.setLayout(layout)

    def _update_settings(self, value: bool) -> None:
        """Update settings for automatically checking updates."""
        settings.get_settings().updates.check_for_updates = value


class UpdateErrorDialog(QDialog):
    """Qt dialog window for displaying Update errors.

    Parameters
    ----------
    parent : QWidget, optional
        Parent of the dialog, to correctly inherit and apply theme.
        Default is None.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        _settings = settings.get_settings()

        # Widgets
        self._icon = QLabel()
        self._text = QLabel(
            trans._(
                'napari could not be updated!',
            )
        )
        self._error = QTextEdit()
        self._button_dismiss = QPushButton(trans._("Dismiss"))

        # Setup
        self.setMinimumWidth(500)
        self.setMaximumHeight(200)
        self.setWindowTitle(trans._("Update napari"))
        theme_name = _settings.appearance.theme
        theme = get_theme(theme_name, as_dict=True)
        icon = QColoredSVGIcon.from_resources("warning")
        self._icon.setPixmap(icon.colored(color=theme['icon']).pixmap(60, 60))

        # Signals
        self._button_dismiss.clicked.connect(self.accept)

        # Layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        buttons_layout.addWidget(self._button_dismiss)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self._text)
        vertical_layout.addWidget(self._error)
        vertical_layout.addLayout(buttons_layout)

        layout = QHBoxLayout()
        layout.addWidget(self._icon, alignment=Qt.AlignTop)
        layout.addLayout(vertical_layout)

        self.setLayout(layout)

    def setErrorText(self, error):
        """Set error text on dialog."""
        self._error.setText(error)
        self._error.moveCursor(QTextCursor.End)
