import shutil
import sys
import uuid
from enum import Enum
from pathlib import Path
from time import sleep
from turtle import update

from qtpy.QtCore import QCoreApplication, QObject, QProcess, Qt, Signal
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from napari.utils import progress
from napari.utils.notifications import notification_manager

from ...settings import get_settings
from ...utils import progress
from ...utils.translations import trans
from ..qt_resources import QColoredSVGIcon
from ..qthreading import create_worker


class UpdateAction(Enum):
    update = 'update'
    update_on_quit = 'update_on_quit'


class UpdateOptionsDialog(QDialog):
    """Qt dialog window for displaying Update options.

    Parameters
    ----------
    parent : QWidget, optional
        Parent of the dialog, to correctly inherit and apply theme.
        Default is None.
    version : str, optional
        The new napari version available for update. Default is ``None``.
    """

    def __init__(self, parent=None, version=None):
        super().__init__(parent)
        self._version = version
        self._action = None

        # Widgets
        self._icon = QLabel()
        self._text = QLabel(
            trans._(
                'A new version of napari is available!<br><br>Install <a href="https://napari.org/release/release_{version_underscores}.html">napari {version}</a> to stay up to date.<br>',
                version=self._version,
                version_underscores=self._version.replace(".", "_"),
            )
        )
        self._button_dismiss = QPushButton(trans._("Dismiss"))
        self._button_skip = QPushButton(trans._("Skip this version"))
        self._button_update = QPushButton(trans._("Update"))
        self._button_update_on_quit = QPushButton(trans._("Update on quit"))

        # Setup
        icon = QColoredSVGIcon.from_resources("warning")
        self._icon.setPixmap(icon.colored(color="#E3B617").pixmap(60, 60))
        self._text.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self._text.setOpenExternalLinks(True)
        self._button_update.setObjectName("primary")
        self._button_update_on_quit.setObjectName("primary")
        self.setWindowTitle(trans._("Update napari"))
        self.setMinimumWidth(500)

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
        """
        Parameters
        ----------
        action : UpdateAction
            The action set for the dialog.
        """
        self._action = action
        self.accept()

    def skip(self):
        """Skip this release so it is not taken into account for updates."""
        versions = get_settings().updates.update_version_skip
        if self._version not in versions:
            versions.append(self._version)
            get_settings().updates.update_version_skip = versions

        self.accept()

    def is_update_on_quit(self):
        """Check if option selected is to update on quit."""
        return self._action == UpdateAction.update_on_quit

    def is_update(self):
        """Check if option selected is to update right away."""
        return self._action == UpdateAction.update


class UpdateManager(QObject):
    """Manager for handling the update process.

    Parameters
    ----------
    parent : QObject, optional
        Parent of the manager. Default is None.
    version: str, optional
        Default is ``None``.
    update : boo., optional
        Default is ``False``.
    """

    finished = Signal()
    quit_requested = Signal()

    def __init__(self, parent=None, version=None, update=False):
        super().__init__(parent=parent)
        self._version = version
        self._finished = False
        self._process = None
        self._worker_thread = None
        self._current_progress = [(0, 'update')]
        self._update_keys = [
            ('conda-forge/noarch', trans._('update - repodata')),
            ('Updating specs:', trans._('update - specs')),
            ('Total download:', trans._('update - downloading')),
            ('Preparing transaction:', trans._('update - files')),
            ('Verifying transaction:', trans._('update - certificates')),
            ('Executing transaction:', trans._('update - packages')),
            ('To activate this environment', trans._('update - shortcuts')),
        ]

        self._worker_thread = create_worker(
            self._dummy_process,
            _progress={'total': len(self._update_keys), 'desc': "update"},
        )
        self._worker_thread.yielded.connect(self._handle_yield)

        if update:
            self._update_napari()

    def _handle_yield(self, total):
        i, msg = self._current_progress[-1]
        pbar = self._worker_thread.pbar
        pbar.set_description(msg)
        pbar.total = (
            int(round(((total * len(self._update_keys)) / (i + 1)), 0)) + 1
        )

    def _dummy_process(self):
        i = 0
        while not self._finished:
            sleep(1)
            i += 1
            yield i

    def _update_napari(self):
        """"""
        self._worker_thread.start()
        envs_path = Path(sys.prefix).parent.parent
        env_path = envs_path / "envs" / f"napari-{self._version}"
        if env_path.exists():
            old_path = str(env_path) + '-' + str(uuid.uuid4()) + "-broken"
            env_path.rename(old_path)
            # FIXME: Clean up at the end?
            # shutil.rmtree(old_path, ignore_errors=True)

        args = [
            "create",
            "-p",
            str(env_path),
            f"napari={self._version}=*pyside*",
            # f"napari-menu={self._version}",
            "--channel",
            "conda-forge",
            "--channel",  ## For testing nightly builds
            "napari",  ## For testing nightly builds
            # "--shortcuts-only=napari-menu",
            # "--json",
            "--yes",
        ]
        print(' '.join(args))
        conda_exec = str(envs_path / "condabin" / "conda")
        conda_exec = str(envs_path / "condabin" / "mamba")
        print(conda_exec)
        self._process = QProcess()
        self._process.setProgram(conda_exec)
        self._process.setArguments(args)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_stdout_ready)
        self._process.finished.connect(self._finished_update)
        self._process.start()

    def _on_stdout_ready(self):
        """"""
        text = self._process.readAllStandardOutput().data().decode()
        print([text])

        # Handle common known messages
        for i, (key, value) in enumerate(self._update_keys):
            data = (i, value)
            if key in text and data not in self._current_progress:
                print(f'\n\nFound {key}\n\n')
                self._current_progress.append(data)
                break

    def _finished_update(self):
        """"""
        self._finished = True
        if self._process.exitCode() == 0:
            msg = trans._("Version updated successfully!")
        else:
            msg = trans._("Version could not be updated!")

        notification_manager.receive_info(trans._(msg))
        self.finished.emit()

    def stop(self):
        """Stop the installation process."""
        if self._worker_thread:
            self._worker_thread.quit()

        if self._process:
            self._process.kill()


class UpdateStatusDialog(QDialog):
    """Qt dialog window for displaying Update information.

    Parameters
    ----------
    parent : QWidget, optional
        Parent of the dialog, to correctly inherit and apply theme.
        Default is None.
    """

    def __init__(self, parent=None, version=None):
        super().__init__(parent)

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
        settings = get_settings()
        self._check.setChecked(settings.updates.check_for_updates)
        icon = QColoredSVGIcon.from_resources("warning")
        self._icon.setPixmap(icon.colored(color="#E3B617").pixmap(60, 60))

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
        get_settings().updates.check_for_updates = value
