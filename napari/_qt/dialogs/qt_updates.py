import json
import shutil
import sys
import uuid
from pathlib import Path

from qtpy.QtCore import QCoreApplication, QProcess, Qt, Signal
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

from ...settings import get_settings
from ...utils.translations import trans
from ..qt_resources import QColoredSVGIcon


class UpdateOptionsDialog(QDialog):
    """Qt dialog window for displaying Update options.

    Parameters
    ----------
    parent : QWidget, optional
        Parent of the dialog, to correctly inherit and apply theme.
        Default is None.
    """

    def __init__(self, parent=None, version=None, update=False):
        super().__init__(parent)

        self._version = version
        self._update_on_quit = False
        self._update = update
        self._messages = []
        self._timer = None
        self._action = None

        # Widgets
        self._icon = QLabel()
        self._text = QLabel()
        self._button_dismiss = QPushButton(trans._("Dismiss"))
        self._button_skip = QPushButton(trans._("Skip this version"))
        self._button_update = QPushButton(trans._("Update"))
        self._button_update_on_quit = QPushButton(trans._("Update on quit"))

        # Setup
        self._text.setText(
            trans._(
                'A new version of napari is available!<br> Install <a href="https://napari.org">napari {version}</a> to stay up to date.',
                version=self._version,
            )
        )
        icon = QColoredSVGIcon.from_resources("warning")
        self._icon.setPixmap(icon.colored(color="#E3B617").pixmap(60, 60))
        self.setWindowTitle(trans._("Update napari"))
        self._text.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self._text.setOpenExternalLinks(True)
        self._button_update.setObjectName("primary")
        self._button_update_on_quit.setObjectName("primary")
        self.setMinimumWidth(500)

        # Signals
        # self.rejected.connect(self.reject)
        self._button_dismiss.clicked.connect(self.accept)
        self._button_skip.clicked.connect(self.skip)
        self._button_update.clicked.connect(
            lambda x: self._set_action("update")
        )
        self._button_update_on_quit.clicked.connect(
            lambda x: self._set_action("update_on_quit")
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
        vertical_layout.addSpacing(10)
        vertical_layout.addLayout(buttons_layout)

        layout = QHBoxLayout()
        layout.addWidget(self._icon, alignment=Qt.AlignTop)
        layout.addLayout(vertical_layout)

        self.setLayout(layout)

    def _set_action(self, action):
        """"""
        self._action = action
        self.accept()

    def skip(self):
        """"""
        # Do something with settings?
        self._action = "skip"
        versions = get_settings().updates.update_version_skip
        if self._version not in versions:
            versions.append(self._version)
            get_settings().updates.update_version_skip = versions

        self.accept()


class UpdateDialog(QDialog):
    """Qt dialog window for displaying Update information.

    Parameters
    ----------
    parent : QWidget, optional
        Parent of the dialog, to correctly inherit and apply theme.
        Default is None.
    """

    finished = Signal(object, object)
    quit_requested = Signal()

    def __init__(self, parent=None, version=None, update=False):
        super().__init__(parent)

        self._version = version
        self._update_on_quit = False
        self._messages = []
        self._timer = None
        self._finished = False

        # Widgets
        self._icon = QLabel()
        self._text = QLabel()
        self._progress = QProgressBar()
        self._progress_text = QLabel("Downloading...")

        self._button_launch = QPushButton(
            trans._("Quit and launch new version")
        )
        self._button_hide = QPushButton(trans._("Hide"))

        # Setup
        self._button_launch.setVisible(False)
        self._button_launch.setObjectName("primary")
        self._progress_text.setAlignment(Qt.AlignRight)
        self._progress.setMinimum(0)
        self._progress.setMaximum(0)
        self._text.setText(
            trans._(
                'Installing napari v{version}!',
                version=version,
            )
        )
        icon = QColoredSVGIcon.from_resources("warning")
        self._icon.setPixmap(icon.colored(color="#E3B617").pixmap(60, 60))
        self.setWindowTitle(trans._("Update napari"))
        self._text.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self._text.setOpenExternalLinks(True)
        self.setMinimumWidth(500)

        # Signals
        self._button_hide.clicked.connect(self.hide)
        self._button_launch.clicked.connect(self.launch)

        # Layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        buttons_layout.addWidget(self._button_hide)
        buttons_layout.addWidget(self._button_launch)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self._text)
        vertical_layout.addWidget(self._progress)
        vertical_layout.addWidget(self._progress_text)
        vertical_layout.addLayout(buttons_layout)

        layout = QHBoxLayout()
        layout.addWidget(self._icon, alignment=Qt.AlignTop)
        layout.addLayout(vertical_layout)

        self.setLayout(layout)
        self.update_napari()

    def update_napari(self):
        """"""
        envs_path = Path(sys.prefix).parent.parent
        env_path = envs_path / "envs" / f"napari-{self._version}"
        if env_path.exists():
            old_path = str(env_path) + "-old-" + str(uuid.uuid4())
            env_path.rename(old_path)
            shutil.rmtree(old_path)

        args = [
            "create",
            "-p",
            str(env_path),
            f"napari={self._version}=*pyside2",
            # f"napari-menu={self._version}",
            "--channel",
            "conda-forge",
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
        self._process.readyReadStandardOutput.connect(
            lambda process=self._process: self._on_stdout_ready(process)
        )
        self._process.finished.connect(self._finished_update)
        self._process.start()
        self._progress.setVisible(True)
        self._progress.setMinimum(0)
        self._progress.setMaximum(0)
        self._messages = []

        # self._timer = QTimer(200)
        # self._timer.timeout.connect(self._update_progress)
        # self._timer.start()

    def _on_stdout_ready(self, process):
        """"""
        # '{"fetch":"msgpack-python-1.0.3 | 82 KB     | ","finished":false,"maxval":1,"progress":1.000000}'
        text = process.readAllStandardOutput().data().decode()
        try:
            data = text.split("\n\x00")
            print(data)
        except Exception as e:
            print(e)
            data = {}

        print(text)
        for chunk in data:
            try:
                chunk = json.loads(chunk)
            except Exception:
                chunk = {}

            self._messages.append(chunk)

            if "fetch" in chunk:
                self._progress.setMinimum(chunk["progress"])
                self._progress.setMaximum(chunk["maxval"])
                self._progress_text.setText(chunk["fetch"])
                QCoreApplication.processEvents()

    def _update_progress(self):
        """Update state of progress bar with a timer."""
        if self._messages:
            chunk = self._messages.pop(0)
            if "fetch" in chunk:
                self._progress.setMinimum(chunk["progress"])
                self._progress.setMaximum(chunk["maxval"])
                self._progress_text.setText(chunk["fetch"])
                QCoreApplication.processEvents()

    def _finished_update(self):
        """"""
        self._button_hide.setText("Dismiss")
        self._progress.setVisible(False)
        self._progress_text.setVisible(False)

        if self._process.exitCode() == 0:
            self._text.setText(trans._("Version updated successfully!"))
            self._button_launch.setVisible(True)
        else:
            self._text.setText(trans._("Version could not be updated!"))
            self._button_launch.setVisible(False)

        if self._timer:
            self._timer.stop()

        self.finished.emit("ok", "error")

    def launch(self):
        """"""
        envs_path = Path(sys.prefix).parent.parent
        napari_path = envs_path / "envs" / f"napari-{self._version}"
        napari_exec = napari_path / "bin" / "napari"
        print(str(napari_exec))
        process = QProcess()
        process.setProgram(str(napari_exec))
        process.startDetached()
        self.quit_requested.emit()

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.hide()

    def stop(self):
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
