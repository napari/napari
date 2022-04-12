from re import L
from qtpy import QtGui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QCheckBox,
    QProgressBar,
)

from qtpy.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QProcess,
    QProcessEnvironment,
    QSize,
    Qt,
    Signal,
    Slot,
)
from ..qt_resources import QColoredSVGIcon
from ...utils import citation_text, sys_info
from ...utils.translations import trans
from ...settings import get_settings


class UpdateDialog(QDialog):
    """Qt dialog window for displaying Update information.

    Parameters
    ----------
    parent : QWidget, optional
        Parent of the dialog, to correctly inherit and apply theme.
        Default is None.
    """

    def __init__(self, parent=None, version=None, update=False):
        super().__init__(parent)

        self._version = "0.5.0"
        self._update_on_quit = False
        self._update = update

        # Widgets
        self._icon = QLabel()
        self._text =  QLabel()
        self._progress = QProgressBar()
        self._button_dismiss = QPushButton(trans._("Dismiss"))
        self._button_skip = QPushButton(trans._("Skip this version"))
        self._button_update = QPushButton(trans._("Update"))
        self._button_update_on_quit = QPushButton(trans._("Update on quit"))

        # Setup
        self._progress.setVisible(False)
        self._text.setText(
            trans._(
                'A new version of napari is available!<br> Install <a href="https://napari.org">napari {version}</a> to stay up to date.',
                version="0.5.0",
            )
        )
        icon = QColoredSVGIcon.from_resources("warning")
        self._icon.setPixmap(
            icon.colored(color="#E3B617").pixmap(60, 60)
        )
        self.setWindowTitle(trans._("Update napari"))
        self._text.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self._text.setOpenExternalLinks(True)
        self._button_update.setObjectName("primary")
        self._button_update_on_quit.setObjectName("primary")

        # Signals
        self.rejected.connect(self.dismiss)
        self._button_dismiss.clicked.connect(self.dismiss)
        self._button_skip.clicked.connect(self.skip)
        self._button_update.clicked.connect(self.update)
        self._button_update_on_quit.clicked.connect(self.update_on_quit)

        # Layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self._button_dismiss)
        buttons_layout.addWidget(self._button_skip)
        buttons_layout.addWidget(self._button_update)
        buttons_layout.addWidget(self._button_update_on_quit)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self._text)
        vertical_layout.addSpacing(10)
        vertical_layout.addWidget(self._progress)
        vertical_layout.addSpacing(10)
        vertical_layout.addLayout(buttons_layout)

        layout = QHBoxLayout()
        layout.addWidget(self._icon, alignment=Qt.AlignTop)
        layout.addLayout(vertical_layout)

        self.setLayout(layout)

        if update:
            self._button_update.setVisible(False)
            self._button_update_on_quit.setVisible(False)
            self._button_skip.setVisible(False)
            self.update()

    def _update_progress(self):
        """"""
        pass

    def _setup(self, update_data=None):
        """"""

    def update_on_quit(self):
        self._update_on_quit = True
        print("boom")
        self.accept()

    def update(self, on_quit=False):
        """"""
        args = ["create", "-p", "/Users/gpenacastellanos/opt/miniconda3/envs/boom", "napari", "--channel", "conda-forge", "--json", "--yes"]
        process = QProcess()
        process.setProgram("conda")
        process.setArguments(args)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(lambda process=process: self._on_stdout_ready(process))
        process.finished.connect(self._finished)
        process.start()
        self._button_update_on_quit.setDisabled(True)
        self._button_skip.setDisabled(True)
        self._button_dismiss.setDisabled(True)
        self._progress.setVisible(True)
        self._progress.setMinimum(0)
        self._progress.setMaximum(0)

    def _on_stdout_ready(self, process):
        """"""
        # '{"fetch":"msgpack-python-1.0.3 | 82 KB     | ","finished":false,"maxval":1,"progress":1.000000}'
        text = process.readAllStandardOutput().data().decode()
        print(text)

    def _finished(self):
        """"""
        # Check if it was succesfull or not?
        self._progress.setVisible(False)
        self._text.setText(trans._("Version updated successfully!"))
        self._button_dismiss.setVisible(False)
        self._button_skip.setVisible(False)
        self._button_update.setVisible(False)
        self._button_update_on_quit.setVisible(False)

        if self._update:
            self.accept()

    def skip(self):
        """"""
        # Do something with settings?
        value = get_settings().updates.update_skip
        if self._version not in value:
            value.append(self._version)
            get_settings().updates.update_skip = value

        self.accept()

    def dismiss(self):
        """"""
        self.reject()


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
        self._text =  QLabel(
            trans._(
                'You are using the latest napari, <a href="https://napari.org">v{version}</a>!',
                version=version,
            )
        )
        self._check = QCheckBox()
        self._button_dismiss = QPushButton()

        # Setup
        self.setWindowTitle(trans._("Update napari"))

        # Signals

        # Layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self._button_dismiss)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self._text)
        vertical_layout.addWidget(self._check)
        vertical_layout.addLayout(buttons_layout)

        layout = QHBoxLayout()
        layout.addWidget(self._icon)
        layout.addLayout(vertical_layout)

        self.setLayout(layout)
