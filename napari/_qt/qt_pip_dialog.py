import os
import sys

from qtpy.QtCore import QProcess, QProcessEnvironment, Qt
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from ..utils._appdirs import user_plugin_dir, user_site_packages
from ..utils.misc import running_as_bundled_app


# TODO: this is a convenience for now, but will be superceeded by #1357
class QtPipDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setAttribute(Qt.WA_DeleteOnClose)

        # create install process
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        # setup process path
        env = QProcessEnvironment()
        combined_paths = os.pathsep.join(
            [user_site_packages(), env.systemEnvironment().value("PYTHONPATH")]
        )
        env.insert("PYTHONPATH", combined_paths)
        self.process.setProcessEnvironment(env)

        # connections
        self.install_button.clicked.connect(self._install)
        self.uninstall_button.clicked.connect(self._uninstall)
        self.process.readyReadStandardOutput.connect(self._on_stdout_ready)

        from ..plugins import plugin_manager

        self.process.finished.connect(plugin_manager.discover)
        self.process.finished.connect(plugin_manager.prune)

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        title = QLabel("Install/Uninstall Packages:")
        title.setObjectName("h2")
        self.line_edit = QLineEdit()
        self.install_button = QPushButton("install", self)
        self.uninstall_button = QPushButton("uninstall", self)
        self.text_area = QTextEdit(self, readOnly=True)
        hlay = QHBoxLayout()
        hlay.addWidget(self.line_edit)
        hlay.addWidget(self.install_button)
        hlay.addWidget(self.uninstall_button)
        layout.addWidget(title)
        layout.addLayout(hlay)
        layout.addWidget(self.text_area)
        self.setFixedSize(700, 400)
        self.setMaximumHeight(800)
        self.setMaximumWidth(1280)

    def _install(self):
        cmd = ['-m', 'pip', 'install']
        if running_as_bundled_app() and sys.platform.startswith('linux'):
            cmd += [
                '--no-warn-script-location',
                '--prefix',
                user_plugin_dir(),
            ]
        self.process.setArguments(cmd + self.line_edit.text().split())
        self.text_area.clear()
        self.process.start()

    def _uninstall(self):
        args = ['-m', 'pip', 'uninstall', '-y']
        self.process.setArguments(args + self.line_edit.text().split())
        self.text_area.clear()
        self.process.start()

    def _on_stdout_ready(self):
        text = self.process.readAllStandardOutput().data().decode()
        self.text_area.append(text)
