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

# from ..utils.misc import running_as_bundled_app
from ..utils._appdirs import user_plugin_dir, user_site_packages


class QtPipDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        title = QLabel("Install/Uninstall Packages:")
        title.setObjectName("h2")
        self.line_edit = QLineEdit()
        self.install_button = QPushButton("install", self)
        self.uninstall_button = QPushButton("uninstall", self)
        text_area = QTextEdit(self, readOnly=True)
        hlay = QHBoxLayout()
        hlay.addWidget(self.line_edit)
        hlay.addWidget(self.install_button)
        hlay.addWidget(self.uninstall_button)
        layout.addWidget(title)
        layout.addLayout(hlay)
        layout.addWidget(text_area)
        self.setLayout(layout)

        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        env = QProcessEnvironment()
        env.insert("PYTHONPATH", user_site_packages())
        self.process.setProcessEnvironment(env)
        self.process.setProcessChannelMode(QProcess.MergedChannels)

        def on_stdout_ready():
            text = self.process.readAllStandardOutput().data().decode()
            text_area.append(text)

        self.process.readyReadStandardOutput.connect(on_stdout_ready)

        def _install():
            from ..plugins import plugin_manager

            text_area.clear()
            cmd = ['-m', 'pip', 'install']
            # if running_as_bundled_app() and sys.platform.startswith('linux'):
            cmd += ['--prefix', user_plugin_dir(), '--no-warn-script-location']
            self.process.setArguments(cmd + self.line_edit.text().split())
            self.process.start()
            self.process.finished.connect(plugin_manager.discover)

        def _uninstall():
            from ..plugins import plugin_manager

            text_area.clear()
            args = ['-m', 'pip', 'uninstall', '-y']
            self.process.setArguments(args + self.line_edit.text().split())
            self.process.start()
            self.process.finished.connect(plugin_manager.prune)

        self.install_button.clicked.connect(_install)
        self.uninstall_button.clicked.connect(_uninstall)

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setFixedSize(700, 400)
        self.setMaximumHeight(800)
        self.setMaximumWidth(1280)
