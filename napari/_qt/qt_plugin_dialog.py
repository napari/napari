import os
import sys
from typing import List

from pkg_resources import parse_version
from qtpy.QtCore import QProcess, QProcessEnvironment, QSize, Qt, Slot
from qtpy.QtGui import QFont, QMovie, QFontMetrics
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..plugins.pypi import (
    ProjectInfo,
    iter_napari_plugin_info,
    normalized_name,
)
from ..utils._appdirs import user_plugin_dir, user_site_packages
from ..utils.misc import running_as_bundled_app
from .qt_plugin_sorter import QtPluginSorter
from .threading import create_worker

LOADER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'resources', "loading.gif")
)


class Installer:
    def __init__(self, output_widget: QTextEdit = None):
        from ..plugins import plugin_manager

        # create install process
        self._output_widget = None
        self.process = QProcess()
        self.process.setProgram(sys.executable)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_stdout_ready)
        # setup process path
        env = QProcessEnvironment()
        combined_paths = os.pathsep.join(
            [user_site_packages(), env.systemEnvironment().value("PYTHONPATH")]
        )
        env.insert("PYTHONPATH", combined_paths)
        self.process.setProcessEnvironment(env)
        self.process.finished.connect(lambda: plugin_manager.discover())
        self.process.finished.connect(lambda: plugin_manager.prune())
        self.set_output_widget(output_widget)

    def set_output_widget(self, output_widget: QTextEdit):
        if output_widget:
            self._output_widget = output_widget
            self.process.setParent(output_widget)

    def _on_stdout_ready(self):
        if self._output_widget:
            text = self.process.readAllStandardOutput().data().decode()
            self._output_widget.append(text)

    def install(self, pkg_list: List[str]):
        cmd = ['-m', 'pip', 'install', '--upgrade']
        if running_as_bundled_app() and sys.platform.startswith('linux'):
            cmd += [
                '--no-warn-script-location',
                '--prefix',
                user_plugin_dir(),
            ]
        self.process.setArguments(cmd + pkg_list)
        if self._output_widget:
            self._output_widget.clear()
        self.process.start()

    def uninstall(self, pkg_list: List[str]):
        args = ['-m', 'pip', 'uninstall', '-y']
        self.process.setArguments(args + pkg_list)
        if self._output_widget:
            self._output_widget.clear()
        self.process.start()


class ElidingLabel(QLabel):
    def __init__(self, text='', parent=None):
        super().__init__(parent)
        self._text = text
        self.fm = QFontMetrics(self.font())

    def setText(self, txt):
        self._text = txt
        short = self.fm.elidedText(self._text, Qt.ElideRight, self.width())
        super().setText(short)

    def resizeEvent(self, rEvent):
        width = rEvent.size().width()
        short = self.fm.elidedText(self._text, Qt.ElideRight, width)
        super().setText(short)
        rEvent.accept()


class PluginListItem(QFrame):
    def __init__(
        self,
        package_name: str,
        version: str = '',
        url: str = '',
        summary: str = '',
        author: str = '',
        license: str = "UNKNOWN",
        *,
        plugin_name: str = None,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.setup_ui()
        if plugin_name:
            self.plugin_name.setText(plugin_name)
            self.package_name.setText(f"{package_name} {version}")
            self.summary.setText(summary)
            self.summary.setIndent(25)
            self.package_author.setText(author)
            self.action_button.setText("remove")
            self.action_button.setObjectName("remove_button")
        else:
            self.plugin_name.setText(package_name)
            self.package_name.setText(version)
            self.summary.setText(summary)
            self.package_author.setText(author)
            self.action_button.setText("install")
            self.enabled_checkbox.hide()

    def setup_ui(self):
        self.v_lay = QVBoxLayout(self)
        self.v_lay.setContentsMargins(-1, -1, -1, 12)
        self.v_lay.setSpacing(0)
        self.row1 = QHBoxLayout()
        self.row1.setSpacing(12)
        self.enabled_checkbox = QCheckBox(self)
        self.enabled_checkbox.setChecked(True)
        self.enabled_checkbox.setDisabled(True)
        self.enabled_checkbox.setToolTip("enable/disable")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.enabled_checkbox.sizePolicy().hasHeightForWidth()
        )
        self.enabled_checkbox.setSizePolicy(sizePolicy)
        self.enabled_checkbox.setMinimumSize(QSize(20, 0))
        self.enabled_checkbox.setText("")
        self.row1.addWidget(self.enabled_checkbox)
        self.plugin_name = QLabel(self)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.plugin_name.sizePolicy().hasHeightForWidth()
        )
        self.plugin_name.setSizePolicy(sizePolicy)
        font18 = QFont()
        font18.setPointSize(18)
        self.plugin_name.setFont(font18)
        self.row1.addWidget(self.plugin_name)
        self.package_name = QLabel(self)
        self.package_name.setAlignment(
            Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter
        )
        self.row1.addWidget(self.package_name)
        self.action_button = QPushButton(self)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.action_button.sizePolicy().hasHeightForWidth()
        )
        self.action_button.setSizePolicy(sizePolicy)
        self.row1.addWidget(self.action_button)
        self.v_lay.addLayout(self.row1)
        self.row2 = QHBoxLayout()
        self.row2.setContentsMargins(-1, 4, 0, -1)
        self.summary = ElidingLabel(parent=self)
        sizePolicy = QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.summary.sizePolicy().hasHeightForWidth()
        )
        self.summary.setSizePolicy(sizePolicy)
        font11 = QFont()
        font11.setPointSize(11)
        self.summary.setFont(font11)
        self.row2.addWidget(self.summary)
        self.package_author = QLabel(self)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.package_author.sizePolicy().hasHeightForWidth()
        )
        self.package_author.setSizePolicy(sizePolicy)
        self.package_author.setFont(font11)
        self.row2.addWidget(self.package_author)
        self.v_lay.addLayout(self.row2)


class QPluginList(QListWidget):
    def __init__(self, parent: QWidget, installer: Installer):
        super().__init__(parent)
        self.installer = installer
        self.setSortingEnabled(True)

    @Slot(ProjectInfo)
    def addItem(self, project_info: ProjectInfo, plugin_name=None):
        # don't add duplicates
        if self.findItems(project_info.name, Qt.MatchFixedString):
            if not plugin_name:
                return

        item = QListWidgetItem(project_info.name, parent=self)
        item.version = project_info.version
        super().addItem(item)

        widg = PluginListItem(
            *project_info, parent=self, plugin_name=plugin_name
        )
        method = getattr(
            self.installer, 'uninstall' if plugin_name else 'install'
        )
        widg.action_button.clicked.connect(lambda: method([project_info.name]))

        item.setSizeHint(widg.sizeHint())
        self.setItemWidget(item, widg)

    @Slot(ProjectInfo)
    def tag_outdated(self, project_info: ProjectInfo):
        for item in self.findItems(project_info.name, Qt.MatchFixedString):
            current = item.version
            latest = project_info.version
            if parse_version(current) >= parse_version(latest):
                continue
            if hasattr(item, 'outdated'):
                # already tagged it
                continue
            item.outdated = True
            widg = self.itemWidget(item)
            update_btn = QPushButton(f"update (v{latest})", widg)
            update_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            update_btn.clicked.connect(
                lambda: self.installer.install([item.text()])
            )
            widg.row1.insertWidget(3, update_btn)


class QtPluginDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.installer = Installer()
        self.setup_ui()
        self.installer.set_output_widget(self.stdout_text)
        self.buttonBox.rejected.connect(self.reject)
        self.installer.process.started.connect(
            lambda: self.show_status_btn.setChecked(True)
        )
        self.installer.process.finished.connect(
            lambda: self.show_status_btn.setChecked(False)
        )
        self.installer.process.started.connect(self.working.show)
        self.installer.process.finished.connect(self.working.hide)

        self.installer.process.finished.connect(self.refresh)
        self.installer.process.finished.connect(self.refresh)
        self.refresh()

    def refresh(self):
        self.installed_list.clear()
        self.available_list.clear()

        # fetch installed
        from ..plugins import plugin_manager

        plugin_manager.discover()  # since they might not be loaded yet

        already_installed = []
        for d in plugin_manager.list_plugin_metadata():
            already_installed.append(d['package'])
            # not showing these in the plugin dialog
            if d['plugin_name'] in ('builtins', 'napari-plugin-engine'):
                continue
            self.installed_list.addItem(
                ProjectInfo(
                    normalized_name(d.get("package") or ''),
                    d['version'],
                    d['url'],
                    '',  # TODO: get summary locally
                    d['author'],
                    d['license'],
                ),
                plugin_name=d['plugin_name'],
            )

        # fetch available plugins
        self.worker = create_worker(iter_napari_plugin_info)

        def _handle_yield(project_info):
            if project_info.name in already_installed:
                self.installed_list.tag_outdated(project_info)
            else:
                self.available_list.addItem(project_info)

        self.worker.yielded.connect(_handle_yield)
        self.worker.finished.connect(self.working.hide)
        self.worker.start()

    def setup_ui(self):
        self.resize(1080, 640)
        self.vlay_1 = QVBoxLayout(self)
        self.h_splitter = QSplitter(self)
        self.h_splitter.setOrientation(Qt.Horizontal)
        self.v_splitter = QSplitter(self.h_splitter)
        self.v_splitter.setOrientation(Qt.Vertical)
        self.plugin_lists = QWidget(self.v_splitter)
        self.plugin_lists.setMinimumWidth(440)
        self.stdout_text = QTextEdit(self.v_splitter)
        self.stdout_text.hide()
        self.vlay_2 = QVBoxLayout(self.plugin_lists)
        self.vlay_2.setContentsMargins(0, 0, 0, 0)
        self.installed_label = QLabel("Installed Plugins", self.plugin_lists)
        self.vlay_2.addWidget(self.installed_label)
        self.installed_list = QPluginList(self.plugin_lists, self.installer)
        self.vlay_2.addWidget(self.installed_list)
        horiz = QHBoxLayout()
        horiz.addWidget(QLabel("Available Plugin Packages", self))
        mov = QMovie(LOADER)
        mov.setScaledSize(QSize(18, 18))
        self.working = QLabel("loading ...", self)
        self.working.setMovie(mov)
        mov.start()
        self.working.setAlignment(Qt.AlignRight)
        horiz.addWidget(self.working)
        self.vlay_2.addLayout(horiz)
        self.available_list = QPluginList(self.plugin_lists, self.installer)
        self.vlay_2.addWidget(self.available_list)
        self.plugin_sorter = QtPluginSorter(parent=self.h_splitter)
        self.plugin_sorter.hide()
        self.vlay_1.addWidget(self.h_splitter)
        self.buttonBox = QDialogButtonBox(self)
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Close)
        self.show_status_btn = self.buttonBox.addButton(
            "Show Status", QDialogButtonBox.HelpRole
        )
        self.show_sorter_btn = self.buttonBox.addButton(
            "Show Sorter", QDialogButtonBox.ActionRole
        )

        self.show_status_btn.setCheckable(True)
        self.show_sorter_btn.setCheckable(True)
        self.show_status_btn.setChecked(False)
        self.show_sorter_btn.setChecked(False)
        self.show_status_btn.toggled.connect(self._toggle_status)
        self.show_sorter_btn.toggled.connect(self._toggle_sorter)
        self.vlay_1.addWidget(self.buttonBox)
        self.v_splitter.setStretchFactor(1, 2)
        self.h_splitter.setStretchFactor(0, 2)

    def _toggle_sorter(self, show):
        if show:
            self.show_sorter_btn.setText("Hide Sorter")
            self.plugin_sorter.show()
        else:
            self.show_sorter_btn.setText("Show Sorter")
            self.plugin_sorter.hide()

    def _toggle_status(self, show):
        if show:
            self.show_status_btn.setText("Hide Status")
            self.stdout_text.show()
        else:
            self.show_status_btn.setText("Show Status")
            self.stdout_text.hide()
