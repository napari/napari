import os
from enum import Enum, auto
from functools import partial
import sys
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path
from typing import Optional, Sequence, Tuple

from npe2 import PackageMetadata, PluginManager
from qtpy.QtCore import QEvent, QPoint, QSize, Qt, Slot
from qtpy.QtGui import QFont, QMovie
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible, QElidingLabel

import napari.resources
from napari._qt.dialogs.qt_package_installer import (
    InstallerActions,
    InstallerQueue,
    InstallerTools,
)
from napari._qt.qt_resources import QColoredSVGIcon
from napari._qt.qthreading import create_worker
from napari._qt.widgets.qt_message_popup import WarnPopup
from napari._qt.widgets.qt_tooltip import QtToolTipLabel
from napari.plugins import plugin_manager
from napari.plugins.hub import iter_hub_plugin_info
from napari.plugins.pypi import iter_napari_plugin_info
from napari.plugins.utils import normalized_name
from napari.settings import get_settings
from napari.utils.misc import (
    parse_version,
    running_as_bundled_app,
    running_as_constructor_app,
)
from napari.utils.translations import trans

InstallerTypes = Literal['pip', 'mamba']


DEFAULT_CHANNEL = "conda-forge"


class superQCollapsible(QCollapsible):
    '''QCallpsible class that emits a signal when toggled.'''

    toggled = Signal()

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(title=title, parent=parent)

    def _toggle(self):
        '''Overwrites toggle method in order to emit signal.'''
        super()._toggle()
        self.toggled.emit()


# TODO: add error icon and handle pip install errors
class Installer(QObject):

    started = Signal()
    finished = Signal(int)

    def __init__(
        self,
        output_widget: QTextEdit = None,
        installer: InstallerTypes = "pip",
    ):
        super().__init__()
        self._queue: List[Tuple[Tuple[str, ...], Callable[[], QProcess]]] = []
        self._processes: Dict[Tuple[str, ...], QProcess] = {}
        self._exit_code = 0
        self._conda_env_path = None
        self._installer_type = installer

        if (Path(sys.prefix) / "conda-meta").is_dir():
            self._conda_env_path = sys.prefix

        # create install process
        self._output_widget = output_widget
        self.process = None

    def _create_process(
        self,
        installer: InstallerTypes = "pip",
    ):
        process = QProcess()
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda process=process: self._on_stdout_ready(process)
        )
        env = QProcessEnvironment.systemEnvironment()

        if installer == "pip":
            process.setProgram(self._sys_executable_or_bundled_python())
            # patch process path
            combined_paths = os.pathsep.join(
                [
                    user_site_packages(),
                    env.systemEnvironment().value("PYTHONPATH"),
                ]
            )
            env.insert("PYTHONPATH", combined_paths)
            env.insert("PIP_USER_AGENT_USER_DATA", _user_agent())
        else:
            process.setProgram(installer)

        if installer == "mamba":
            from napari._version import version_tuple

            # To avoid napari version changing when installing a plugin, we
            # add a pin to the current napari version, that way we can
            # restrict any changes to the actual napari application.
            # Conda/mamba also pin python by default, so we effectively
            # constrain python and napari versions from changing, when
            # installing plugins inside the constructor bundled application.
            # See: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#preventing-packages-from-updating-pinning
            napari_version = ".".join(str(v) for v in version_tuple[:3])
            if env.contains("CONDA_PINNED_PACKAGES"):
                # str delimiter is '&'
                system_pins = f"&{env.value('CONDA_PINNED_PACKAGES')}"
            else:
                system_pins = ""
            env.insert(
                "CONDA_PINNED_PACKAGES",
                f"napari={napari_version}{system_pins}",
            )
            if os.name == "nt":
                # workaround https://github.com/napari/napari/issues/4247, 4484
                if not env.contains("TEMP"):
                    temp = gettempdir()
                    env.insert("TMP", temp)
                    env.insert("TEMP", temp)
                if not env.contains("USERPROFILE"):
                    env.insert("HOME", os.path.expanduser("~"))
                    env.insert("USERPROFILE", os.path.expanduser("~"))

        process.setProcessEnvironment(env)
        self.set_output_widget(self._output_widget)
        process.finished.connect(
            lambda ec, es: self._on_process_finished(process, ec, es)
        )  # FIXME connecting lambda to finished signal is bug creating and may end with segfault when garbage
        # collection will consume Installer object before process end.
        return process

    def _sys_executable_or_bundled_python(self):
        # Note: is_bundled_app() returns False even if using a Briefcase bundle...
        # Workaround: see if sys.executable is set to something something napari on Mac
        if sys.executable.endswith("napari") and sys.platform == 'darwin':
            # sys.prefix should be <napari.app>/Contents/Resources/Support/Python/Resources
            python = os.path.join(sys.prefix, "bin", "python3")
            if os.path.isfile(python):
                return python
        return sys.executable

    def set_output_widget(self, output_widget: QTextEdit):
        if output_widget:
            self._output_widget = output_widget

    def _on_process_finished(self, process, exit_code, exit_status):
        if exit_code != 0:
            self._exit_code = 0

        process_to_terminate = [
            pkg_list
            for pkg_list, proc in self._processes.items()
            if proc == process
        ]

        for pkg_list in process_to_terminate:
            process = self._processes.pop(pkg_list)
            process.terminate()

        self._handle_action()

    def _on_stdout_ready(self, process):
        if self._output_widget:
            text = process.readAllStandardOutput().data().decode()
            self._output_widget.append(text)

    def _handle_action(self):
        if self._queue:
            pkg_list, func = self._queue.pop()
            self.started.emit()
            process = func()
            self._processes[pkg_list] = process

        if not self._processes:
            from napari.plugins import plugin_manager

            plugin_manager.discover()
            plugin_manager.prune()
            self.finished.emit(self._exit_code)

    def install(
        self,
        pkg_list: Sequence[str],
        installer: Optional[InstallerTypes] = None,
        channels: Sequence[str] = ("conda-forge",),
        versions: Optional[Sequence[str]] = None,
    ):
        installer = installer or self._installer_type
        self._queue.insert(
            0,
            (
                tuple(pkg_list),
                lambda: self._install(pkg_list, installer, channels, versions),
            ),
        )
        self._handle_action()

    def _install(
        self,
        pkg_list: Sequence[str],
        installer: Optional[InstallerTypes] = None,
        channels: Sequence[str] = ("conda-forge",),
        versions: Optional[Sequence[str]] = None,
    ):
        installer = installer or self._installer_type
        process = self._create_process(installer)
        if installer != "pip":
            cmd = [
                'install',
                '-y',
                '--prefix',
                self._conda_env_path,
            ]
            for channel in channels:
                cmd.extend(["-c", channel])
        else:
            if versions:
                cmd = ['-m', 'pip', 'install']
            else:
                cmd = ['-m', 'pip', 'install', '--upgrade']

        if (
            running_as_bundled_app()
            and sys.platform.startswith('linux')
            and not self._conda_env_path
        ):
            cmd += [
                '--no-warn-script-location',
                '--prefix',
                user_plugin_dir(),
            ]
        if versions:
            for idx, pkg in enumerate(pkg_list):
                pk_ver = pkg + '==' + versions[idx]
                pkg_list[idx] = pk_ver

        process.setArguments(cmd + list(pkg_list))
        if self._output_widget and self._queue:
            self._output_widget.clear()

        process.start()
        return process

    def uninstall(
        self,
        pkg_list: Sequence[str],
        installer: Optional[InstallerTypes] = None,
        channels: Sequence[str] = ("conda-forge",),
    ):
        installer = installer or self._installer_type
        self._queue.insert(
            0,
            (
                tuple(pkg_list),
                lambda: self._uninstall(pkg_list, installer, channels),
            ),
        )
        self._handle_action()

    def _uninstall(
        self,
        pkg_list: Sequence[str],
        installer: Optional[InstallerTypes] = None,
        channels: Sequence[str] = ("conda-forge",),
    ):
        installer = installer or self._installer_type
        if installer != "pip":
            args = [
                'remove',
                '-y',
                '--prefix',
                self._conda_env_path,
            ]

            for channel in channels:
                args.extend(["-c", channel])
        else:
            args = ['-m', 'pip', 'uninstall', '-y']

        process = self._create_process(installer)
        process.setArguments(args + list(pkg_list))
        if self._output_widget and self._queue:
            self._output_widget.clear()

        process.start()

        pm2 = PluginManager.instance()

        for pkg in pkg_list:
            if pkg in pm2:
                pm2.unregister(pkg)
            else:
                plugin_manager.unregister(pkg)

        return process

    def cancel(
        self,
        pkg_list: Sequence[str] = None,
    ):
        if pkg_list is None:
            for _, process in self._processes.items():
                process.terminate()

            self._processes = {}
        else:
            with contextlib.suppress(KeyError):
                process = self._processes.pop(tuple(pkg_list))
                process.terminate()

    @staticmethod
    def _is_installed_with_conda():
        """
        Check if conda was used to install qt and napari.
        """
        from qtpy import QT_VERSION

        from napari._version import version_tuple

        parts = [str(part) for part in version_tuple[:3]]
        napari_version_string = f"napari-{'.'.join(parts)}-"
        qt_version_string = f"qt-{QT_VERSION}-"
        conda_meta_path = Path(sys.prefix) / "conda-meta"
        if conda_meta_path.is_dir():
            for file in conda_meta_path.iterdir():
                fname = file.parts[-1]
                if (
                    fname.startswith(napari_version_string)
                    or fname.startswith(qt_version_string)
                ) and fname.endswith(".json"):
                    return True
        return False


def is_conda_package(pkg):
    """Determines if plugin was installed through conda.

    Returns
    -------
    bool: True if a conda package, False if not
    """
    conda_meta_dir = Path(sys.prefix) / 'conda-meta'
    try:
        for fname in conda_meta_dir.iterdir():
            if fname.suffix == '.json':
                *name, _, _ = fname.name.rsplit('-')
                name = "-".join(name)
                if pkg == name:
                    return True
    except FileNotFoundError:
        return False

    return False


class PluginListItem(QFrame):
    """An entry in the plugin dialog.  This will include the package name, summary,
    author, source, version, and buttons to update, install/uinstall, etc."""

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
        enabled: bool = True,
        installed: bool = False,
        npe_version=1,
        versions_conda: list = [],
        versions_pypi: list = [],
    ):
        super().__init__(parent)
        self.url = url
        self._versions = {}
        self._versions['Conda'] = versions_conda
        self._versions['PyPi'] = versions_pypi
        self.setup_ui(enabled)
        self.plugin_name.setText(package_name)

        self._populate_version_dropdown('PyPi')
        self.package_name.setText(version)
        if summary:
            self.summary.setText(summary)
        if author:
            self.package_author.setText(author)
        self.package_author.setWordWrap(True)
        self.cancel_btn.setVisible(False)

        self._handle_npe2_plugin(npe_version)

        if installed:
            if is_conda_package(package_name):
                self.source.setText('Conda')
            self.enabled_checkbox.show()
            self.action_button.setText(trans._("Uninstall"))
            self.action_button.setObjectName("remove_button")
            self.info_choice_wdg.hide()
            self.source_choice_dropdown.hide()
            self.install_info_button.addWidget(self.info_widget)
            self.info_widget.show()
        else:
            self.enabled_checkbox.hide()
            self.action_button.setText(trans._("Install"))
            self.action_button.setObjectName("install_button")
            self.info_widget.hide()
            self.install_info_button.addWidget(self.info_choice_wdg)
            self.info_choice_wdg.show()
            self.source_choice_dropdown.show()

    def _handle_npe2_plugin(self, npe_version):
        if npe_version in (None, 1):
            return
        opacity = 0.4 if npe_version == 'shim' else 1
        lbl = trans._('npe1 (adapted)') if npe_version == 'shim' else 'npe2'
        npe2_icon = QLabel(self)
        icon = QColoredSVGIcon.from_resources('logo_silhouette')
        npe2_icon.setPixmap(
            icon.colored(color='#33F0FF', opacity=opacity).pixmap(20, 20)
        )
        self.row1.insertWidget(2, QLabel(lbl))
        self.row1.insertWidget(2, npe2_icon)

    def _get_dialog(self) -> QDialog:
        p = self.parent()
        while not isinstance(p, QDialog) and p.parent():
            p = p.parent()
        return p

    def set_busy(
        self, text: str, action_name: str = None, update: bool = False
    ):
        """Updates status text and what buttons are visible when any button is pushed.

        Parameters
        ----------
        text: str
            The new string to be displayed as the status.
        action_name: str
            The action of the button pressed.
        update: bool
            States whether this install is an update or not.

        """
        self.item_status.setText(text)
        if action_name == 'install' and update is True:
            self.cancel_btn.setVisible(True)
            self.action_button.setVisible(False)
            self.old_action = 'update'
        elif (
            action_name == 'uninstall' or action_name == 'install'
        ) and update is False:
            self.action_button.setVisible(False)
            self.cancel_btn.setVisible(True)
            self.old_action = action_name
        elif action_name == 'cancel':
            self.action_button.setVisible(True)
            self.action_button.setDisabled(False)
            self.cancel_btn.setVisible(False)

    def setup_ui(self, enabled=True):
        """Define the layout of the PluginListItem"""

        self.v_lay = QVBoxLayout(self)
        self.v_lay.setContentsMargins(-1, 6, -1, 6)
        self.v_lay.setSpacing(0)
        self.row1 = QHBoxLayout()
        self.row1.setSpacing(6)
        self.enabled_checkbox = QCheckBox(self)
        self.enabled_checkbox.setChecked(enabled)
        self.enabled_checkbox.stateChanged.connect(self._on_enabled_checkbox)
        self.enabled_checkbox.setToolTip(trans._("enable/disable"))
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
        self.plugin_name = QPushButton(self)
        # Do not want to highlight on hover unless there is a website.
        if self.url:
            self.plugin_name.setObjectName('plugin_name_web')
        else:
            self.plugin_name.setObjectName('plugin_name')

        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.plugin_name.sizePolicy().hasHeightForWidth()
        )
        self.plugin_name.setSizePolicy(sizePolicy)
        font15 = QFont()
        font15.setPointSize(15)
        font15.setUnderline(True)
        self.plugin_name.setFont(font15)
        self.row1.addWidget(self.plugin_name)

        icon = QColoredSVGIcon.from_resources("warning")
        self.warning_tooltip = QtToolTipLabel(self)
        # TODO: This color should come from the theme but the theme needs
        # to provide the right color. Default warning should be orange, not
        # red. Code example:
        # theme_name = get_settings().appearance.theme
        # napari.utils.theme.get_theme(theme_name, as_dict=False).warning.as_hex()
        self.warning_tooltip.setPixmap(
            icon.colored(color="#E3B617").pixmap(15, 15)
        )
        self.warning_tooltip.setVisible(False)
        self.row1.addWidget(self.warning_tooltip)

        self.item_status = QLabel(self)
        self.item_status.setObjectName("small_italic_text")
        self.item_status.setSizePolicy(sizePolicy)
        self.row1.addWidget(self.item_status)
        self.row1.addStretch()
        self.v_lay.addLayout(self.row1)

        self.row2 = QHBoxLayout()
        self.error_indicator = QPushButton()
        self.error_indicator.setObjectName("warning_icon")
        self.error_indicator.setCursor(Qt.CursorShape.PointingHandCursor)
        self.error_indicator.hide()
        self.row2.addWidget(
            self.error_indicator, alignment=Qt.AlignmentFlag.AlignTop
        )
        self.row2.setContentsMargins(-1, 4, 0, -1)
        self.row2.setSpacing(5)
        self.summary = QElidingLabel(parent=self)
        self.summary.setObjectName('summary_text')
        self.summary.setWordWrap(True)
        dlg_width = self.parent().parent().sizeHint().width()
        self.summary.setFixedWidth(int(dlg_width * 1.5))

        sizePolicy = QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed
        )

        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.summary.setSizePolicy(sizePolicy)
        self.row2.addWidget(self.summary, alignment=Qt.AlignmentFlag.AlignTop)

        self.package_author = QElidingLabel(self)
        self.package_author.setObjectName('author_text')
        self.package_author.setWordWrap(True)
        sizePolicy = QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.package_author.sizePolicy().hasHeightForWidth()
        )
        self.package_author.setSizePolicy(sizePolicy)
        self.row2.addWidget(
            self.package_author, alignment=Qt.AlignmentFlag.AlignTop
        )

        self.update_btn = QPushButton('Update', self)
        self.update_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.update_btn.setObjectName("install_button")
        self.update_btn.setVisible(False)

        self.row2.addWidget(
            self.update_btn, alignment=Qt.AlignmentFlag.AlignTop
        )

        self.info_choice_wdg = QWidget(self)
        self.info_choice_wdg.setObjectName('install_choice')
        self.install_info_button = superQCollapsible("Installation Info")
        self.install_info_button.layout().setContentsMargins(0, 0, 0, 0)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.install_info_button.sizePolicy().hasHeightForWidth()
        )
        self.install_info_button.setSizePolicy(sizePolicy)
        self.install_info_button.setObjectName("install_info_button")
        self.source_choice_text = QLabel('Source ')
        self.version_choice_text = QLabel('Version ')
        self.source_choice_dropdown = QComboBox()
        self.source_choice_dropdown.addItem('PyPi')
        self.source_choice_dropdown.addItem('Conda')
        self.source_choice_dropdown.currentTextChanged.connect(
            self._populate_version_dropdown
        )
        self.source_choice_dropdown.hide()
        self.version_choice_dropdown = QComboBox()
        self.install_info_button.content().layout().setContentsMargins(
            0, 0, 0, 0
        )
        self.version_choice_dropdown.setFixedWidth(80)
        self.row2.addWidget(
            self.install_info_button, alignment=Qt.AlignmentFlag.AlignTop
        )

        info_layout = QGridLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.addWidget(self.source_choice_text, 0, 0)
        info_layout.addWidget(self.source_choice_dropdown, 1, 0)
        info_layout.addWidget(self.version_choice_text, 0, 1)
        info_layout.addWidget(self.version_choice_dropdown, 1, 1)
        self.info_choice_wdg.setLayout(info_layout)
        self.info_choice_wdg.setObjectName("install_choice_widget")
        self.row2.addWidget(
            self.info_choice_wdg, alignment=Qt.AlignmentFlag.AlignTop
        )
        self.info_choice_wdg.hide()

        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.cancel_btn.setObjectName("remove_button")
        self.row2.addWidget(
            self.cancel_btn, alignment=Qt.AlignmentFlag.AlignTop
        )

        self.action_button = QPushButton(self)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.action_button.sizePolicy().hasHeightForWidth()
        )

        self.action_button.setSizePolicy(sizePolicy)
        self.row2.addWidget(
            self.action_button, alignment=Qt.AlignmentFlag.AlignTop
        )

        self.v_lay.addLayout(self.row2)

        self.info_widget = QWidget(self)

        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.info_widget.setSizePolicy(sizePolicy)
        self.info_widget.setObjectName("info_widget")
        info_layout = QGridLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        self.version_text = QLabel('Version:')
        self.version_text.setStyleSheet('margin: 0px; padding: 0px;')
        self.package_name = QLabel()
        self.package_name.setStyleSheet('margin: 0px; padding: 0px;')
        self.source_text = QLabel('Source:')
        self.source_text.setStyleSheet('margin-right: 7px; padding: 0px;')
        self.source = QLabel('PyPi')
        self.source.setStyleSheet('margin-right: 7px; padding: 0px;')

        info_layout.addWidget(self.source_text, 0, 0)
        info_layout.addWidget(self.source, 1, 0)
        info_layout.addWidget(self.version_text, 0, 1)
        info_layout.addWidget(self.package_name, 1, 1)

        self.info_widget.setLayout(info_layout)

    def _populate_version_dropdown(self, e):
        """Display the versions available after selecting a source: pypi or conda."""

        versions = self._versions[e][::-1]
        self.version_choice_dropdown.clear()
        if len(versions) > 0:
            for version in versions:
                self.version_choice_dropdown.addItem(version)

    def _on_enabled_checkbox(self, state: int):
        """Called with `state` when checkbox is clicked."""
        enabled = bool(state)
        plugin_name = self.plugin_name.text()
        pm2 = PluginManager.instance()
        if plugin_name in pm2:
            pm2.enable(plugin_name) if state else pm2.disable(plugin_name)
            return

        for npe1_name, _, distname in plugin_manager.iter_available():
            if distname and (normalized_name(distname) == plugin_name):
                plugin_manager.set_blocked(npe1_name, not enabled)

    def show_warning(self, message: str = ""):
        """Show warning icon and tooltip."""
        self.warning_tooltip.setVisible(bool(message))
        self.warning_tooltip.setToolTip(message)


class QPluginList(QListWidget):
    def __init__(self, parent: QWidget, installer: InstallerQueue):
        super().__init__(parent)
        self.installer = installer
        self.setSortingEnabled(True)
        self._remove_list = []

    def _count_visible(self) -> int:
        """Return the number of visible items.

        Visible items are the result of the normal `count` method minus
        any hidden items.
        """
        hidden = 0
        count = self.count()
        for i in range(count):
            item = self.item(i)
            hidden += item.isHidden()

        return count - hidden

    @Slot(PackageMetadata)
    def addItem(
        self,
        project_info_versions: Tuple[PackageMetadata, list, list],
        installed=False,
        plugin_name=None,
        enabled=True,
        npe_version=None,
    ):
        project_info = project_info_versions[0]
        versions_pypi = project_info_versions[1]
        versions_conda = project_info_versions[2]

        pkg_name = project_info.name
        # don't add duplicates
        if (
            self.findItems(pkg_name, Qt.MatchFlag.MatchFixedString)
            and not plugin_name
        ):
            return

        # including summary here for sake of filtering below.
        searchable_text = f"{pkg_name} {project_info.summary}"
        item = QListWidgetItem(searchable_text, self)
        item.version = project_info.version
        super().addItem(item)
        widg = PluginListItem(
            package_name=pkg_name,
            version=project_info.version,
            url=project_info.home_page,
            summary=project_info.summary,
            author=project_info.author,
            license=project_info.license,
            parent=self,
            plugin_name=plugin_name,
            enabled=enabled,
            installed=installed,
            npe_version=npe_version,
            versions_conda=versions_conda,
            versions_pypi=versions_pypi,
        )
        item.widget = widg
        item.npe_version = npe_version
        action_name = 'uninstall' if installed else 'install'
        item.setSizeHint(widg.sizeHint())
        self.setItemWidget(item, widg)

        if project_info.home_page:
            import webbrowser

            widg.plugin_name.clicked.connect(
                partial(webbrowser.open, project_info.home_page)
            )

        widg.action_button.clicked.connect(
            partial(self.handle_action, item, pkg_name, action_name,
                version=widg.version_choice_dropdown.currentText(),
                installer_choice=widg.source_choice_dropdown.currentText(),
                update=False,
            )
        )

        widg.update_btn.clicked.connect(
            partial(
                self.handle_action,
                item,
                pkg_name,
                InstallerActions.INSTALL,
                update=True,
            )
        )
        widg.cancel_btn.clicked.connect(
            partial(
                self.handle_action, item, pkg_name, InstallerActions.CANCEL
            )
        )

        item.setSizeHint(widg.sizeHint())
        self.setItemWidget(item, widg)
        widg.install_info_button.setDuration(0)
        widg.install_info_button.toggled.connect(
            lambda: self._resize_pluginlistitem(item)
        )

    def _resize_pluginlistitem(self, item):
        """Resize the plugin list item, especially after toggling QCollapsible."""
        height = item.widget.height()
        if item.widget.install_info_button.isExpanded():
            item.widget.setFixedHeight(int(height * 1.8))
        else:
            item.widget.setFixedHeight(int(height / 1.8))
        item.setSizeHint(item.widget.size())

    def handle_action(
        self,
        item: QListWidgetItem,
        pkg_name: str,
        action_name: InstallerActions,
        update: bool = False,
        version: str = None,
        installer_choice: Optional[str] = None,
    ):
        """Determine which action is called (install, uninstall, update, cancel).
        Update buttons appropriately and run the action."""
        # TODO: 'tool' should be configurable per item, depending on dropdown
        tool = (
            InstallerTools.CONDA
            if running_as_constructor_app()
            else InstallerTools.PIP
        )

        widget = item.widget
        item.setText(f"0-{item.text()}")
        self._remove_list.append((pkg_name, item))
        self._warn_dialog = None
        # TODO: NPE version unknown before installing
        if item.npe_version != 1 and action_name == InstallerActions.UNINSTALL:
            # show warning pop up dialog
            message = trans._(
                'When installing/uninstalling npe2 plugins, you must '
                'restart napari for UI changes to take effect.'
            )
            self._warn_dialog = WarnPopup(text=message)

            delta_x = 75
            global_point = widget.action_button.mapToGlobal(
                widget.action_button.rect().topLeft()
            )
            global_point = QPoint(global_point.x() - delta_x, global_point.y())
            self._warn_dialog.move(global_point)

        if action_name == InstallerActions.INSTALL:
            if update:
                if hasattr(item, 'latest_version'):
                    pkg_name += f"=={item.latest_version}"

                widget.set_busy(trans._("updating..."), action_name, update)
                widget.action_button.setDisabled(True)
            else:
                widget.set_busy(trans._("installing..."), update)
                widget.set_busy(trans._("installing..."), action_name, update)

            job_id = self.installer.install(
                tool=tool,
                pkgs=[pkg_name],
                # origins="TODO",
            )
            if self._warn_dialog:
                self._warn_dialog.exec_()
            self.scrollToTop()
        elif action_name == InstallerActions.UNINSTALL:
            widget.set_busy(trans._("uninstalling..."), action_name, False)
            widget.update_btn.setDisabled(True)
            job_id = self.installer.uninstall(
                tool=tool,
                pkgs=[pkg_name],
                # origins="TODO",
            )
            widget.setProperty("current_job_id", job_id)
            if self._warn_dialog:
                self._warn_dialog.exec_()
            self.scrollToTop()
        elif action_name == InstallerActions.CANCEL:
            widget.set_busy(trans._("cancelling..."), action_name, False)
            try:
                job_id = widget.property("current_job_id")
                self.installer.cancel(job_id)
            finally:
                widget.setProperty("current_job_id", None)

    @Slot(PackageMetadata, bool)
    def tag_outdated(self, project_info: PackageMetadata, is_available: bool):
        """Determines if an installed plugin is up to date with the latest version.
        If it is not, the latest version will be displayed on the update button."""
        if not is_available:
            return

        for item in self.findItems(
            project_info.name, Qt.MatchFlag.MatchStartsWith
        ):
            current = item.version
            latest = project_info.version
            if parse_version(current) >= parse_version(latest):
                continue
            if hasattr(item, 'outdated'):
                # already tagged it
                continue

            item.outdated = True
            item.latest_version = latest
            widg = self.itemWidget(item)
            widg.update_btn.setVisible(True)
            widg.update_btn.setText(
                trans._("update (v{latest})", latest=latest)
            )

    def tag_unavailable(self, project_info: PackageMetadata):
        """
        Tag list items as unavailable for install with conda-forge.

        This will disable the item and the install button and add a warning
        icon with a hover tooltip.
        """
        for item in self.findItems(
            project_info.name, Qt.MatchFlag.MatchStartsWith
        ):
            widget = self.itemWidget(item)
            widget.show_warning(
                trans._(
                    "Plugin not yet available for installation within the bundle application"
                )
            )
            widget.setObjectName("unavailable")
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            widget.action_button.setEnabled(False)
            widget.warning_tooltip.setVisible(True)

    def filter(self, text: str):
        """Filter items to those containing `text`."""
        if text:
            # PySide has some issues, so we compare using id
            # See: https://bugreports.qt.io/browse/PYSIDE-74
            shown = [
                id(it)
                for it in self.findItems(text, Qt.MatchFlag.MatchContains)
            ]
            for i in range(self.count()):
                item = self.item(i)
                item.setHidden(id(item) not in shown)
        else:
            for i in range(self.count()):
                item = self.item(i)
                item.setHidden(False)


class RefreshState(Enum):
    REFRESHING = auto()
    OUTDATED = auto()
    DONE = auto()


class QtPluginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.refresh_state = RefreshState.DONE
        self.already_installed = set()

        self.installer = InstallerQueue()
        self.setWindowTitle(trans._('Plugin Manager'))
        self.setup_ui()
        self.installer.set_output_widget(self.stdout_text)
        self.installer.started.connect(self._on_installer_start)
        self.installer.finished.connect(self._on_installer_done)
        self.refresh()

    def _on_installer_start(self):
        """Updates dialog buttons and status when installing a plugin."""
        self.cancel_all_btn.setVisible(True)
        self.working_indicator.show()
        self.process_success_indicator.hide()
        self.process_error_indicator.hide()
        self.close_btn.setDisabled(True)

    def _on_installer_done(self, exit_code):
        """Updates buttons and status when plugin is done installing."""
        self.working_indicator.hide()
        if exit_code:
            self.process_error_indicator.show()
        else:
            self.process_success_indicator.show()
        self.cancel_all_btn.setVisible(False)
        self.close_btn.setDisabled(False)
        self.refresh()

    def closeEvent(self, event):
        if self.close_btn.isEnabled():
            super().closeEvent(event)

        event.ignore()

    def refresh(self):
        if self.refresh_state != RefreshState.DONE:
            self.refresh_state = RefreshState.OUTDATED
            return
        self.refresh_state = RefreshState.REFRESHING
        self.installed_list.clear()
        self.available_list.clear()

        # fetch installed
        from npe2 import PluginManager

        from napari.plugins import plugin_manager

        self.already_installed = set()

        def _add_to_installed(distname, enabled, npe_version=1):
            norm_name = normalized_name(distname or '')
            if distname:
                try:
                    meta = metadata(distname)
                except PackageNotFoundError:
                    self.refresh_state = RefreshState.OUTDATED
                    return  # a race condition has occurred and the package is uninstalled by another thread
                if len(meta) == 0:
                    # will not add builtins.
                    return
                self.already_installed.add(norm_name)
            else:
                meta = {}

            self.installed_list.addItem(
                (
                    PackageMetadata(
                        metadata_version="1.0",
                        name=norm_name,
                        version=meta.get('version', ''),
                        summary=meta.get('summary', ''),
                        home_page=meta.get('url', ''),
                        author=meta.get('author', ''),
                        license=meta.get('license', ''),
                    ),
                    [],
                    [],
                ),
                installed=True,
                enabled=enabled,
                npe_version=npe_version,
            )

        pm2 = PluginManager.instance()
        discovered = pm2.discover()
        for manifest in pm2.iter_manifests():
            distname = normalized_name(manifest.name or '')
            if distname in self.already_installed or distname == 'napari':
                continue
            enabled = not pm2.is_disabled(manifest.name)
            # if it's an Npe1 adaptor, call it v1
            npev = 'shim' if manifest.npe1_shim else 2
            _add_to_installed(distname, enabled, npe_version=npev)

        plugin_manager.discover()  # since they might not be loaded yet
        for plugin_name, _, distname in plugin_manager.iter_available():
            # not showing these in the plugin dialog
            if plugin_name in ('napari_plugin_engine',):
                continue
            if normalized_name(distname or '') in self.already_installed:
                continue
            _add_to_installed(
                distname, not plugin_manager.is_blocked(plugin_name)
            )

        self.installed_label.setText(
            trans._(
                "Installed Plugins ({amount})",
                amount=len(self.already_installed),
            )
        )

        # fetch available plugins
        settings = get_settings()
        use_hub = (
            running_as_bundled_app()
            or running_as_constructor_app()
            or settings.plugins.plugin_api.name == "napari_hub"
        )
        if use_hub:
            conda_forge = running_as_constructor_app()
            self.worker = create_worker(
                iter_hub_plugin_info, conda_forge=conda_forge
            )
        else:
            self.worker = create_worker(iter_napari_plugin_info)

        self.worker.yielded.connect(self._handle_yield)
        self.worker.finished.connect(self.working_indicator.hide)
        self.worker.finished.connect(self._update_count_in_label)
        self.worker.finished.connect(self._end_refresh)
        self.worker.start()
        if discovered:
            message = trans._(
                'When installing/uninstalling npe2 plugins, '
                'you must restart napari for UI changes to take effect.'
            )
            self._warn_dialog = WarnPopup(text=message)
            global_point = self.process_error_indicator.mapToGlobal(
                self.process_error_indicator.rect().topLeft()
            )
            global_point = QPoint(global_point.x(), global_point.y() - 75)
            self._warn_dialog.move(global_point)
            self._warn_dialog.exec_()

    def setup_ui(self):
        """Defines the layout for the PluginDialog."""

        self.resize(950, 640)
        vlay_1 = QVBoxLayout(self)
        self.h_splitter = QSplitter(self)
        vlay_1.addWidget(self.h_splitter)
        self.h_splitter.setOrientation(Qt.Orientation.Horizontal)
        self.v_splitter = QSplitter(self.h_splitter)
        self.v_splitter.setOrientation(Qt.Orientation.Vertical)
        self.v_splitter.setMinimumWidth(500)

        installed = QWidget(self.v_splitter)
        lay = QVBoxLayout(installed)
        lay.setContentsMargins(0, 2, 0, 2)
        self.installed_label = QLabel(trans._("Installed Plugins"))
        self.packages_filter = QLineEdit()
        self.packages_filter.setPlaceholderText(trans._("filter..."))
        self.packages_filter.setMaximumWidth(350)
        self.packages_filter.setClearButtonEnabled(True)
        mid_layout = QVBoxLayout()
        mid_layout.addWidget(self.packages_filter)
        mid_layout.addWidget(self.installed_label)
        lay.addLayout(mid_layout)

        self.installed_list = QPluginList(installed, self.installer)
        self.packages_filter.textChanged.connect(self.installed_list.filter)
        lay.addWidget(self.installed_list)

        uninstalled = QWidget(self.v_splitter)
        lay = QVBoxLayout(uninstalled)
        lay.setContentsMargins(0, 2, 0, 2)
        self.avail_label = QLabel(trans._("Available Plugins"))
        mid_layout = QHBoxLayout()
        mid_layout.addWidget(self.avail_label)
        mid_layout.addStretch()
        lay.addLayout(mid_layout)
        self.available_list = QPluginList(uninstalled, self.installer)
        self.packages_filter.textChanged.connect(self.available_list.filter)
        lay.addWidget(self.available_list)

        self.stdout_text = QTextEdit(self.v_splitter)
        self.stdout_text.setReadOnly(True)
        self.stdout_text.setObjectName("plugin_manager_process_status")
        self.stdout_text.hide()

        buttonBox = QHBoxLayout()
        self.working_indicator = QLabel(trans._("loading ..."), self)
        sp = self.working_indicator.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.working_indicator.setSizePolicy(sp)
        self.process_error_indicator = QLabel(self)
        self.process_error_indicator.setObjectName("error_label")
        self.process_error_indicator.hide()
        self.process_success_indicator = QLabel(self)
        self.process_success_indicator.setObjectName("success_label")
        self.process_success_indicator.hide()
        load_gif = str(Path(napari.resources.__file__).parent / "loading.gif")
        mov = QMovie(load_gif)
        mov.setScaledSize(QSize(18, 18))
        self.working_indicator.setMovie(mov)
        mov.start()

        visibility_direct_entry = not running_as_constructor_app()
        self.direct_entry_edit = QLineEdit(self)
        self.direct_entry_edit.installEventFilter(self)
        self.direct_entry_edit.setPlaceholderText(
            trans._('install by name/url, or drop file...')
        )
        self.direct_entry_edit.setVisible(visibility_direct_entry)
        self.direct_entry_btn = QPushButton(trans._("Install"), self)
        self.direct_entry_btn.setVisible(visibility_direct_entry)
        self.direct_entry_btn.clicked.connect(self._install_packages)

        self.show_status_btn = QPushButton(trans._("Show Status"), self)
        self.show_status_btn.setFixedWidth(100)

        self.cancel_all_btn = QPushButton(trans._("cancel all actions"), self)
        self.cancel_all_btn.setObjectName("remove_button")
        self.cancel_all_btn.setVisible(False)
        self.cancel_all_btn.clicked.connect(self.installer.cancel)

        self.close_btn = QPushButton(trans._("Close"), self)
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setObjectName("close_button")
        buttonBox.addWidget(self.show_status_btn)
        buttonBox.addWidget(self.working_indicator)
        buttonBox.addWidget(self.direct_entry_edit)
        buttonBox.addWidget(self.direct_entry_btn)
        if not visibility_direct_entry:
            buttonBox.addStretch()
        buttonBox.addWidget(self.process_success_indicator)
        buttonBox.addWidget(self.process_error_indicator)
        buttonBox.addSpacing(20)
        buttonBox.addWidget(self.cancel_all_btn)
        buttonBox.addSpacing(20)
        buttonBox.addWidget(self.close_btn)
        buttonBox.setContentsMargins(0, 0, 4, 0)
        vlay_1.addLayout(buttonBox)

        self.show_status_btn.setCheckable(True)
        self.show_status_btn.setChecked(False)
        self.show_status_btn.toggled.connect(self._toggle_status)

        self.v_splitter.setStretchFactor(1, 2)
        self.h_splitter.setStretchFactor(0, 2)

        self.packages_filter.setFocus()

    def _update_count_in_label(self):
        """Counts all available but not installed plugins. Updates value."""

        count = self.available_list.count()
        self.avail_label.setText(
            trans._("Available Plugins ({count})", count=count)
        )

    def _end_refresh(self):
        refresh_state = self.refresh_state
        self.refresh_state = RefreshState.DONE
        if refresh_state == RefreshState.OUTDATED:
            self.refresh()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.DragEnter:
            # we need to accept this event explicitly to be able
            # to receive QDropEvents!
            event.accept()
        if event.type() == QEvent.Drop:
            md = event.mimeData()
            if md.hasUrls():
                files = [url.toLocalFile() for url in md.urls()]
                self.direct_entry_edit.setText(files[0])
                return True
        return super().eventFilter(watched, event)

    def _toggle_status(self, show):
        if show:
            self.show_status_btn.setText(trans._("Hide Status"))
            self.stdout_text.show()
        else:
            self.show_status_btn.setText(trans._("Show Status"))
            self.stdout_text.hide()

    def _install_packages(
        self,
        packages: Sequence[str] = (),
        versions: Optional[Sequence[str]] = None,
        installer: Optional[InstallerTypes] = None,
    ):
        if not packages:
            _packages = self.direct_entry_edit.text()
            packages = (
                [_packages] if os.path.exists(_packages) else _packages.split()
            )
            self.direct_entry_edit.clear()
        if packages:
            self.installer.install(
                packages, versions=versions, installer=installer
            )

    def _handle_yield(
        self, data: Tuple[PackageMetadata, bool, List[str], List[str]]
    ):
        """Output from a worker process.  Includes information about the plugin,
        including available versions on conda and pypi."""

        project_info, is_available, versions_pypi, versions_conda = data
        if project_info.name in self.already_installed:
            self.installed_list.tag_outdated(project_info, is_available)
        else:
            self.available_list.addItem(
                (project_info, versions_pypi, versions_conda)
            )
            if not is_available:
                self.available_list.tag_unavailable(project_info)

        self.filter()

    def filter(self, text: Optional[str] = None) -> None:
        """Filter by text or set current text as filter."""
        if text is None:
            text = self.packages_filter.text()
        else:
            self.packages_filter.setText(text)

        self.installed_list.filter(text)
        self.available_list.filter(text)


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    w = QtPluginDialog()
    w.show()
    app.exec_()
