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


def is_conda_package(pkg):
    conda_meta_dir = Path(sys.prefix) / 'conda-meta'
    for fname in conda_meta_dir.iterdir():
        if fname.suffix == '.json':
            *name, _, _ = fname.name.rsplit('-')
            name = "-".join(name)
            if pkg == name:
                return True

    return False


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
        enabled: bool = True,
        installed: bool = False,
        npe_version=1,
        versions_conda: list = [],
        versions_pypi: list = [],
    ):
        super().__init__(parent)

        self._versions = {}
        self._versions['Conda'] = versions_conda
        self._versions['PyPi'] = versions_pypi
        self.setup_ui(enabled)
        self.plugin_name.setText(package_name)

        self._populate_version_dropdown('PyPi')
        self.package_name.setText(version)
        if summary:
            self.summary.setText(summary)
        self.package_author.setText(author)
        self.cancel_btn.setVisible(False)

        self.help_button.setText(trans._("Website"))
        self.help_button.setObjectName("help_button")
        self._handle_npe2_plugin(npe_version)

        if installed:
            if is_conda_package(package_name):
                self.source.setText('Conda')
            self.enabled_checkbox.show()
            self.action_button.setText(trans._("Uninstall"))
            self.action_button.setObjectName("remove_button")
            self.info_choice_wdg.hide()
            self.source_choice_dropdown.hide()
            self.install_info_button.show()
            self.latest_version_text.show()
        else:
            self.enabled_checkbox.hide()
            self.action_button.setText(trans._("Install"))
            self.action_button.setObjectName("install_button")
            self.info_choice_wdg.show()
            self.install_info_button.hide()
            self.update_btn.setVisible(False)
            self.latest_version_text.hide()
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

    def set_busy(self, text: str, update: bool = False):
        self.item_status.setText(text)
        self.cancel_btn.setVisible(True)
        if not update:
            self.action_button.setVisible(False)
        else:
            self.update_btn.setVisible(False)

    def setup_ui(self, enabled=True):
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
        self.plugin_name = QLabel(self)
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
        self.summary = QElidingLabel(parent=self)
        self.summary.setObjectName('summary_text')
        self.summary.setWordWrap(True)
        dlg_width = self.parent().parent().sizeHint().width()
        self.summary.setFixedWidth(int(dlg_width) * 1.5)

        sizePolicy = QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed
        )

        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.summary.setSizePolicy(sizePolicy)
        self.row2.addWidget(self.summary, alignment=Qt.AlignmentFlag.AlignTop)

        self.package_author = QLabel(self)
        self.package_author.setObjectName('author_text')
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

        self.help_button = QPushButton(self)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.help_button.sizePolicy().hasHeightForWidth()
        )
        self.help_button.setSizePolicy(sizePolicy)
        self.row2.addWidget(
            self.help_button, alignment=Qt.AlignmentFlag.AlignTop
        )

        self.info_choice_wdg = QWidget(self)
        self.info_choice_wdg.setObjectName('install_choice')
        self.install_info_button = superQCollapsible("Installation Info")
        self.install_info_button.layout().setContentsMargins(0, 0, 0, 0)
        sizePolicy = QSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.MinimumExpanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.install_info_button.sizePolicy().hasHeightForWidth()
        )
        self.install_info_button.setSizePolicy(sizePolicy)
        self.install_info_button.setObjectName("install_info_button")
        self.source_choice_text = QLabel('Install via ')
        self.version_choice_text = QLabel('Version ')
        self.source_choice_dropdown = QComboBox()
        self.source_choice_dropdown.addItem('PyPi')
        self.source_choice_dropdown.addItem('Conda')
        self.source_choice_dropdown.currentTextChanged.connect(
            self._populate_version_dropdown
        )
        self.source_choice_dropdown.hide()
        self.version_choice_dropdown = QComboBox()

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

        self.cancel_btn = QPushButton("cancel", self)
        self.cancel_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.cancel_btn.setObjectName("remove_button")
        self.row2.addWidget(
            self.cancel_btn, alignment=Qt.AlignmentFlag.AlignTop
        )

        self.update_wdg = QWidget()
        update_layout = QVBoxLayout()
        self.update_btn = QPushButton('Update', self)
        self.update_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.update_btn.setObjectName("install_button")
        self.latest_version_text = QLabel()
        self.latest_version_text.setObjectName('latest_version_text')
        update_layout.addWidget(self.update_btn)
        update_layout.addWidget(self.latest_version_text)
        update_layout.setContentsMargins(0, 0, 0, 0)

        self.update_wdg.setLayout(update_layout)
        self.row2.addWidget(
            self.update_wdg, alignment=Qt.AlignmentFlag.AlignTop
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
        self.version_text.setStyleSheet('margin: 0px; padding: 4px;')
        self.package_name = QLabel()
        self.package_name.setStyleSheet('margin: 0px; padding: 4px;')
        self.source_text = QLabel('Source:')
        self.source_text.setStyleSheet('margin: 0px; padding: 0px;')
        self.source = QLabel('PyPi')
        self.source.setStyleSheet('margin: 0px; padding: 4px;')

        info_layout.addWidget(self.version_text, 0, 0)
        info_layout.addWidget(self.package_name, 1, 0)
        info_layout.addWidget(self.source_text, 0, 1)
        info_layout.addWidget(self.source, 1, 1)

        self.info_widget.setLayout(info_layout)
        self.install_info_button.addWidget(self.info_widget)

    def _populate_version_dropdown(self, e):
        versions = self._versions[e]

        self.version_choice_dropdown.clear()

        if len(versions) > 0:
            for version in versions:
                self.version_choice_dropdown.addItem(version)

            self.latest_version_text.setText(f'to {versions[0]}')

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

            widg.help_button.clicked.connect(
                partial(webbrowser.open, project_info.home_page)
            )
        else:
            widg.help_button.setVisible(False)

        widg.action_button.clicked.connect(
            partial(self.handle_action, item, pkg_name, action_name,
                version=widg.version_choice_dropdown.currentText(),
                installer_choice=widg.source_choice_dropdown.currentText()
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
        height = int(item.widget.height())
        if item.widget.install_info_button.isExpanded():
            item.widget.setFixedHeight(height * 1.5)
        else:
            item.widget.setFixedHeight(height / 1.5)
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

                widget.set_busy(trans._("updating..."), update)
                widget.action_button.setDisabled(True)
            else:
                widget.set_busy(trans._("installing..."), update)

            job_id = self.installer.install(
                tool=tool,
                pkgs=[pkg_name],
                # origins="TODO",
            )
            if self._warn_dialog:
                self._warn_dialog.exec_()
            self.scrollToTop()
        elif action_name == InstallerActions.UNINSTALL:
            widget.set_busy(trans._("uninstalling..."), update)
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
            widget.set_busy(trans._("cancelling..."), update)
            try:
                job_id = widget.property("current_job_id")
                self.installer.cancel(job_id)
            finally:
                widget.setProperty("current_job_id", None)

    @Slot(PackageMetadata, bool)
    def tag_outdated(self, project_info: PackageMetadata, is_available: bool):
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
        self.cancel_all_btn.setVisible(True)
        self.working_indicator.show()
        self.process_success_indicator.hide()
        self.process_error_indicator.hide()
        self.close_btn.setDisabled(True)

    def _on_installer_done(self, exit_code):
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
        self.resize(1080, 640)
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

    def _handle_yield(self, data: Tuple[PackageMetadata, bool]):
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
