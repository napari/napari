"""Provides a QtPluginErrReporter that allows the user report plugin errors.
"""
from typing import Optional

from qtpy.QtGui import QGuiApplication
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari.plugins.manager import NapariPluginManager

from ..plugins.exceptions import (
    PLUGIN_ERRORS,
    format_exceptions,
    fetch_module_metadata,
)
from ..plugins import plugin_manager as napari_plugin_manager


class QtPluginErrReporter(QDialog):
    NULL_OPTION = 'select plugin... '

    def __init__(
        self,
        plugin_manager: Optional[NapariPluginManager] = None,
        parent: Optional[QWidget] = None,
        *,
        initial_plugin: Optional[str] = None,
        errors_only: bool = True,
    ) -> None:

        plugin_manager = plugin_manager or napari_plugin_manager
        super().__init__(parent)
        self.plugin_manager = plugin_manager

        self.setWindowTitle('Recorded Plugin Exceptions')
        self.setWindowModality(Qt.NonModal)
        self.layout = QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.layout)

        # stacked keybindings widgets
        self.textEditBox = QTextEdit()
        self.textEditBox.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.textEditBox.setMinimumWidth(360)

        # plugin selection
        self.pluginComboBox = QComboBox()
        listview = QListView()
        self.pluginComboBox.setView(listview)

        self.pluginComboBox.addItem(self.NULL_OPTION)

        plugin_names = set(PLUGIN_ERRORS).union(
            set(plugin_manager._name2plugin)
        )
        self.pluginComboBox.addItems(list(sorted(plugin_names)))
        self.pluginComboBox.activated[str].connect(self.set_plugin)
        self.pluginComboBox.setCurrentText(self.NULL_OPTION)

        self.sendToDeveloperButton = QPushButton('open issue at plugin', self)
        self.sendToDeveloperButton.hide()
        self.copyButton = QPushButton()
        self.copyButton.hide()
        self.copyButton.setObjectName("QtCopyToClipboardButton")
        self.setToolTip("Copy to clipboard")
        self.copyButton.clicked.connect(self.copyToClipboard)

        top_row_layout = QHBoxLayout()
        top_row_layout.setContentsMargins(11, 5, 10, 0)
        top_row_layout.addWidget(self.pluginComboBox)
        top_row_layout.addStretch(1)
        top_row_layout.addWidget(self.sendToDeveloperButton)
        top_row_layout.addWidget(self.copyButton)
        top_row_layout.setSpacing(5)

        row2_layout = QHBoxLayout()
        row2_layout.setContentsMargins(11, 0, 10, 5)
        row2_layout.setSpacing(6)
        self.onlyErrorsCheckbox = QCheckBox(self)
        self.onlyErrorsCheckbox.stateChanged.connect(self._on_errbox_change)
        self.onlyErrorsCheckbox.setChecked(True)
        row2_layout.addWidget(self.onlyErrorsCheckbox)
        row2_layout.addWidget(QLabel('only show plugins with errors'))
        row2_layout.addStretch(1)
        self.layout.addLayout(row2_layout)
        self.layout.addLayout(top_row_layout)
        self.layout.addWidget(self.textEditBox, 1)
        self.setMinimumWidth(750)
        self.setMinimumHeight(600)

    def set_plugin(self, plugin: str) -> None:
        self.sendToDeveloperButton.hide()
        self.copyButton.hide()
        try:
            self.sendToDeveloperButton.clicked.disconnect()
        except RuntimeError:
            pass
        if plugin in PLUGIN_ERRORS:
            err_string = format_exceptions(plugin, as_html=True)
            self.textEditBox.setHtml(err_string)
            self.copyButton.show()

            err0 = PLUGIN_ERRORS[plugin][0]
            meta = fetch_module_metadata(err0.plugin_module)
            if meta and 'github.com' in meta.get('url', ''):

                def onclick():
                    import webbrowser

                    err = format_exceptions(plugin, as_html=False)
                    err = (
                        "\n\n\n\n<details>\n<summary>Traceback</summary>"
                        f"\n\n```\n{err}\n```\n</details>"
                    )
                    url = f'{meta.get("url")}/issues/new?&body={err}'
                    webbrowser.open(url, new=2)

                self.sendToDeveloperButton.clicked.connect(onclick)
                self.sendToDeveloperButton.show()
        else:
            self.textEditBox.setText('')

    def _on_errbox_change(self, state: bool):
        view = self.pluginComboBox.view()
        _shown = 1
        for row in range(1, self.pluginComboBox.count()):
            plugin_name = self.pluginComboBox.itemText(row)
            has_err = plugin_name in PLUGIN_ERRORS
            if state and not has_err:
                view.setRowHidden(row, True)
            else:
                view.setRowHidden(row, False)
                _shown += 1
        view.setMinimumHeight(_shown * 18)

    def copyToClipboard(self) -> None:
        plugin = self.pluginComboBox.currentText()
        err_string = format_exceptions(plugin, as_html=False)
        cb = QGuiApplication.clipboard()
        cb.setText(err_string)
