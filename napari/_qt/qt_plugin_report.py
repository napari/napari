"""Provides a QtPluginErrReporter that allows the user report plugin errors.
"""
import webbrowser
from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication
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

from ..plugins import plugin_manager
from ..plugins.exceptions import (
    PLUGIN_ERRORS,
    fetch_module_metadata,
    format_exceptions,
)


class QtPluginErrReporter(QDialog):
    """Dialog that allows users to review and report PluginError tracebacks.

    Parameters
    ----------
    parent : QWidget, optional
        Optional parent widget for this widget.
    initial_plugin : str, optional
        If provided, errors from ``initial_plugin`` will be shown when the
        dialog is created, by default None

    Attributes
    ----------
    textEditBox : qtpy.QtWidgets.QTextEdit
        The text area where traceback information will be shown.
    pluginComboBox : qtpy.QtWidgets.QComboBox
        The dropdown menu used to select the current plugin
    openAtGithubButton : qtpy.QtWidgets.QPushButton
        A button that, when pressed, will open an issue at the current plugin's
        github issue tracker, prepopulated with a formatted traceback.  Button
        is only visible if a github URL is detected in the package metadata for
        the current plugin.
    copyButton : qtpy.QtWidgets.QPushButton
        A button that, when pressed, copies the current traceback information
        to the clipboard.  (HTML tags are removed in the copied text.)
    onlyErrorsCheckbox : qtpy.QtWidgets.QCheckBox
        When checked, only plugins that have raised errors during this session
        will be visible in the ``pluginComboBox``.
    """

    NULL_OPTION = 'select plugin... '

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        initial_plugin: Optional[str] = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle('Recorded Plugin Exceptions')
        self.setWindowModality(Qt.NonModal)
        self.layout = QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.layout)

        self.textEditBox = QTextEdit()
        self.textEditBox.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.textEditBox.setMinimumWidth(360)

        # Create plugin dropdown menu
        # the union here is only needed if we want to also have plugins
        # wITHOUT errors in the dropdown
        plugin_names = set(PLUGIN_ERRORS).union(
            set(plugin_manager._name2plugin)
        )
        listview = QListView()
        self.pluginComboBox = QComboBox()
        self.pluginComboBox.setView(listview)
        self.pluginComboBox.addItem(self.NULL_OPTION)
        self.pluginComboBox.addItems(list(sorted(plugin_names)))
        self.pluginComboBox.activated[str].connect(self.set_plugin)
        self.pluginComboBox.setCurrentText(self.NULL_OPTION)

        # create github button (gets connected in self.set_plugin)
        self.openAtGithubButton = QPushButton('Open issue at github', self)
        self.openAtGithubButton.setToolTip(
            "Open webrowser and submit this traceback\n"
            "to the developer's github issue tracker"
        )
        self.openAtGithubButton.hide()

        # create copy to clipboard button
        self.copyButton = QPushButton()
        self.copyButton.hide()
        self.copyButton.setObjectName("QtCopyToClipboardButton")
        self.setToolTip("Copy traceback to clipboard")
        self.copyButton.clicked.connect(self.copyToClipboard)

        # make layout
        top_row_layout = QHBoxLayout()
        top_row_layout.setContentsMargins(11, 5, 10, 0)
        top_row_layout.addWidget(self.pluginComboBox)
        top_row_layout.addStretch(1)
        top_row_layout.addWidget(self.openAtGithubButton)
        top_row_layout.addWidget(self.copyButton)
        top_row_layout.setSpacing(5)
        row2_layout = QHBoxLayout()
        row2_layout.setContentsMargins(11, 0, 10, 5)
        row2_layout.setSpacing(2)
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

        if initial_plugin:
            self.set_plugin(initial_plugin)

    def set_plugin(self, plugin: str) -> None:
        """Set the current plugin shown in the dropdown and text area."""
        self.pluginComboBox.setCurrentText(plugin)
        self.openAtGithubButton.hide()
        self.copyButton.hide()
        try:
            self.openAtGithubButton.clicked.disconnect()
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
                    err = format_exceptions(plugin, as_html=False)
                    err = (
                        "<!--Provide detail on the error here-->\n\n\n\n"
                        "<details>\n<summary>Traceback from napari</summary>"
                        f"\n\n```\n{err}\n```\n</details>"
                    )
                    url = f'{meta.get("url")}/issues/new?&body={err}'
                    webbrowser.open(url, new=2)

                self.openAtGithubButton.clicked.connect(onclick)
                self.openAtGithubButton.show()
        else:
            self.textEditBox.setText('')

    def _on_errbox_change(self, state: bool):
        """Handle click event on the onlyErrorsCheckbox."""
        view = self.pluginComboBox.view()
        _shown = 1
        for row in range(1, self.pluginComboBox.count()):
            plugin_name = self.pluginComboBox.itemText(row)
            has_err = plugin_name in PLUGIN_ERRORS
            # if the box is checked, hide plugins that have no errors
            if state and not has_err:
                view.setRowHidden(row, True)
            else:
                view.setRowHidden(row, False)
                _shown += 1
        view.setMinimumHeight(_shown * 18)

    def copyToClipboard(self) -> None:
        """Copy current plugin traceback info to clipboard as plain text."""
        plugin = self.pluginComboBox.currentText()
        err_string = format_exceptions(plugin, as_html=False)
        cb = QGuiApplication.clipboard()
        cb.setText(err_string)
