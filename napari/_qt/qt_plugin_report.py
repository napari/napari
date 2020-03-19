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
    text_box : qtpy.QtWidgets.QTextEdit
        The text area where traceback information will be shown.
    plugin_combo : qtpy.QtWidgets.QComboBox
        The dropdown menu used to select the current plugin
    github_button : qtpy.QtWidgets.QPushButton
        A button that, when pressed, will open an issue at the current plugin's
        github issue tracker, prepopulated with a formatted traceback.  Button
        is only visible if a github URL is detected in the package metadata for
        the current plugin.
    clipboard_button : qtpy.QtWidgets.QPushButton
        A button that, when pressed, copies the current traceback information
        to the clipboard.  (HTML tags are removed in the copied text.)
    only_errors_checkbox : qtpy.QtWidgets.QCheckBox
        When checked, only plugins that have raised errors during this session
        will be visible in the ``plugin_combo``.
    plugin_meta : qtpy.QtWidgets.QLabel
        A label that will show available plugin metadata (such as home page).
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

        self.text_box = QTextEdit()
        self.text_box.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.text_box.setMinimumWidth(360)

        # Create plugin dropdown menu
        # the union here is only needed if we want to also have plugins
        # wITHOUT errors in the dropdown
        plugin_names = set(PLUGIN_ERRORS).union(
            set(plugin_manager._name2plugin)
        )
        listview = QListView()
        self.plugin_combo = QComboBox()
        self.plugin_combo.setView(listview)
        self.plugin_combo.addItem(self.NULL_OPTION)
        self.plugin_combo.addItems(list(sorted(plugin_names)))
        self.plugin_combo.activated[str].connect(self.set_plugin)
        self.plugin_combo.setCurrentText(self.NULL_OPTION)

        # create github button (gets connected in self.set_plugin)
        self.github_button = QPushButton('Open issue at github', self)
        self.github_button.setToolTip(
            "Open webrowser and submit this traceback\n"
            "to the developer's github issue tracker"
        )
        self.github_button.hide()

        # create copy to clipboard button
        self.clipboard_button = QPushButton()
        self.clipboard_button.hide()
        self.clipboard_button.setObjectName("QtCopyToClipboardButton")
        self.clipboard_button.setToolTip("Copy traceback to clipboard")
        self.clipboard_button.clicked.connect(self.copyToClipboard)

        self.only_errors_checkbox = QCheckBox(
            'only show plugins with errors', self
        )
        self.only_errors_checkbox.stateChanged.connect(self._on_errbox_change)
        self.only_errors_checkbox.setChecked(True)

        self.plugin_meta = QLabel('', parent=self)
        self.plugin_meta.setObjectName("pluginInfo")
        self.plugin_meta.setTextFormat(Qt.RichText)
        self.plugin_meta.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.plugin_meta.setOpenExternalLinks(True)
        self.plugin_meta.setAlignment(Qt.AlignRight)

        # make layout
        row_1_layout = QHBoxLayout()
        row_1_layout.setContentsMargins(11, 5, 10, 0)
        row_1_layout.addWidget(self.only_errors_checkbox)
        row_1_layout.addStretch(1)
        row_1_layout.addWidget(self.plugin_meta)

        row_2_layout = QHBoxLayout()
        row_2_layout.setContentsMargins(11, 5, 10, 0)
        row_2_layout.addWidget(self.plugin_combo)
        row_2_layout.addStretch(1)
        row_2_layout.addWidget(self.github_button)
        row_2_layout.addWidget(self.clipboard_button)
        row_2_layout.setSpacing(5)

        self.layout.addLayout(row_1_layout)
        self.layout.addLayout(row_2_layout)
        self.layout.addWidget(self.text_box, 1)
        self.setMinimumWidth(750)
        self.setMinimumHeight(600)

        if initial_plugin:
            self.set_plugin(initial_plugin)

    def set_plugin(self, plugin: str) -> None:
        """Set the current plugin shown in the dropdown and text area."""
        self.plugin_combo.setCurrentText(plugin)
        self.github_button.hide()
        self.clipboard_button.hide()
        try:
            self.github_button.clicked.disconnect()
        except RuntimeError:
            pass
        if plugin in PLUGIN_ERRORS:
            err_string = format_exceptions(plugin, as_html=True)
            self.text_box.setHtml(err_string)
            self.clipboard_button.show()
            self._set_meta(plugin)
        else:
            self.plugin_meta.setText('')
            self.text_box.setText(f'No errors recorded for plugin "{plugin}"')

    def _set_meta(self, plugin: str):
        err0 = PLUGIN_ERRORS[plugin][0]
        meta = fetch_module_metadata(err0.plugin_module)
        meta_text = ''
        if not meta:
            self.plugin_meta.setText(meta_text)
            return

        url = meta.get('url')
        if url:
            meta_text += (
                '<span style="color:#999;">plugin home page:&nbsp;&nbsp;</span>'
                f'<a href="{url}" style="color:#999">{url}</a>'
            )
        self.plugin_meta.setText(meta_text)
        if 'github.com' in meta.get('url', ''):

            def onclick():
                err = format_exceptions(plugin, as_html=False)
                err = (
                    "<!--Provide detail on the error here-->\n\n\n\n"
                    "<details>\n<summary>Traceback from napari</summary>"
                    f"\n\n```\n{err}\n```\n</details>"
                )
                url = f'{meta.get("url")}/issues/new?&body={err}'
                webbrowser.open(url, new=2)

            self.github_button.clicked.connect(onclick)
            self.github_button.show()

    def _on_errbox_change(self, state: bool):
        """Handle click event on the only_errors_checkbox."""
        view = self.plugin_combo.view()
        _shown = 1
        for row in range(1, self.plugin_combo.count()):
            plugin_name = self.plugin_combo.itemText(row)
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
        plugin = self.plugin_combo.currentText()
        err_string = format_exceptions(plugin, as_html=False)
        cb = QGuiApplication.clipboard()
        cb.setText(err_string)
