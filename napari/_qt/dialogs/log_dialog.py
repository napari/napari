from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
)

from napari._qt.dialogs.qt_about import QtCopyToClipboardButton
from napari.utils._logging import get_filtered_logs_html
from napari.utils.translations import trans


class LogDialog(QDialog):
    def __init__(
        self,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.layout = QVBoxLayout()

        # Description
        title_label = QLabel(trans._('napari log'))
        title_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.layout.addWidget(title_label)

        # level selection
        self.level_selection = QComboBox()
        self.level_selection.addItems(
            ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        )
        self.layout.addWidget(self.level_selection)
        self.level_selection.currentTextChanged.connect(self._on_level_change)

        # text filter
        self.text_filter = QLineEdit()
        self.text_filter.setPlaceholderText('text filter')
        self.layout.addWidget(self.text_filter)
        self.text_filter.textChanged.connect(self._on_filter_change)

        # log text box
        self.log_text_box = QTextEdit()
        self.log_text_box.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.log_text_box.setLineWrapMode(QTextEdit.NoWrap)
        # Add text copy button
        self.infoCopyButton = QtCopyToClipboardButton(self.log_text_box)
        self.info_layout = QHBoxLayout()
        self.info_layout.addWidget(self.log_text_box, 1)
        self.info_layout.addWidget(
            self.infoCopyButton, 0, Qt.AlignmentFlag.AlignTop
        )
        self.info_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.addLayout(self.info_layout)

        self._on_level_change(self.level_selection.currentText())

        self.setLayout(self.layout)

    @staticmethod
    def showLog(parent=None):
        d = LogDialog(parent)
        d.setObjectName('LogDialog')
        d.setWindowTitle(trans._('napari Log'))
        d.setWindowModality(Qt.WindowModality.ApplicationModal)
        d.exec_()

    def _on_level_change(self, level):
        logs = get_filtered_logs_html(level, self.text_filter.text())
        self.log_text_box.setHtml(logs)

    def _on_filter_change(self, text_filter):
        logs = get_filtered_logs_html(
            self.level_selection.currentText(), text_filter
        )
        self.log_text_box.setHtml(logs)
