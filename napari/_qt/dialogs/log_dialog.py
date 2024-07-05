from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
)

from napari import _LOG_STREAM
from napari._qt.dialogs.qt_about import QtCopyToClipboardButton
from napari.utils.translations import trans


class LogDialog(QDialog):
    def __init__(
        self,
        parent,
    ) -> None:
        super().__init__(parent._qt_window)

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

        self.setObjectName('LogDialog')
        self.setWindowTitle(trans._('napari log'))
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.exec_()

    def _on_level_change(self, level):
        self.log_text_box.setText(_LOG_STREAM.formatted_at_level(level))
