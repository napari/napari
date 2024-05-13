from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
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
        parent=None,
    ) -> None:
        super().__init__(parent._qt_window)

        self.layout = QVBoxLayout()

        # Description
        title_label = QLabel(trans._('napari log'))
        title_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.layout.addWidget(title_label)

        # Add information
        self.infoTextBox = QTextEdit()
        self.infoTextBox.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.infoTextBox.setLineWrapMode(QTextEdit.NoWrap)
        # Add text copy button
        self.infoCopyButton = QtCopyToClipboardButton(self.infoTextBox)
        self.info_layout = QHBoxLayout()
        self.info_layout.addWidget(self.infoTextBox, 1)
        self.info_layout.addWidget(
            self.infoCopyButton, 0, Qt.AlignmentFlag.AlignTop
        )
        self.info_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.addLayout(self.info_layout)

        self.infoTextBox.setText(str(_LOG_STREAM))
        self.infoTextBox.setMinimumSize(
            int(self.infoTextBox.document().size().width() + 19),
            int(min(self.infoTextBox.document().size().height() + 10, 500)),
        )

        self.setLayout(self.layout)

        self.setObjectName('LogDialog')
        self.setWindowTitle(trans._('napari log'))
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.exec_()
