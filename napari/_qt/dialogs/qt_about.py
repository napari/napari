from qtpy import QtGui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from napari.utils import citation_text, sys_info
from napari.utils.translations import trans


class QtAbout(QDialog):
    """Qt dialog window for displaying 'About napari' information.

    Parameters
    ----------
    parent : QWidget, optional
        Parent of the dialog, to correctly inherit and apply theme.
        Default is None.

    Attributes
    ----------
    citationCopyButton : napari._qt.qt_about.QtCopyToClipboardButton
        Button to copy citation information to the clipboard.
    citationTextBox : qtpy.QtWidgets.QTextEdit
        Text box containing napari citation information.
    citation_layout : qtpy.QtWidgets.QHBoxLayout
        Layout widget for napari citation information.
    infoCopyButton : napari._qt.qt_about.QtCopyToClipboardButton
        Button to copy napari version information to the clipboard.
    info_layout : qtpy.QtWidgets.QHBoxLayout
        Layout widget for napari version information.
    infoTextBox : qtpy.QtWidgets.QTextEdit
        Text box containing napari version information.
    layout : qtpy.QtWidgets.QVBoxLayout
        Layout widget for the entire 'About napari' dialog.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.layout = QVBoxLayout()

        # Description
        title_label = QLabel(
            trans._(
                "<b>napari: a multi-dimensional image viewer for python</b>"
            )
        )
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

        self.infoTextBox.setText(sys_info(as_html=True))
        self.infoTextBox.setMinimumSize(
            int(self.infoTextBox.document().size().width() + 19),
            int(min(self.infoTextBox.document().size().height() + 10, 500)),
        )

        self.layout.addWidget(QLabel(trans._('<b>citation information:</b>')))
        self.citationTextBox = QTextEdit(citation_text)
        self.citationTextBox.setFixedHeight(64)
        self.citationCopyButton = QtCopyToClipboardButton(self.citationTextBox)
        self.citation_layout = QHBoxLayout()
        self.citation_layout.addWidget(self.citationTextBox, 1)
        self.citation_layout.addWidget(
            self.citationCopyButton, 0, Qt.AlignmentFlag.AlignTop
        )
        self.layout.addLayout(self.citation_layout)

        self.setLayout(self.layout)

    @staticmethod
    def showAbout(parent=None):
        """Display the 'About napari' dialog box.

        Parameters
        ----------
        parent : QWidget, optional
            Parent of the dialog, to correctly inherit and apply theme.
            Default is None.
        """
        d = QtAbout(parent)
        d.setObjectName('QtAbout')
        d.setWindowTitle(trans._('About'))
        d.setWindowModality(Qt.WindowModality.ApplicationModal)
        d.exec_()


class QtCopyToClipboardButton(QPushButton):
    """Button to copy text box information to the clipboard.

    Parameters
    ----------
    text_edit : qtpy.QtWidgets.QTextEdit
        The text box contents linked to copy to clipboard button.

    Attributes
    ----------
    text_edit : qtpy.QtWidgets.QTextEdit
        The text box contents linked to copy to clipboard button.
    """

    def __init__(self, text_edit) -> None:
        super().__init__()
        self.setObjectName("QtCopyToClipboardButton")
        self.text_edit = text_edit
        self.setToolTip(trans._("Copy to clipboard"))
        self.clicked.connect(self.copyToClipboard)

    def copyToClipboard(self):
        """Copy text to the clipboard."""
        cb = QtGui.QGuiApplication.clipboard()
        cb.setText(str(self.text_edit.toPlainText()))
