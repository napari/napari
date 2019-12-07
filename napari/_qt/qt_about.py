import sys
import platform
import skimage
import vispy
import scipy
import numpy
import dask

from qtpy import QtCore, API_NAME, PYSIDE_VERSION, PYQT_VERSION
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QVBoxLayout, QTextEdit, QDialog, QLabel

import napari


class QtAbout(QDialog):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        # Description
        title_label = QLabel(
            "<b>napari: a multi-dimensional image viewer for python</b>"
        )
        title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.layout.addWidget(title_label)

        # Add information
        self.infoTextBox = QTextEdit()
        self.infoTextBox.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.infoTextBox.setLineWrapMode(QTextEdit.NoWrap)
        self.layout.addWidget(self.infoTextBox, 1)

        if API_NAME == 'PySide2':
            API_VERSION = PYSIDE_VERSION
        elif API_NAME == 'PyQt5':
            API_VERSION = PYQT_VERSION
        else:
            API_VERSION = ''
        sys_version = sys.version.replace('\n', ' ')

        versions = (
            f"<b>napari</b>: {napari.__version__} <br>"
            f"<b>Platform</b>: {platform.platform()} <br>"
            f"<b>Python</b>: {sys_version} <br>"
            f"<b>{API_NAME}</b>: {API_VERSION} <br>"
            f"<b>Qt</b>: {QtCore.__version__} <br>"
            f"<b>VisPy</b>: {vispy.__version__} <br>"
            f"<b>NumPy</b>: {numpy.__version__} <br>"
            f"<b>SciPy</b>: {scipy.__version__} <br>"
            f"<b>scikit-image</b>: {skimage.__version__} <br>"
            f"<b>Dask</b>: {dask.__version__} <br>"
        )

        sys_info_text = "<br>".join(
            [vispy.sys_info().split("\n")[index] for index in [-4, -3]]
        )

        text = f'{versions} <br> {sys_info_text} <br>'
        self.infoTextBox.setText(text)

        self.layout.addWidget(QLabel('<b>citation information:</b>'))

        citation_text = (
            'napari contributors (2019). napari: a '
            'multi-dimensional image viewer for python. '
            'doi:10.5281/zenodo.3555620'
        )
        self.citationTextBox = QTextEdit(citation_text)
        self.citationTextBox.setFixedHeight(64)
        self.layout.addWidget(self.citationTextBox)

        self.setLayout(self.layout)

    @staticmethod
    def showAbout(qt_viewer):
        d = QtAbout()
        d.setStyleSheet(qt_viewer.styleSheet())
        d.setWindowTitle('About')
        d.setWindowModality(Qt.ApplicationModal)
        d.exec_()
