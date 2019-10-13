import skimage
import vispy
import scipy
import numpy

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QDialog, QFrame

import napari


class QtAbout(QWidget):
    def __init__(self, parent):
        super(QtAbout, self).__init__(parent)

        self.layout = QVBoxLayout()

        # Description
        title_label = QLabel(
            "<b>napari</b>: a fast n-dimensional image viewer"
        )
        title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.layout.addWidget(title_label)

        # Horizontal Line Break
        self.hline_break1 = QFrame()
        self.hline_break1.setFrameShape(QFrame.HLine)
        self.hline_break1.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(self.hline_break1)

        # Versions
        versions_label = QLabel(
            "napari, "
            + napari.__version__
            + "\n"
            + "Qt, "
            + QtCore.__version__
            + "\n"
            + "NumPy, "
            + numpy.__version__
            + "\n"
            + "SciPy, "
            + scipy.__version__
            + "\n"
            + "VisPy, "
            + vispy.__version__
            + "\n"
            + "scikit-image, "
            + skimage.__version__
            + "\n"
        )
        versions_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.layout.addWidget(versions_label)

        # Horizontal Line Break
        self.hline_break1 = QFrame()
        self.hline_break1.setFrameShape(QFrame.HLine)
        self.hline_break1.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(self.hline_break1)

        sys_info_lines = "\n".join(
            [
                vispy.sys_info().split("\n")[index]
                for index in [0, 1, 3, -4, -3]
            ]
        )
        sys_info_label = QLabel(sys_info_lines)
        sys_info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.layout.addWidget(sys_info_label)

        self.setLayout(self.layout)

    @staticmethod
    def showAbout(qt_viewer):
        d = QDialog()
        d.setObjectName('QtAbout')
        d.setStyleSheet(qt_viewer.styleSheet())
        QtAbout(d)
        d.setWindowTitle('About')
        d.setWindowModality(Qt.ApplicationModal)
        d.exec_()
