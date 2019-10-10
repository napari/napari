import skimage
import vispy
import scipy
import numpy

import qtpy
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QDialog, QFrame

import napari


class AboutPage(QWidget):
    def __init__(self, parent):
        super(AboutPage, self).__init__(parent)

        self.layout = QVBoxLayout()

        # Description
        self.layout.addWidget(
            QLabel("<b>napari</b>: a fast n-dimensional image viewer")
        )

        # Horizontal Line Break
        self.hline_break1 = QFrame()
        self.hline_break1.setFrameShape(QFrame.HLine)
        self.hline_break1.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(self.hline_break1)

        # Versions
        self.layout.addWidget(QLabel("napari, " + napari.__version__))
        self.layout.addWidget(QLabel("Qt, " + qtpy.__version__))
        self.layout.addWidget(QLabel("NumPy, " + numpy.__version__))
        self.layout.addWidget(QLabel("SciPy, " + scipy.__version__))
        self.layout.addWidget(QLabel("VisPy, " + vispy.__version__))
        self.layout.addWidget(QLabel("scikit-image, " + skimage.__version__))

        self.setLayout(self.layout)

    @staticmethod
    def showAbout():
        d = QDialog()
        d.setGeometry(150, 150, 350, 400)
        d.setFixedSize(350, 400)
        AboutPage(d)
        d.setWindowTitle("About")
        d.setWindowModality(Qt.ApplicationModal)
        d.exec_()
