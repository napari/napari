from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QDialog, QFrame


class AboutPage(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        self.layout = QVBoxLayout()

        # Horizontal Line Break
        self.hline_break1 = QFrame()
        self.hline_break1.setFrameShape(QFrame.HLine)
        self.hline_break1.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(self.hline_break1)

        # Description
        self.layout.addWidget(
            QLabel("napari - a Qt- and VisPy-based ndarray visualization tool")
        )

        self.setLayout(self.layout)

    @staticmethod
    def showAbout(self):
        d = QDialog()
        d.setGeometry(150, 150, 350, 400)
        d.setFixedSize(350, 400)
        AboutPage(d)
        d.setWindowTitle("About")
        d.setWindowModality(Qt.ApplicationModal)
        d.exec_()
