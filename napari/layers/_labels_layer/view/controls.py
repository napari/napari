from PyQt5.QtWidgets import QWidget


class QtImageControls(QWidget):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer

        self.setMouseTracking(True)
