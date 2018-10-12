from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout


class QtWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.widget = QWidget()
        self.setCentralWidget(self.widget)

        self.widget.setLayout(QHBoxLayout())
        self.statusBar().showMessage('Ready')

    def add_viewer(self, viewer):
        self.widget.layout().addWidget(viewer._qt)
