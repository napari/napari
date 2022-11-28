"""
Method to get napari style in magicgui based windows
====================================================

Example how to embed magicgui widget in dialog to inherit style
from main napari window.
"""


from qtpy.QtWidgets import QDialog, QWidget, QVBoxLayout, QPushButton, QGridLayout, QLabel, QSpinBox

from magicgui import magicgui

import napari
from napari.qt import get_stylesheet
from napari.settings import get_settings

# The magicgui widget shown by selecting the 'Show widget' button of MyWidget
@magicgui
def sample_add(a: int, b: int) -> int:
    return a + b

def change_style():
    sample_add.native.setStyleSheet(get_stylesheet(get_settings().appearance.theme))


get_settings().appearance.events.theme.connect(change_style)
change_style()


class MyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.first_input = QSpinBox()
        self.second_input = QSpinBox()
        self.btn = QPushButton('Add')
        layout = QGridLayout()
        layout.addWidget(QLabel("first input"), 0, 0)
        layout.addWidget(self.first_input, 0, 1)
        layout.addWidget(QLabel("second input"), 1, 0)
        layout.addWidget(self.second_input, 1, 1)
        layout.addWidget(self.btn, 2, 0, 1, 2)
        self.setLayout(layout)
        self.btn.clicked.connect(self.run)

    def run(self):
        print('run', self.first_input.value() + self.second_input.value())
        self.close()

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.btn1 = QPushButton('Show dialog')
        self.btn1.clicked.connect(self.show_dialog)
        self.btn2 = QPushButton('Show widget')
        self.btn2.clicked.connect(self.show_widget)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.btn1)
        self.layout.addWidget(self.btn2)
        self.setLayout(self.layout)

    def show_dialog(self):
        dialog = MyDialog(self)
        dialog.exec_()

    def show_widget(self):
        sample_add.show()



viewer = napari.Viewer()

widget = MyWidget()
viewer.window.add_dock_widget(widget, area='right')
napari.run()
