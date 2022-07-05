"""
Method to get napari style in magicgui based windows
====================================================

Example how to embed magicgui widget in dialog to inherit style
from main napari window.
"""

from typing import Callable

from qtpy.QtWidgets import QDialog, QWidget, QVBoxLayout, QPushButton

from magicgui import magicgui

import napari
from napari.qt import get_stylesheet
from napari.settings import get_settings

def sample_add(a: int, b: int) -> int:
    return a + b

@magicgui
def sample_add2(a: int, b: int) -> int:
    return a + b

def change_style():
    sample_add2.native.setStyleSheet(get_stylesheet(get_settings().appearance.theme))


get_settings().appearance.events.theme.connect(change_style)
change_style()


class MguiDialog(QDialog):
    def __init__(self, fun: Callable, parent=None):
        super().__init__(parent)
        self.mgui_widget = magicgui(fun)  # close of dialog will destroy widget
        layout = QVBoxLayout()
        layout.addWidget(self.mgui_widget.native)
        self.setLayout(layout)
        self.mgui_widget.called.connect(self.run)

    def run(self, value):
        print('run', value)
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
        dialog = MguiDialog(sample_add, self)
        dialog.exec_()

    def show_widget(self):
        sample_add2.show()



viewer = napari.Viewer()

widget = MyWidget()
viewer.window.add_dock_widget(widget, area='right')
napari.run()
