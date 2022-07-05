"""
magicgui in dialog
==================

Example how to embed magicgui widget in dialog to inherit style
from main napari window.
"""

from typing import Callable

from qtpy.QtWidgets import QDialog, QWidget, QVBoxLayout, QPushButton

from magicgui import magicgui

import napari

def sample_add(a: int, b: int) -> int:
    return a + b

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
        self.btn = QPushButton('Click me')
        self.btn.clicked.connect(self.click)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.btn)
        self.setLayout(self.layout)


    def click(self):
        dialog = MguiDialog(sample_add, self)
        dialog.exec_()



viewer = napari.Viewer()

widget = MyWidget()
viewer.window.add_dock_widget(widget, area='right')
napari.run()