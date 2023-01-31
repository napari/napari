"""
Clipboard
=========

Copy screenshot of the canvas or the whole viewer to clipboard.

.. tags:: gui
"""

from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
from skimage import data

import napari

# create the viewer with an image
viewer = napari.view_image(data.moon())

class Grabber(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.copy_canvas_btn = QPushButton("Copy Canvas to Clipboard", self)
        self.copy_canvas_btn.setToolTip("Copy screenshot of the canvas to clipboard.")
        self.copy_viewer_btn = QPushButton("Copy Viewer to Clipboard", self)
        self.copy_viewer_btn.setToolTip("Copy screenshot of the entire viewer to clipboard.")

        layout = QVBoxLayout(self)
        layout.addWidget(self.copy_canvas_btn)
        layout.addWidget(self.copy_viewer_btn)


def create_grabber_widget():
    """Create widget"""
    widget = Grabber()

    # connect buttons
    widget.copy_canvas_btn.clicked.connect(lambda: viewer.window.qt_viewer.clipboard())
    widget.copy_viewer_btn.clicked.connect(lambda: viewer.window.clipboard())
    return widget


widget = create_grabber_widget()
viewer.window.add_dock_widget(widget)

if __name__ == '__main__':
    napari.run()
