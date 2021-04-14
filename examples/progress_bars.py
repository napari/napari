from PyQt5 import QtWidgets
import napari
from time import sleep
from napari.utils.progress import progress
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget


def arbitrary_steps():
    """We can manually control updating the value of the progress bar. 
    If we try to set a value that exceeds the total, the progress bar
    becomes indeterminate.
    """
    pbr = progress(total=3)
    sleep(3)
    pbr.update(1, "Step 1 Complete")    

    sleep(1)
    pbr.update(2, "Step 2 Complete")

    sleep(2)
    pbr.update(3, "Step 3 Done!")

    sleep(1)
    pbr.update(4, "Total Exceeded...")


def indeterminate():
    """By passing a total of 0, we can have an indeterminate progress bar
    """
    pbr = progress(total=0)


def iterable():
    """progress can be used as a wrapper around an iterable object
    """
    for i in progress(range(10)):
        sleep(0.1)


viewer = napari.Viewer()
button_layout = QVBoxLayout()

steps_btn = QPushButton("Arbitrary Steps")
steps_btn.clicked.connect(arbitrary_steps)
button_layout.addWidget(steps_btn)

indeterminate_btn = QPushButton("Indeterminate")
indeterminate_btn.clicked.connect(indeterminate)
button_layout.addWidget(indeterminate_btn)

iterable_btn = QPushButton("Iterable")
iterable_btn.clicked.connect(iterable)
button_layout.addWidget(iterable_btn)

pbar_widget = QWidget()
pbar_widget.setLayout(button_layout)

viewer.window.add_dock_widget(pbar_widget)


napari.run()

