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
    pbr.set_description("Step 1 Complete")
    pbr.update(1)    

    sleep(1)
    pbr.set_description("Step 2 Complete")
    pbr.update(1)

    sleep(2)
    pbr.set_description("Step 3 Done!")
    pbr.update(1)

    pbr.close()

def indeterminate():
    """By passing a total of 0, we can have an indeterminate progress bar
    """
    pbr = progress(total=0)


def iterable():
    """progress can be used as a wrapper around an iterable object
    """
    my_animals = ["cat", "dog", "wolf", "turtle"]
    pbr = progress(my_animals)
    for animal in pbr:
        pbr.set_description(f"{animal}")
        sleep(0.4)

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

