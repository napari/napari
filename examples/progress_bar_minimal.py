"""This file provides minimal working examples of progress bars in
the napari viewer.
"""

import napari
from time import sleep
from napari.utils.progress import progress
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

def iterable():
    """progress can be used as a wrapper around an iterable object
    """
    my_animals = ["cat", "dog", "wolf", "turtle"]
    
    # progress provides a context manager we can use for automatic
    # teardown of our widget once iteration is complete
    with progress(my_animals) as pbr:
        for animal in pbr:
            # using a context manager also allows us to manipulate
            # the progress object e.g. by setting a description
            pbr.set_description(f"{animal}")
            sleep(0.4)

def indeterminate():
    """By passing a total of 0, we can have an indeterminate progress bar
    """
    pbr = progress(total=0)

    for i in range(int(1e4)):
        pbr.set_description(f"Processing {i}")

    # if not using progress in a context manager or through a for loop
    # we have to manually close it
    pbr.close()

def arbitrary_steps():
    """We can manually control updating the value of the progress bar.
    """
    pbr = progress(total=3)
    sleep(3)
    pbr.set_description("Step 1 Complete")
    # manually updating the progress bar by 1
    pbr.update(1)    

    sleep(1)
    pbr.set_description("Step 2 Complete")
    pbr.update(1)

    sleep(2)
    pbr.set_description("Step 3 Done!")
    pbr.update(1)

    # sleeping so we can see full completion
    sleep(1)

    # again, it's important we manually close the progress bar
    pbr.close()

viewer = napari.Viewer()
button_layout = QVBoxLayout()

iterable_btn = QPushButton("Iterable")
iterable_btn.clicked.connect(iterable)
button_layout.addWidget(iterable_btn)

indeterminate_btn = QPushButton("Indeterminate")
indeterminate_btn.clicked.connect(indeterminate)
button_layout.addWidget(indeterminate_btn)

steps_btn = QPushButton("Arbitrary Steps")
steps_btn.clicked.connect(arbitrary_steps)
button_layout.addWidget(steps_btn)

pbar_widget = QWidget()
pbar_widget.setLayout(button_layout)

viewer.window.add_dock_widget(pbar_widget)
napari.run()