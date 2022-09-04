"""
Progress bar threading
======================

This file provides a minimal working example using a progress bar alongside
``@thread_worker`` to report progress.

.. tags:: interactivity
"""
from time import sleep
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
import napari
from napari.qt import thread_worker

viewer = napari.Viewer()


def handle_yields(yielded_val):
    print(f"Just yielded: {yielded_val}")


# generator thread workers can provide progress updates on each yield
@thread_worker(
    # passing a progress dictionary with the total number of expected yields
    # will place a progress bar in the activity dock and increment its value
    # with each yield. We can optionally pass a description for the bar
    # using the 'desc' key.
    progress={'total': 5, 'desc': 'thread-progress'},
    # this does not preclude us from connecting other functions to any of the
    # worker signals (including `yielded`)
    connect={'yielded': handle_yields},
)
def my_long_running_thread(*_):
    for i in range(5):
        sleep(0.1)
        yield i


@thread_worker(
    # If we are unsure of the number of expected yields,
    # we can still pass an estimate to total,
    # and the progress bar will become indeterminate
    # once this number is exceeded.
    progress={'total': 5},
    # we can also get a simple indeterminate progress bar
    # by passing progress=True
    connect={'yielded': handle_yields},
)
def my_indeterminate_thread(*_):
    for i in range(10):
        sleep(0.1)
        yield i


def return_func(return_val):
    print(f"Returned: {return_val}")


# finally, a FunctionWorker can still provide an indeterminate
# progress bar, but will not take a total>0
@thread_worker(
    progress={'total': 0, 'desc': 'FunctionWorker'},
    # can use progress=True if not passing description
    connect={'returned': return_func},
)
def my_function(*_):
    sum = 0
    for i in range(10):
        sum += i
        sleep(0.1)
    return sum


button_layout = QVBoxLayout()
start_btn = QPushButton("Start")
start_btn.clicked.connect(my_long_running_thread)
button_layout.addWidget(start_btn)

start_btn2 = QPushButton("Start Indeterminate")
start_btn2.clicked.connect(my_indeterminate_thread)
button_layout.addWidget(start_btn2)

start_btn3 = QPushButton("Start FunctionWorker")
start_btn3.clicked.connect(my_function)
button_layout.addWidget(start_btn3)

pbar_widget = QWidget()
pbar_widget.setLayout(button_layout)
pbar_widget.setObjectName("Threading Examples")
viewer.window.add_dock_widget(pbar_widget, allowed_areas=["right"])

# showing the activity dock so we can see the progress bars
viewer.window._status_bar._toggle_activity_dock(True)

if __name__ == '__main__':
    napari.run()
