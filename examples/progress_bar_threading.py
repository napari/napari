"""This file provides a minimal working example using a progress bar alongside
@thread_worker to report progress.
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
    # with each yield
    progress={'total': 5},
    # this does not preclude us from connecting other functions to any of the
    # worker signals (including `yielded`)
    connect={'yielded': handle_yields} 
    )
def my_long_running_thread(_):
    for i in range(5):
        sleep(0.1)
        yield i

# in the previous example, yielding values beyond the total value passed to
# the progress dictionary simply leaves the progress bar on 100% with 
# no further indicator.
@thread_worker(
    # If we are unsure of the number of expected yields,
    # we can instruct the progress bar to become indeterminate once the 
    # expected number of yields is exceeded using `may_exceed_total`
    progress={'total': 5, 'may_exceed_total': True},
    connect={'yielded': handle_yields}
)
def my_indeterminate_thread(_):
    for i in range(10):
        sleep(0.1)
        yield i

button_layout = QVBoxLayout()
start_btn = QPushButton("Start")
start_btn.clicked.connect(my_long_running_thread)
button_layout.addWidget(start_btn)

start_btn2 = QPushButton("Start Indeterminate")
start_btn2.clicked.connect(my_indeterminate_thread)
button_layout.addWidget(start_btn2)

pbar_widget = QWidget()
pbar_widget.setLayout(button_layout)
viewer.window.add_dock_widget(pbar_widget, allowed_areas=["right"])

napari.run()