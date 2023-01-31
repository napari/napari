"""
Multithreading two-way
======================

.. tags:: interactivity
"""
import time

import numpy as np
from qtpy.QtWidgets import (
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QWidget,
)

import napari
from napari.qt.threading import thread_worker


@thread_worker
def two_way_communication_with_args(start, end):
    """Both sends and receives values to & from the main thread.

    Accepts arguments, puts them on the worker object.
    Receives values from main thread with ``incoming = yield``
    Optionally returns a value at the end
    """

    # do computationally intensive work here
    i = start
    while i < end:
        i += 1
        time.sleep(0.1)
        # incoming receives values from the main thread
        # while yielding sends values back to the main thread
        incoming = yield i
        i = incoming if incoming is not None else i

    # do optional teardown here
    return "done"


class Controller(QWidget):
    def __init__(self) -> None:
        super().__init__()

        layout = QGridLayout()
        self.setLayout(layout)
        self.status = QLabel('Click "Start"', self)
        self.play_btn = QPushButton("Start", self)
        self.abort_btn = QPushButton("Abort!", self)
        self.reset_btn = QPushButton("Reset", self)
        self.progress_bar = QProgressBar()

        layout.addWidget(self.play_btn, 0, 0)
        layout.addWidget(self.reset_btn, 0, 1)
        layout.addWidget(self.abort_btn, 0, 2)
        layout.addWidget(self.status, 0, 3)
        layout.setColumnStretch(3, 1)
        layout.addWidget(self.progress_bar, 1, 0, 1, 4)


def create_connected_widget():
    """Builds a widget that can control a function in another thread."""
    w = Controller()
    steps = 40

    # the decorated function now returns a GeneratorWorker object, and the
    # Qthread in which it's running.
    # (optionally pass start=False to prevent immediate running)
    worker = two_way_communication_with_args(0, steps)

    w.play_btn.clicked.connect(worker.start)

    # it provides signals like {started, yielded, returned, errored, finished}
    worker.returned.connect(lambda x: w.status.setText(f"worker returned {x}"))
    worker.errored.connect(lambda x: w.status.setText(f"worker errored {x}"))
    worker.started.connect(lambda: w.status.setText("worker started..."))
    worker.aborted.connect(lambda: w.status.setText("worker aborted"))

    # send values into the function (like generator.send) using worker.send
    # abort thread with worker.abort()
    w.abort_btn.clicked.connect(lambda: worker.quit())

    def on_reset_button_pressed():
        # we want to avoid sending into a unstarted worker
        if worker.is_running:
            worker.send(0)

    def on_yield(x):
        # Receive events and update widget progress
        w.progress_bar.setValue(100 * x // steps)
        w.status.setText(f"worker yielded {x}")

    def on_start():
        def handle_pause():
            worker.toggle_pause()
            w.play_btn.setText("Pause" if worker.is_paused else "Continue")

        w.play_btn.clicked.disconnect(worker.start)
        w.play_btn.setText("Pause")
        w.play_btn.clicked.connect(handle_pause)

    def on_finish():
        w.play_btn.setDisabled(True)
        w.reset_btn.setDisabled(True)
        w.abort_btn.setDisabled(True)
        w.play_btn.setText("Done")

    w.reset_btn.clicked.connect(on_reset_button_pressed)
    worker.yielded.connect(on_yield)
    worker.started.connect(on_start)
    worker.finished.connect(on_finish)
    return w


if __name__ == "__main__":

    viewer = napari.view_image(np.random.rand(512, 512))
    w = create_connected_widget()
    viewer.window.add_dock_widget(w)

    napari.run()
