import time

from qtpy.QtWidgets import (
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QWidget,
)

import napari
import numpy as np
from napari._qt.threading import ProgressWorker, thread_worker


@thread_worker(worker_class=ProgressWorker)
def two_way_communication_with_args(start, end):
    """Both sends and receives values to & from the main thread.

    Accepts arguments, puts them on the worker object.
    Receives values from main thread with ``incoming = yield``
    Optionally returns a value at the end
    """

    # optionally, do setup here... this happens in the main thread
    # if you yield a dict, they will be added as attributes to the worker
    yield {
        'start_val': start,
        'end_val': end,
        'some_other_val': 'hi',
        '__len__': end,  # note, the '__len__' key is required for ProgressWorker
    }

    # do computationally intensive work here
    i = start
    while i <= end:
        time.sleep(0.1)
        # incoming receives values from the main thread
        # while yielding sends values back to the main thread
        incoming = yield i
        i = incoming if incoming is not None else i + 1

    # do optional teardown here
    return "done"


class Controller(QWidget):
    def __init__(self):
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

    # the decorated function now returns a GeneratorWorker object, and the
    # Qthread in which it's running.
    # (optionally pass start=False to prevent immediate running)
    worker = two_way_communication_with_args(0, 40, _start_thread=False)
    w.play_btn.clicked.connect(worker.start)

    # it provides signals like {started, yielded, returned, errored, finished}
    worker.yielded.connect(lambda x: w.status.setText(f"worker yielded {x}"))
    worker.returned.connect(lambda x: w.status.setText(f"worker returned {x}"))
    worker.errored.connect(lambda x: w.status.setText(f"worker errored {x}"))
    worker.started.connect(lambda: w.status.setText("worker started..."))
    worker.aborted.connect(lambda: w.status.setText("worker aborted"))

    # if you chose to pass start=False, you can start the thread manually
    worker.finished.connect(lambda: w.play_btn.setDisabled(True))
    worker.finished.connect(lambda: w.play_btn.setText("Done"))

    # send values into the function (like generator.send) using worker.send
    w.reset_btn.clicked.connect(lambda: worker.send(0))
    w.reset_btn.clicked.connect(lambda: worker.set_counter(-1))
    # abort thread with worker.abort()
    w.abort_btn.clicked.connect(lambda: worker.quit())
    # Receive events and update widget progress
    worker.progress.connect(w.progress_bar.setValue)

    def on_start():
        def handle_pause():
            worker.toggle_pause()
            w.play_btn.setText("Pause" if worker.is_paused else "Continue")

        w.play_btn.clicked.disconnect(worker.start)
        w.play_btn.setText("Pause")
        w.play_btn.clicked.connect(handle_pause)

    worker.started.connect(on_start)
    return w


if __name__ == "__main__":

    with napari.gui_qt():
        viewer = napari.view_image(np.random.rand(512, 512))
        w = create_connected_widget()
        viewer.window.add_dock_widget(w)
