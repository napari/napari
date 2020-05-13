from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QWidget,
    QGridLayout,
    QLabel,
    QProgressBar,
)

import time
import sys
from napari._qt.threading import (
    thread_worker,
    ProgressWorker,
    wait_for_workers_to_quit,
)


@thread_worker
def simplest_function():
    """Just a long running function, most like viewer.update."""
    time.sleep(5)  # long function
    return 1


@thread_worker
def one_way_communication():
    """Periodically sends values back to the main thread."""
    # optionally, do setup here... this happens in the main thread
    yield 'ready'

    # do computationally intensitve work here

    for i in range(10):
        time.sleep(0.5)
        yield i

    # do optional teardown here
    return "done"


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
        '__len__': end,
    }

    # do computationally intensive work here
    i = start
    while i <= end:
        time.sleep(0.1)
        incoming = yield i  # incoming receives values from the main thread
        i = incoming if incoming is not None else i + 1

    # do optional teardown here
    return "done"


if __name__ == "__main__":
    app = QApplication([])

    window = QMainWindow()
    central = QWidget()
    layout = QGridLayout()
    central.setLayout(layout)
    window.setCentralWidget(central)
    status = QLabel('click start')

    # ##### This is the main addition here
    # the decorated function now returns a GeneratorWorker object, and the
    # Qthread in which it's running.
    # (optionally pass start=False to prevent immediate running)
    worker = two_way_communication_with_args(0, 40, start_thread=False)

    # it provides 5 signals: {started, yielded, returned, errored, finished}
    worker.yielded.connect(lambda x: status.setText(f"worker yielded {x}"))
    worker.returned.connect(lambda x: status.setText(f"worker returned {x}"))
    worker.errored.connect(lambda x: status.setText(f"worker errored {x}"))
    worker.started.connect(lambda: status.setText("worker started..."))
    worker.aborted.connect(lambda: status.setText("worker aborted"))

    # if you chose to pass start=False, you can start the thread manually
    start = QPushButton("Start", window)
    start.clicked.connect(worker.start)
    worker.finished.connect(lambda: start.setDisabled(True))
    worker.finished.connect(lambda: start.setText("Done"))

    def on_start():
        def handle_pause():
            worker.toggle_pause()
            start.setText("Pause" if worker.is_paused else "Continue")

        start.clicked.disconnect(worker.start)
        start.setText("Pause")
        start.clicked.connect(handle_pause)

    worker.started.connect(on_start)

    # send values into the function (like generator.send) using worker.send
    reset = QPushButton("reset count", window)
    reset.clicked.connect(lambda: worker.send(0))
    reset.clicked.connect(lambda: worker.reset_counter())

    # abort thread with worker.abort()
    abort = QPushButton("abort", window)
    abort.clicked.connect(lambda: worker.quit())

    # Progressbar
    progress = QProgressBar()
    worker.progress.connect(progress.setValue)

    layout.addWidget(start, 0, 0)
    layout.addWidget(reset, 0, 1)
    layout.addWidget(abort, 0, 2)
    layout.setColumnStretch(3, 1)
    layout.addWidget(status, 0, 3)
    layout.addWidget(progress, 1, 0, 1, 4)
    window.show()
    window.setFixedWidth(500)

    app.aboutToQuit.connect(wait_for_workers_to_quit)
    sys.exit(app.exec_())
