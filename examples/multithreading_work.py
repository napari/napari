from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QWidget,
    QHBoxLayout,
    QLabel,
)
import time
import sys
from napari._qt.utils import qthreaded


@qthreaded
def simplest_function():
    """Just a long running function, most like viewer.update."""
    yield
    time.sleep(5)  # long function
    return


@qthreaded
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


@qthreaded
def two_way_communication_with_args(start, end):
    """Both sends and receives values to & from the main thread.

    Accepts arguments, puts them on the worker object.
    Receives values from main thread with ``incoming = yield``
    Optionally returns a value at the end
    """

    # optionally, do setup here... this happens in the main thread
    # if you yield a dict, they will be added as attributes to the worker
    yield {'start_val': start, 'end_val': end, 'some_other_val': 'hi'}

    # do computationally intensive work here
    i = start
    while i <= end:
        time.sleep(0.5)
        incoming = yield i  # incoming receives values from the main thread
        i = incoming if incoming is not None else i + 1

    # do optional teardown here
    return "done"


if __name__ == "__main__":
    app = QApplication([])

    window = QMainWindow()
    central = QWidget()
    layout = QHBoxLayout()
    central.setLayout(layout)
    window.setCentralWidget(central)
    status = QLabel('click start')

    # ##### This is the main addition here
    # the decorated function now returns a GeneratorWorker object, and the
    # Qthread in which it's running.
    # (optionally pass start=False to prevent immediate running)
    worker, thread = two_way_communication_with_args(0, 8, start=False)

    # it provides 5 signals: {started, yielded, returned, errored, finished}
    worker.yielded.connect(lambda x: status.setText(f"worker yielded {x}"))
    worker.returned.connect(lambda x: status.setText(f"worker returned {x}"))
    worker.errored.connect(lambda x: status.setText(f"worker errored {x}"))

    # if you chose to pass start=False, you can start the thread manually
    start = QPushButton("start", window)
    start.clicked.connect(thread.start)

    # send values into the function (like generator.send) using worker.send
    reset = QPushButton("reset count", window)
    reset.clicked.connect(lambda: worker.send(0))

    # abort thread with worker.abort()
    abort = QPushButton("abort", window)
    abort.clicked.connect(lambda: worker.abort())

    layout.addWidget(start)
    layout.addWidget(reset)
    layout.addWidget(abort)
    layout.addWidget(status)

    window.show()
    sys.exit(app.exec_())
