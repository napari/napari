from qtpy.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel
import time
from napari._qt.threading import thread_worker


@thread_worker
def long_running_function():
    """Just a long running function, most like viewer.update."""
    time.sleep(2)  # long function
    return 'finished!'


if __name__ == "__main__":
    app = QApplication([])

    widget = QWidget()
    layout = QHBoxLayout()
    widget.setLayout(layout)

    status = QLabel('ready...')
    layout.addWidget(status)

    worker = long_running_function(start_thread=False)
    # Note that signals/slots are best connected *before* starting the worker.
    worker.started.connect(lambda: status.setText("worker is running..."))
    worker.returned.connect(lambda x: status.setText(f"worker returned {x!r}"))
    worker.start()

    # # The above syntax is equivalent to this:
    # worker = long_running_function(
    #     connections={
    #         'started': lambda: status.setText("worker is running..."),
    #         'returned': lambda x: status.setText(f"worker returned {x!r}"),
    #     }
    # )

    widget.show()
    app.exec_()
