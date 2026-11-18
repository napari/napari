"""
Message routing in napari
=========================

Compare terminal output, logs, warnings, notifications, and errors.

Run napari from a terminal and optionally open **Help > Show logs**.
Use the buttons in the dock widget to compare what appears in the terminal,
the bottom-right popup area, and the log dock.

napari shows a Python warning once per line of code that raises it, even if
the message changes, so the ``warnings.warn()`` and thread_worker warning
buttons only show their first press.

.. tags:: gui, dev
"""

import logging
import warnings

from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

import napari
from napari.qt import thread_worker
from napari.utils.notifications import (
    notification_manager,
    show_error,
    show_info,
    show_warning,
)

logger = logging.getLogger('napari.examples.message_routing')
logger.setLevel(logging.DEBUG)


def _raise_value_error(message: str) -> None:
    raise ValueError(message)


def print_message() -> None:
    print('print(): terminal only')


def log_warning() -> None:
    logger.warning(
        'logger.warning(): visible in logging handlers and Help > Show logs'
    )


def python_warning() -> None:
    warnings.warn(
        'warnings.warn(): shown once, later presses are hidden',
        stacklevel=1,
    )


def napari_info() -> None:
    show_info('show_info(): explicit napari info notification')


def napari_warning() -> None:
    show_warning('show_warning(): explicit napari warning notification')


def flattened_error() -> None:
    try:
        _raise_value_error(
            'show_error(): caught exception flattened to a plain message'
        )
    except Exception as exc:  # noqa: BLE001
        show_error(str(exc))


def forwarded_error() -> None:
    try:
        _raise_value_error(
            'receive_error(): caught exception forwarded with traceback'
        )
    except Exception as exc:  # noqa: BLE001
        notification_manager.receive_error(
            type(exc), exc, exc.__traceback__
        )


def uncaught_error() -> None:
    raise ValueError('uncaught exception: napari should show View Traceback')


@thread_worker(start_thread=True)
def worker_warning() -> None:
    warnings.warn(
        'thread_worker warning: raised in a worker thread',
        stacklevel=1,
    )


@thread_worker(start_thread=True)
def worker_error() -> None:
    raise ValueError('thread_worker error: raised in a worker thread')


def _button(text: str, callback) -> QPushButton:
    button = QPushButton(text)
    # `clicked` sends a `checked` argument; a thread_worker would pass it on
    # to the function, so call the callback without arguments.
    button.clicked.connect(lambda: callback())
    return button


def _build_controls() -> QWidget:
    controls = QWidget()
    layout = QVBoxLayout(controls)
    instructions = QLabel(
        'Compare the terminal, the popup area and Help > Show logs'
        ' while pressing the buttons below.'
    )
    instructions.setWordWrap(True)
    layout.addWidget(instructions)
    layout.addWidget(_button('print()', print_message))
    layout.addWidget(_button('logger.warning()', log_warning))
    layout.addWidget(_button('warnings.warn()', python_warning))
    layout.addWidget(_button('show_info()', napari_info))
    layout.addWidget(_button('show_warning()', napari_warning))
    layout.addWidget(_button('show_error(str(exc))', flattened_error))
    layout.addWidget(
        _button('receive_error(type(exc), exc, tb)', forwarded_error)
    )
    layout.addWidget(
        _button('uncaught raise ValueError(...)', uncaught_error)
    )
    layout.addWidget(_button('thread_worker warning', worker_warning))
    layout.addWidget(_button('thread_worker error', worker_error))
    layout.addStretch(1)
    return controls


viewer = napari.Viewer()
viewer.window.add_dock_widget(
    _build_controls(), area='right', name='message routing'
)

if __name__ == '__main__':
    napari.run()
