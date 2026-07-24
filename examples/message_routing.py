"""
Message routing in napari
=========================

Compare terminal output, logs, warnings, notifications, and errors.

Run napari from a terminal and optionally open **Help > Show logs**.
Use the buttons in the dock widget to compare what appears in the terminal,
the bottom-right popup area, and the log dock.

Repeated identical Python warnings are deduplicated, so this example includes
both a repeated-warning button and a unique-warning button.

.. tags:: gui, dev
"""

import logging
import warnings
from itertools import count

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
warning_counter = count(1)


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
        (
            'warnings.warn(): repeated identical warning; it will appear '
            'as a napari popup once hooks are installed'
        ),
        stacklevel=2,
    )


def python_warning_unique() -> None:
    count_value = next(warning_counter)
    warnings.warn(
        (
            f'warnings.warn(): unique warning #{count_value}; it may also '
            'appear as a napari popup once hooks are installed'
        ),
        stacklevel=2,
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
        'thread_worker warning: forwarded from a worker thread',
        stacklevel=2,
    )


@thread_worker(start_thread=True)
def worker_error() -> None:
    raise ValueError('thread_worker error: forwarded from a worker thread')


def _button(text: str, callback) -> QPushButton:
    button = QPushButton(text)
    button.clicked.connect(callback)
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
    layout.addWidget(_button('warnings.warn() repeated', python_warning))
    layout.addWidget(_button('warnings.warn() unique', python_warning_unique))
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
