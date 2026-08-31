from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager
from typing import TYPE_CHECKING

from psygnal import Signal

_LOG_SEPARATOR = '<NAPARI_LOG_SEPARATOR>'


if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any


def _get_log_level_value(log_level_name: str | None) -> int:
    if log_level_name is None:
        return logging.NOTSET
    return logging.getLevelNamesMapping().get(log_level_name, logging.NOTSET)


class _LogStream:
    """
    Custom stream object to receive logging info.

    Needs to define `write` and `flush` methods which are called by the log handler.
    """

    changed = Signal()

    def __init__(self) -> None:
        self.logs: deque[tuple[Any, ...]] = deque(maxlen=100_000)

    def write(self, log_msg: str) -> None:
        logger, level_name, time, thread, msg = log_msg.split(_LOG_SEPARATOR)
        level_value = _get_log_level_value(level_name)
        self.logs.append((logger, level_value, level_name, time, thread, msg))
        # TODO: actually save log to a file somewhere so it can be retrieved?
        self.changed()

    def flush(self) -> None:
        pass

    def get_filtered_logs_html(
        self,
        level: int = logging.DEBUG,
        text_filter: str = '',
        last_only: bool = False,
    ) -> list[str]:
        if isinstance(level, str):
            level = _get_log_level_value(level)

        logs = [_LOG_STREAM.logs[-1]] if last_only else _LOG_STREAM.logs

        selected = [
            (logger_name, level_value, *others)
            for logger_name, level_value, *others in logs
            if level_value >= level
        ]

        # TODO: fuzzy search?
        text_filter = text_filter.lower()
        filtered = [
            log
            for log in selected
            if any(text_filter in str(field).lower() for field in log)
        ]

        return [
            f'{_html_tag_for_level(level_name, level_value)} '
            f'<b>{name}</b> '
            f'<font style="color:gray"><i>[{time}] ({thread})</i></font>: '
            f'{msg}'
            for name, level_value, level_name, time, thread, msg in filtered
        ]


_LOG_STREAM = _LogStream()
_LOG_HANDLER = logging.StreamHandler(_LOG_STREAM)
_LOG_HANDLER.setFormatter(
    logging.Formatter(
        f'%(name)s{_LOG_SEPARATOR}%(levelname)s{_LOG_SEPARATOR}%(asctime)s{_LOG_SEPARATOR}%(threadName)s{_LOG_SEPARATOR}%(message)s'
    )
)
_LOG_HANDLER.setLevel(logging.DEBUG)


def _html_tag_for_level(level_name: str, level_value: int) -> str:
    """
    Generate html tag for the appropriate logging level.
    """
    colors = {
        logging.INFO: 'cyan',
        logging.WARNING: 'orange',
        logging.ERROR: 'red',
        logging.CRITICAL: 'magenta',
    }
    color = 'blue'
    for level, level_color in colors.items():
        if level_value >= level:
            color = level_color
    # this is ugly AF but html is weird and I don't get it
    padding = '&nbsp;' * (8 - len(level_name))
    return f'<font style="color:{color}">{level_name}{padding}</font>'


def register_logger_to_napari_handler(module: str) -> None:
    """
    Register a specific module's logger to use our custom log handler.

    .. version-added:: 0.10.0
    """
    logger = logging.getLogger(module)
    # ensure the default "last resort" logging to console remains
    if not logger.handlers and logging.lastResort:
        logger.addHandler(logging.lastResort)
    logger.addHandler(_LOG_HANDLER)


def deregister_logger_from_napari_handler(module: str) -> None:
    """
    Deregister a specific module's logger from using our custom log handler.

    .. version-added:: 0.10.0
    """
    logger = logging.getLogger(module)
    # fails silently
    logger.removeHandler(_LOG_HANDLER)


@contextmanager
def using_napari_log_handler(
    module: str,
) -> Generator[None, None, None]:
    """
    Context manager to temporarily register a module's logger to our custom handler.

    .. version-added:: 0.10.0
    """
    register_logger_to_napari_handler(module)
    yield
    deregister_logger_from_napari_handler(module)
