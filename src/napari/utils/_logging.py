from __future__ import annotations

import logging
import sys
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from psygnal import Signal

_LOG_SEPARATOR = '<NAPARI_LOG_SEPARATOR>'


if TYPE_CHECKING:
    from typing import Any


if sys.version_info < (3, 11):
    # in python 3.10 there's no public mapping, and this getLevelName function
    # is a bit less resilient (e.g: it "works" both ways (name <-> value), and if
    # invalid values are passed, it just makes up a new level)
    def get_log_level_value(log_level_name: str | None) -> int:
        if log_level_name is None:
            return logging.NOTSET
        return logging.getLevelName(log_level_name)
else:

    def get_log_level_value(log_level_name: str | None) -> int:
        if log_level_name is None:
            return logging.NOTSET
        return logging.getLevelNamesMapping().get(
            log_level_name, logging.NOTSET
        )


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
        level_value = get_log_level_value(level_name)
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
            level = get_log_level_value(level)

        logs = [LOG_STREAM.logs[-1]] if last_only else LOG_STREAM.logs

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


LOG_STREAM = _LogStream()
LOG_HANDLER = logging.StreamHandler(LOG_STREAM)
LOG_HANDLER.setFormatter(
    logging.Formatter(
        f'%(name)s{_LOG_SEPARATOR}%(levelname)s{_LOG_SEPARATOR}%(asctime)s{_LOG_SEPARATOR}%(threadName)s{_LOG_SEPARATOR}%(message)s'
    )
)
LOG_HANDLER.setLevel(logging.DEBUG)


@contextmanager
def register_logger_to_napari_handler(
    module: str,
) -> Generator[None, None, None]:
    """
    Register a specific module's logger to use our custom log handler.
    """
    logger = logging.getLogger(module)
    # ensure the default "last resort" logging to console remains
    if not logger.handlers and logging.lastResort:
        logger.addHandler(logging.lastResort)
    logger.addHandler(LOG_HANDLER)
    yield
    logger.removeHandler(LOG_HANDLER)


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
