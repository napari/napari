"""Development widgets."""

from __future__ import annotations

import logging
import os
import typing as ty

logger = logging.getLogger()

if ty.TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget
    from qtreload.qt_reload import QtReloadWidget


def qdev(
    parent: QWidget | None = None,
    modules: ty.Iterable[str] = ('napari', 'napari_builtins'),
) -> QtReloadWidget:
    """Create reload widget."""
    from qtreload.qt_reload import QtReloadWidget

    dev_modules = os.environ.get('NAPARI_DEV_MODULES', '').split(',')
    modules = [*modules, *dev_modules]
    modules = [m for m in modules if m]  # filter out empty strings
    modules = set(modules)

    logger.debug('Creating reload widget for modules: {}.', modules)
    return QtReloadWidget(modules, parent=parent)
