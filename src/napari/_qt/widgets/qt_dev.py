"""Development widgets."""

import logging
import os
import typing as ty

logger = logging.getLogger()

if ty.TYPE_CHECKING:
    from qtreload.qt_reload import QtReloadWidget


def qdev(
    parent=None,
    modules: ty.Iterable[str] = ('napari', 'napari_builtins'),
) -> 'QtReloadWidget':
    """Create reload widget."""
    from qtreload.qt_reload import QtReloadWidget

    dev_modules = os.environ.get('NAPARI_DEV_MODULES', '').split(',')
    modules = [*modules, *dev_modules]
    modules = set(modules)

    logger.debug('Creating reload widget for modules: {}.', modules)
    return QtReloadWidget(modules, parent=parent)
