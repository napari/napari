from collections.abc import Generator

import pytest
from pytestqt.qtbot import QtBot

from napari._qt.widgets.qt_dims import QtDims
from napari.components import Dims


@pytest.fixture
def qt_dims(qtbot: QtBot) -> Generator[Dims, None, None]:
    """Fixture that provides a QtDims and ensures
    that animation is stopped after the test failed.

    The test itself should ensure that the animation is stopped,
    but in case of a failure before the end of the test,
    this fixture will take care of stopping it
    to avoid segfaults in following tests.
    """
    dims = Dims(ndim=2)
    qt_dims = QtDims(dims)
    qtbot.addWidget(qt_dims)
    yield qt_dims
    if qt_dims.is_playing:
        qt_dims.stop()
        qtbot.waitUntil(lambda: not qt_dims.is_playing)
