from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from napari._qt.widgets.qt_dims import QtDims
from napari.components import Dims

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@pytest.fixture
def qt_dims(qtbot: QtBot) -> QtDims:
    """Fixture that provides a QtDims and ensures
    that animation is stopped after the test failed.

    The test itself should ensure that the animation is stopped,
    but in case of a failure before the end of the test,
    this fixture will take care of stopping it
    to avoid segfaults in following tests.
    """
    qt_dims = QtDims(Dims(ndim=2))
    qtbot.addWidget(qt_dims, before_close_func=_stop_animation)
    return qt_dims


def _stop_animation(qt_dims: QtDims) -> None:
    if qt_dims.is_playing:
        qt_dims.stop()
