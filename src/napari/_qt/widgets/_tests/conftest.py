from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from napari._qt.widgets.qt_dims import QtDims
from napari.components import Dims

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@pytest.fixture
def qt_dims(qtbot: QtBot) -> QtDims:
    """Fixture that provides a QtDims registered with qtbot.

    pytest-qt closes the widget during test teardown and
    ``QtDims.closeEvent`` stops any running animation, so a test that fails
    while playing cannot leave the animation thread running when the
    widget is deleted.
    """
    qt_dims = QtDims(Dims(ndim=2))
    qtbot.addWidget(qt_dims)
    return qt_dims
