import pytest

from napari._qt.widgets.qt_dims import QtDims
from napari.components import Dims


@pytest.fixture
def qt_dims(qtbot):
    dims = Dims(ndim=2)
    qt_dims = QtDims(dims)
    qtbot.addWidget(qt_dims)
    yield qt_dims
    if qt_dims.is_playing:
        qt_dims.stop()
        qtbot.waitUntil(lambda: qt_dims.is_playing)
