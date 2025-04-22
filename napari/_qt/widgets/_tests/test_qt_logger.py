import pytest

from napari._qt.widgets.qt_logger import LogWidget


@pytest.skip('This test is not currently working')
def test_qt_logger(qtbot):
    widget = LogWidget()
    qtbot.addWidget(widget)
    widget.show()
    widget.hide()
