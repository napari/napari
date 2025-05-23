from napari._qt.widgets.qt_logger import LogWidget


def test_qt_logger(qtbot):
    widget = LogWidget()
    qtbot.addWidget(widget)
    widget.show()
    widget.hide()
