from qtpy import PYQT5, PYSIDE2
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QApplication

from napari._qt.widgets.qt_scrollbar import ModifiedScrollBar

# Enable high DPI scaling prior to instantiating QApplication
# This is required for the test to pass on high DPI displays
if PYQT5 or PYSIDE2:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)


def test_modified_scrollbar_click(qtbot):
    w = ModifiedScrollBar(Qt.Horizontal)
    w.resize(100, 10)
    assert w.value() == 0
    qtbot.mousePress(w, Qt.LeftButton, pos=QPoint(50, 5))
    # the normal QScrollBar would have moved to "10"
    assert w.value() >= 40
