from qtpy.QtCore import Qt, QPoint
from ..qt_scrollbar import ModifiedScrollBar


def test_modified_scrollbar_click(qtbot):
    w = ModifiedScrollBar(Qt.Horizontal)
    w.resize(100, 10)
    assert w.value() == 0
    qtbot.mousePress(w, Qt.LeftButton, pos=QPoint(50, 5))
    # the normal QScrollBar would have moved to "10"
    assert w.value() >= 40
