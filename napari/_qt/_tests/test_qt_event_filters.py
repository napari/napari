import pytest

from napari._qt.qt_event_filters import QtToolTipEventFilter

pytest.importorskip('qtpy', reason='Cannot test event filters without qtpy.')


@pytest.mark.parametrize(
    "tooltip,is_qt_tag_present",
    [
        (
            "<html>"
            "<p>A widget to test that a rich text tooltip might be detected "
            "and therefore not changed to include a qt tag</p>"
            "</html>",
            False,
        ),
        (
            "A widget to test that a non-rich text tooltip might "
            "be detected and therefore changed",
            True,
        ),
    ],
)
def test_qt_tooltip_event_filter(qtbot, tooltip, is_qt_tag_present):
    """
    Check that the tooltip event filter only changes tooltips with non-rich text.
    """
    from qtpy.QtCore import QEvent
    from qtpy.QtWidgets import QWidget

    # event filter object and QEvent
    event_filter_handler = QtToolTipEventFilter()
    qevent = QEvent(QEvent.ToolTipChange)

    # check if tooltip is changed by the event filter
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setToolTip(tooltip)
    event_filter_handler.eventFilter(widget, qevent)
    assert ("<qt>" in widget.toolTip()) == is_qt_tag_present
