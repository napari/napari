from argparse import Namespace

from napari._qt.widgets.qt_progress_bar import (
    QtLabeledProgressBar,
    QtProgressBarGroup,
)
from napari.utils.progress import cancelable_progress


def test_create_qt_labeled_progress_bar(qtbot):
    progress = QtLabeledProgressBar()
    qtbot.addWidget(progress)


def test_qt_labeled_progress_bar_base(qtbot):
    progress = QtLabeledProgressBar()
    qtbot.addWidget(progress)
    progress.setRange(0, 10)
    assert progress.qt_progress_bar.value() == -1
    progress.setValue(5)
    assert progress.qt_progress_bar.value() == 5
    progress.setDescription("text")
    assert progress.description_label.text() == "text: "


def test_qt_labeled_progress_bar_event_handle(qtbot):
    progress = QtLabeledProgressBar()
    qtbot.addWidget(progress)

    assert progress.qt_progress_bar.maximum() != 10
    progress._set_total(Namespace(value=10))
    assert progress.qt_progress_bar.maximum() == 10
    assert progress._get_value() == -1
    progress._set_value(Namespace(value=5))
    assert progress._get_value() == 5
    assert progress.description_label.text() == ""
    progress._set_description(Namespace(value="text"))
    assert progress.description_label.text() == "text: "
    assert progress.eta_label.text() == ""
    progress._set_eta(Namespace(value="test"))
    assert progress.eta_label.text() == "test"
    progress._make_indeterminate(None)
    assert progress.qt_progress_bar.maximum() == 0


def test_qt_labeled_progress_bar_cancel(qtbot):
    prog = cancelable_progress(total=10)
    progress = QtLabeledProgressBar(prog=prog)

    progress.cancel_button.clicked.emit()
    qtbot.waitUntil(lambda: prog.is_canceled, timeout=500)


def test_create_qt_progress_bar_group(qtbot):
    group = QtProgressBarGroup(QtLabeledProgressBar())
    qtbot.addWidget(group)
