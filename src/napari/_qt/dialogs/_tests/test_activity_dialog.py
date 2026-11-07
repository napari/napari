import os
import sys
import threading
from contextlib import contextmanager

import pytest
from qtpy.QtWidgets import QApplication

from napari._qt.widgets.qt_progress_bar import (
    QtLabeledProgressBar,
    QtProgressBarGroup,
)
from napari.utils import progress

SHOW = bool(sys.platform == 'linux' or os.getenv('CI'))


def qt_viewer_has_pbar(qt_viewer):
    """Returns True if the viewer has an active progress bar, else False"""
    return bool(
        qt_viewer.window._qt_viewer.window()._activity_dialog.findChildren(
            QtLabeledProgressBar
        )
    )


@contextmanager
def assert_pbar_added_to(viewer):
    """Context manager checks that progress bar is added on construction"""
    assert not qt_viewer_has_pbar(viewer)
    yield
    assert qt_viewer_has_pbar(viewer)


def activity_button_shows_indicator(activity_dialog):
    """Returns True if the progress indicator is visible, else False"""
    return activity_dialog._toggleButton._inProgressIndicator.isVisible()


def get_qt_labeled_progress_bar(prog, viewer):
    """Given viewer and progress, finds associated QtLabeledProgressBar"""
    activity_dialog = viewer.window._qt_viewer.window()._activity_dialog
    pbar = activity_dialog.get_pbar_from_prog(prog)

    return pbar


def get_progress_groups(qt_viewer):
    """Given viewer, find all QtProgressBarGroups in activity dialog"""
    return qt_viewer.window()._activity_dialog.findChildren(QtProgressBarGroup)


def test_activity_dialog_holds_progress(make_napari_viewer):
    """Progress gets added to dialog & once finished it gets removed"""
    viewer = make_napari_viewer(show=SHOW)

    with assert_pbar_added_to(viewer):
        r = range(100)
        prog = progress(r)
    pbar = get_qt_labeled_progress_bar(prog, viewer)
    assert pbar is not None
    assert pbar.progress is prog
    assert pbar.qt_progress_bar.maximum() == prog.total

    prog.close()
    assert not pbar.isVisible()


def test_progress_in_background_thread(make_napari_viewer, qtbot, monkeypatch):
    """Progress bar widgets are created and updated on the main thread."""
    viewer = make_napari_viewer(show=SHOW)
    used_on = []
    values = []
    set_value = QtLabeledProgressBar.setValue

    def _record_thread(self, value):
        used_on.append(threading.current_thread())
        values.append(value)
        set_value(self, value)

    monkeypatch.setattr(QtLabeledProgressBar, 'setValue', _record_thread)
    bar_created = threading.Event()

    def _work():
        with progress(total=3) as prog:
            bar_created.wait(timeout=5)
            for _ in range(3):
                prog.update(1)

    thread = threading.Thread(target=_work)
    thread.start()
    qtbot.waitUntil(lambda: qt_viewer_has_pbar(viewer))
    bar_created.set()
    thread.join()
    QApplication.processEvents()  # run the calls queued for the main thread

    # quick updates may be combined, but the last value must be shown
    assert values[-1] == 3
    assert set(used_on) == {threading.main_thread()}


def test_progress_with_context(make_napari_viewer):
    """Test adding/removing of progress bar with context manager"""
    viewer = make_napari_viewer(show=SHOW)

    with assert_pbar_added_to(viewer), progress(range(100)) as prog:
        pbar = get_qt_labeled_progress_bar(prog, viewer)
        assert pbar.qt_progress_bar.maximum() == prog.total == 100


def test_closing_viewer_no_error(make_napari_viewer):
    """Closing viewer with active progress doesn't cause RuntimeError"""
    viewer = make_napari_viewer(show=SHOW)

    assert not qt_viewer_has_pbar(viewer)
    with progress(range(100)):
        assert qt_viewer_has_pbar(viewer)
        viewer.close()


def test_progress_nested(make_napari_viewer):
    """Test nested progress bars are added with QtProgressBarGroup"""
    viewer = make_napari_viewer(show=SHOW)

    assert not qt_viewer_has_pbar(viewer)
    with progress(range(10)) as pbr:
        assert qt_viewer_has_pbar(viewer)
        pbr2 = progress(range(2), nest_under=pbr)
        prog_groups = get_progress_groups(viewer.window._qt_viewer)
        assert len(prog_groups) == 1
        # two progress bars + separator
        assert prog_groups[0].layout().count() == 3
        pbr2.close()
    assert not prog_groups[0].isVisible()


@pytest.mark.skipif(
    not SHOW,
    reason='viewer needs to be shown to test indicator',
)
def test_progress_indicator(make_napari_viewer):
    viewer = make_napari_viewer(show=SHOW)
    activity_dialog = viewer.window._qt_viewer.window()._activity_dialog

    # it's not clear why, but using the context manager here
    # causes test to fail, so we make the assertions explicitly
    assert not qt_viewer_has_pbar(viewer)
    with progress(range(10)):
        assert qt_viewer_has_pbar(viewer)
        assert activity_button_shows_indicator(activity_dialog)


@pytest.mark.skipif(
    bool(sys.platform == 'linux'),
    reason='need to debug sefaults with set_description',
)
def test_progress_set_description(make_napari_viewer):
    viewer = make_napari_viewer(show=SHOW)

    prog = progress(total=5)
    prog.set_description('Test')
    pbar = get_qt_labeled_progress_bar(prog, viewer)

    assert pbar.description_label.text() == 'Test: '

    prog.close()
