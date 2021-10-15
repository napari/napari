import os
import sys
from contextlib import contextmanager

import pytest

from napari._qt.widgets.qt_progress_bar import (
    QtLabeledProgressBar,
    QtProgressBarGroup,
)
from napari.utils import progress

SHOW = bool(sys.platform == 'linux' or os.getenv("CI"))


def get_progress_groups(qt_viewer):
    return qt_viewer.window()._activity_dialog.findChildren(QtProgressBarGroup)


def qt_viewer_has_pbar(qt_viewer):
    return bool(
        qt_viewer.window.qt_viewer.window()._activity_dialog.findChildren(
            QtLabeledProgressBar
        )
    )


def activity_button_shows_indicator(activity_dialog):
    return activity_dialog._toggleButton._inProgressIndicator.isVisible()


def get_qt_labeled_progress_bar(prog, viewer):
    activity_dialog = viewer.window.qt_viewer.window()._activity_dialog
    pbar = activity_dialog.get_pbar_from_prog(prog)

    return pbar


@contextmanager
def assert_pbar_added_to(viewer):
    assert not qt_viewer_has_pbar(viewer)
    yield
    assert qt_viewer_has_pbar(viewer)


# if we change evented attributes of progress QtLabeledProgress bar changes
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


def test_progress_with_context(make_napari_viewer):
    viewer = make_napari_viewer(show=SHOW)

    assert not qt_viewer_has_pbar(viewer)
    with progress(range(100)) as prog:
        assert qt_viewer_has_pbar(viewer)
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
    viewer = make_napari_viewer(show=SHOW)

    assert not qt_viewer_has_pbar(viewer)
    with progress(range(10)) as pbr:
        assert qt_viewer_has_pbar(viewer)
        pbr2 = progress(range(2), nest_under=pbr)
        prog_groups = get_progress_groups(viewer.window.qt_viewer)
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
    activity_dialog = viewer.window.qt_viewer.window()._activity_dialog

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
    prog.set_description("Test")
    pbar = get_qt_labeled_progress_bar(prog, viewer)

    assert pbar.qt_progress_bar.description_label.text() == "Test: "

    prog.close()
