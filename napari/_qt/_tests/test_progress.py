import contextvars
from contextlib import contextmanager

import numpy as np
import pytest

pytest.importorskip('qtpy', reason='Cannot test progress without qtpy.')

from napari._qt.widgets.qt_progress_bar import ProgressBarGroup  # noqa
from napari.qt import progrange, progress  # noqa


def get_progress_groups(viewer):
    viewer_window = viewer.window.qt_viewer.window()
    return viewer_window._activity_dialog.findChildren(ProgressBarGroup)


def qt_viewer_has_pbar(viewer):
    return bool(get_progress_groups(viewer))


@contextmanager
def assert_pbar_added_to(viewer):
    assert not qt_viewer_has_pbar(viewer)
    yield
    assert qt_viewer_has_pbar(viewer)


def test_progress_with_iterable(make_napari_viewer):
    viewer = make_napari_viewer()

    with assert_pbar_added_to(viewer):
        r = range(100)
        pbr = progress(r)

    assert pbr.iterable is r
    assert pbr.n == 0
    assert pbr._pbar.pbar.maximum() == pbr.total == 100

    pbr.close()


def test_progress_with_ndarray(make_napari_viewer):
    viewer = make_napari_viewer()

    with assert_pbar_added_to(viewer):
        iter = np.random.random((100, 100))
        pbr = progress(iter)

    assert pbr.iterable is iter
    assert pbr.n == 0
    assert pbr._pbar.pbar.maximum() == pbr.total

    pbr.close()


def test_progress_with_total(make_napari_viewer):
    viewer = make_napari_viewer()

    with assert_pbar_added_to(viewer):
        pbr = progress(total=5)

    assert pbr.n == 0
    assert pbr._pbar.pbar.maximum() == pbr.total == 5

    pbr.update(1)
    assert pbr.n == 1

    pbr.close()


def test_progress_with_context(make_napari_viewer):
    viewer = make_napari_viewer()

    with assert_pbar_added_to(viewer):
        with progress(range(100)) as pbr:
            assert pbr.n == 0
            assert pbr._pbar.pbar.maximum() == pbr.total == 100


def test_progress_no_viewer():
    assert list(progress(range(10))) == list(range(10))

    with progress(total=5) as pbr:
        pbr.set_description('Test')
        assert pbr.desc == "Test: "

        pbr.update(3)
        assert pbr.n == 3


def test_progress_nested_viewer(make_napari_viewer):
    viewer = make_napari_viewer()

    with assert_pbar_added_to(viewer):
        with progress(range(10)):
            pbr2 = progress(range(2))
            prog_groups = get_progress_groups(viewer)
            assert len(prog_groups) == 1
            # two pbars + separator
            assert prog_groups[0].layout().count() == 3
            pbr2.close()


def test_progress_nested_context(make_napari_viewer):
    make_napari_viewer()
    with progress(range(2)) as pbr:
        assert isinstance(pbr._group_token, contextvars.Token)
        assert pbr._group_token.var.get()

    pbr2 = progress(range(2))
    assert pbr2._group_token is None
    pbr2.close()


def test_progress_update(make_napari_viewer):
    make_napari_viewer()

    pbr = progress(total=10)

    assert pbr.n == 0
    assert pbr._pbar.pbar.value() == 0

    pbr.update(1)
    pbr.refresh()  # not sure why this has to be called manually here

    assert pbr.n == 1
    assert pbr._pbar.pbar.value() == 1

    pbr.update(2)
    pbr.refresh()

    assert pbr.n == 3
    assert pbr._pbar.pbar.value() == 3

    pbr.close()


def test_progress_set_description(make_napari_viewer):
    make_napari_viewer()

    pbr = progress(total=5)
    pbr.set_description("Test")

    assert pbr.desc == "Test: "
    assert pbr._pbar.description_label.text() == "Test: "

    pbr.close()


def test_progrange():
    assert progress(range(10)).iterable == progrange(10).iterable
