import os
import sys
from contextlib import contextmanager

import numpy as np
import pytest

pytest.importorskip('qtpy', reason='Cannot test progress without qtpy.')

from napari._qt.widgets.qt_progress_bar import ProgressBar  # noqa
from napari.qt import progrange, progress  # noqa

SHOW = bool(sys.platform == 'linux' or os.getenv("CI"))


def qt_viewer_has_pbar(qt_viewer):
    return bool(qt_viewer.activityDock.widget().findChild(ProgressBar))


@contextmanager
def assert_pbar_added_to(viewer):
    assert not qt_viewer_has_pbar(viewer.window.qt_viewer)
    yield
    assert qt_viewer_has_pbar(viewer.window.qt_viewer)


def test_progress_with_iterable(make_napari_viewer):
    viewer = make_napari_viewer(show=SHOW)

    with assert_pbar_added_to(viewer):
        r = range(100)
        pbr = progress(r)
    assert pbr.iterable is r
    assert pbr.n == 0
    assert pbr._pbar.pbar.maximum() == pbr.total == 100

    pbr.close()


def test_progress_with_ndarray(make_napari_viewer):
    viewer = make_napari_viewer(show=SHOW)

    with assert_pbar_added_to(viewer):
        iter_ = np.random.random((100, 100))
        pbr = progress(iter_)

    assert pbr.iterable is iter_
    assert pbr.n == 0
    assert pbr._pbar.pbar.maximum() == pbr.total

    pbr.close()


def test_progress_with_total(make_napari_viewer):
    viewer = make_napari_viewer(show=SHOW)

    with assert_pbar_added_to(viewer):
        pbr = progress(total=5)

    assert pbr.n == 0
    assert pbr._pbar.pbar.maximum() == pbr.total == 5

    pbr.update(1)
    assert pbr.n == 1

    pbr.close()


def test_progress_with_context(make_napari_viewer):
    viewer = make_napari_viewer(show=SHOW)

    with assert_pbar_added_to(viewer):
        with progress(range(100)) as pbr:
            assert pbr.n == 0
            assert pbr._pbar.pbar.maximum() == pbr.total == 100


def test_progress_no_viewer():
    assert list(progress(range(10))) == list(range(10))

    with progress(total=5) as pbr:
        # TODO: debug segfaults
        if sys.platform != 'linux':
            pbr.set_description('Test')
            assert pbr.desc == "Test: "

        pbr.update(3)
        assert pbr.n == 3


def test_progress_update(make_napari_viewer):
    make_napari_viewer(show=SHOW)

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


@pytest.mark.skipif(
    bool(sys.platform == 'linux'),
    reason='need to debug sefaults with set_description',
)
def test_progress_set_description(make_napari_viewer):
    make_napari_viewer(show=SHOW)

    pbr = progress(total=5)
    pbr.set_description("Test")

    assert pbr.desc == "Test: "
    assert pbr._pbar.description_label.text() == "Test: "

    pbr.close()


def test_progrange():
    with progrange(10) as pbr:
        with progress(range(10)) as pbr2:
            assert pbr.iterable == pbr2.iterable
