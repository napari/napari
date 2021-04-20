import numpy as np
import pytest

from ..progress import ProgressBar, progress

try:
    import qtpy  # noqa
except ImportError:
    pytest.skip('Cannot test progress without qtpy.', allow_module_level=True)
except RuntimeError:
    pytest.skip(
        'Cannot test progress without Qt bindings.', allow_module_level=True
    )


def activity_dock_children(viewer):
    return viewer.window.qt_viewer.activityDock.children()[4].children()


def test_progress_with_iterable(make_napari_viewer):
    viewer = make_napari_viewer()
    r = range(100)
    pbr = progress(r)

    assert any(
        isinstance(wdg, ProgressBar) for wdg in activity_dock_children(viewer)
    )
    assert pbr.iterable is r
    assert pbr.n == 0
    assert pbr._pbar.pbar.maximum() == pbr.total == 100


def test_progress_with_ndarray(make_napari_viewer):
    viewer = make_napari_viewer()
    iter = np.random.random((100, 100))
    pbr = progress(iter)

    assert any(
        isinstance(wdg, ProgressBar) for wdg in activity_dock_children(viewer)
    )
    assert pbr.iterable is iter
    assert pbr.n == 0
    assert pbr._pbar.pbar.maximum() == pbr.total


def test_progress_with_total(make_napari_viewer):
    viewer = make_napari_viewer()
    pbr = progress(total=5)

    assert any(
        isinstance(wdg, ProgressBar) for wdg in activity_dock_children(viewer)
    )
    assert pbr.n == 0
    assert pbr._pbar.pbar.maximum() == pbr.total == 5

    pbr.update(1)
    assert pbr.n == 1

    pbr.close()


def test_progress_with_context(make_napari_viewer):
    viewer = make_napari_viewer()

    with progress(range(100)) as pbr:
        assert any(
            isinstance(wdg, ProgressBar)
            for wdg in activity_dock_children(viewer)
        )
        assert pbr.n == 0
        assert pbr._pbar.pbar.maximum() == pbr.total == 100


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


def test_progress_set_description(make_napari_viewer):
    make_napari_viewer()

    pbr = progress(total=5)
    pbr.set_description("Test")

    assert pbr.desc == "Test: "
    assert pbr._pbar.description_label.text() == "Test: "
