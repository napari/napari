from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from napari._tests.utils import skip_local_popups, skip_on_win_ci

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

    from napari import Viewer

    MakeNapariViewer = Callable[..., Viewer]


@skip_local_popups
def test_memory_leaking(
    qtbot: QtBot, make_napari_viewer: MakeNapariViewer
) -> None:
    data = np.zeros((5, 20, 20, 20), dtype=int)
    data[1, 0:10, 0:10, 0:10] = 1
    viewer = make_napari_viewer()
    image = weakref.ref(viewer.add_image(data))
    labels = weakref.ref(viewer.add_labels(data))
    del viewer.layers[0]
    del viewer.layers[0]
    qtbot.wait(100)
    gc.collect()
    gc.collect()
    assert len(viewer.layers) == 0
    assert image() is None
    assert labels() is None


@skip_on_win_ci
@skip_local_popups
def test_leaks_image(
    qtbot: QtBot, make_napari_viewer: MakeNapariViewer
) -> None:
    viewer = make_napari_viewer(show=True)
    lr = weakref.ref(viewer.add_image(np.zeros((10, 10))))
    dr = weakref.ref(lr().data)

    viewer.layers.clear()
    qtbot.wait(100)
    gc.collect()
    gc.collect()
    assert lr() is None
    assert dr() is None


@skip_on_win_ci
@skip_local_popups
def test_leaks_labels(
    qtbot: QtBot, make_napari_viewer: MakeNapariViewer
) -> None:
    viewer = make_napari_viewer(show=True)
    rng = np.random.default_rng(0)
    lr = weakref.ref(
        viewer.add_labels(rng.integers(0, 10, size=(10, 10), dtype=np.int8)),
    )
    dr = weakref.ref(lr().data)
    viewer.layers.clear()
    qtbot.wait(100)
    gc.collect()
    gc.collect()
    assert lr() is None
    assert dr() is None
