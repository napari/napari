from contextlib import contextmanager

import numpy as np
import pytest

from napari._qt._constants import LoopMode
from napari._qt.widgets.qt_dims import QtDims
from napari._qt.widgets.qt_dims_slider import AnimationWorker
from napari.components import Dims


@contextmanager
def make_worker(
    qtbot, nframes=8, fps=20, frame_range=None, loop_mode=LoopMode.LOOP
):
    # sets up an AnimationWorker ready for testing, and breaks down when done
    dims = Dims(ndim=4)
    qtdims = QtDims(dims)
    qtbot.addWidget(qtdims)
    nz = 8
    max_index = nz - 1
    step = 1
    dims.set_range(0, (0, max_index, step))
    slider_widget = qtdims.slider_widgets[0]
    slider_widget.loop_mode = loop_mode
    slider_widget.fps = fps
    slider_widget.frame_range = frame_range

    worker = AnimationWorker(slider_widget)
    worker._count = 0
    worker.nz = nz

    def bump(*args):
        if worker._count < nframes:
            worker._count += 1
        else:
            worker.finish()

    def count_reached():
        assert worker._count >= nframes

    def go():
        worker.work()
        qtbot.waitUntil(count_reached, timeout=6000)
        return worker.current

    worker.frame_requested.connect(bump)
    worker.go = go

    yield worker


# Each tuple represents different arguments we will pass to make_thread
# frames, fps, mode, frame_range, expected_result(nframes, nz)
CONDITIONS = [
    # regular nframes < nz
    (5, 10, LoopMode.LOOP, None, lambda x, y: x),
    # loops around to the beginning
    (10, 10, LoopMode.LOOP, None, lambda x, y: x % y),
    # loops correctly with frame_range specified
    (10, 10, LoopMode.LOOP, (2, 6), lambda x, y: x % y),
    # loops correctly going backwards
    (10, -10, LoopMode.LOOP, None, lambda x, y: y - (x % y)),
    # loops back and forth
    (10, 10, LoopMode.BACK_AND_FORTH, None, lambda x, y: x - y + 2),
    # loops back and forth, with negative fps
    (10, -10, LoopMode.BACK_AND_FORTH, None, lambda x, y: y - (x % y) - 2),
]


@pytest.mark.parametrize("nframes,fps,mode,rng,result", CONDITIONS)
def test_animation_thread_variants(qtbot, nframes, fps, mode, rng, result):
    """This is mostly testing that AnimationWorker.advance works as expected"""
    with make_worker(
        qtbot, fps=fps, nframes=nframes, frame_range=rng, loop_mode=mode
    ) as worker:
        current = worker.go()
    if rng:
        nrange = rng[1] - rng[0] + 1
        expected = rng[0] + result(nframes, nrange)
        assert expected - 1 <= current <= expected + 1
    else:
        expected = result(nframes, worker.nz)
        # assert current == expected
        # relaxing for CI OSX tests
        assert expected - 1 <= current <= expected + 1


def test_animation_thread_once(qtbot):
    """Single shot animation should stop when it reaches the last frame"""
    nframes = 13
    with make_worker(
        qtbot, nframes=nframes, loop_mode=LoopMode.ONCE
    ) as worker:
        with qtbot.waitSignal(worker.finished, timeout=8000):
            worker.work()
    assert worker.current == worker.nz


@pytest.fixture()
def view(make_napari_viewer):
    """basic viewer with data that we will use a few times"""
    viewer = make_napari_viewer()

    np.random.seed(0)
    data = np.random.random((10, 10, 15))
    viewer.add_image(data)

    return viewer.window.qt_viewer


def test_play_raises_index_errors(qtbot, view):
    # play axis is out of range
    with pytest.raises(IndexError):
        view.dims.play(4, 20)
        qtbot.wait(20)
        view.dims.stop()

    # data doesn't have 20 frames
    with pytest.raises(IndexError):
        view.dims.play(0, 20, frame_range=[2, 20])
        qtbot.wait(20)
        view.dims.stop()


def test_play_raises_value_errors(qtbot, view):
    # frame_range[1] not > frame_range[0]
    with pytest.raises(ValueError):
        view.dims.play(0, 20, frame_range=[2, 2])
        qtbot.wait(20)
        view.dims.stop()

    # that's not a valid loop_mode
    with pytest.raises(ValueError):
        view.dims.play(0, 20, loop_mode=5)
        qtbot.wait(20)
        view.dims.stop()


@pytest.mark.skip(reason="fails too often... tested indirectly elsewhere")
def test_play_api(qtbot, view):
    """Test that the QtDims.play() function advances a few frames"""
    view.dims._frame = 0

    def increment(e):
        view.dims._frame += 1
        # if we don't "enable play" again, view.dims won't request a new frame
        view.dims._play_ready = True

    view.dims.dims.events.current_step.connect(increment)

    view.dims.play(0, 20)
    # wait for the thread to start before timing...
    qtbot.waitSignal(view.dims._animation_thread.started, timeout=10000)
    qtbot.wait(370)
    with qtbot.waitSignal(view.dims._animation_thread.finished, timeout=7000):
        view.dims.stop()
    A = view.dims._frame
    assert A >= 3

    # make sure the stop button actually worked
    qtbot.wait(150)
    assert A == view.dims._frame


def test_playing_hidden_slider_does_nothing(view):
    """Make sure playing a dimension without a slider does nothing"""

    def increment(e):
        view.dims._frame = e.value  # this is provided by the step event
        # if we don't "enable play" again, view.dims won't request a new frame
        view.dims._play_ready = True

    view.dims.dims.events.current_step.connect(increment)

    with pytest.warns(UserWarning):
        view.dims.play(2, 20)
    assert not view.dims.is_playing
