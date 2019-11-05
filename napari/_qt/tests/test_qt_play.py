from contextlib import contextmanager

import numpy as np
import pytest

from napari._qt.qt_viewer import QtViewer
from napari.components import ViewerModel

from ...components import Dims
from ..qt_dims import AnimationThread


@contextmanager
def make_thread(
    qtbot, nframes=10, fps=20, frame_range=None, playback_mode='loop'
):
    # sets up an AnimationThread ready for testing, and breaks down when done
    dims = Dims(4)
    nz = 10
    step = 1
    dims.set_range(0, (0, nz, step))
    thread = AnimationThread(
        dims, 0, fps=fps, frame_range=frame_range, playback_mode=playback_mode
    )
    thread._count = 0
    thread.nz = nz

    def bump(*args):
        thread._count += 1

    def count_reached():
        assert thread._count >= nframes

    def go():
        thread.start()
        qtbot.waitUntil(count_reached, timeout=6000)
        thread.timer.stop()
        return thread.current

    thread.incremented.connect(bump)
    thread.go = go

    yield thread
    thread.quit()
    thread.wait()


# Each tuple represents different arguments we will pass to make_thread
# frames, fps, mode, frame_range, expected_result(nframes, nz)
CONDITIONS = [
    # regular nframes < nz
    (5, 20, 'loop', None, lambda x, y: x),
    # loops around to the beginning
    (13, 20, 'loop', None, lambda x, y: x % y),
    # loops correctly with frame_range specified
    (13, 20, 'loop', (2, 6), lambda x, y: x % y),
    # loops correctly going backwards
    (13, -20, 'loop', None, lambda x, y: y - (x % y)),
    # loops back and forth
    (13, 20, 'loop_back_and_forth', None, lambda x, y: x - y + 2),
    # loops back and forth, with negative fps
    # (13, -20, 'loop_back_and_forth', None, lambda x, y: y - (x % y) - 2),
]


@pytest.mark.parametrize("nframes,fps,mode,rng,result", CONDITIONS)
def test_animation_thread_variants(qtbot, nframes, fps, mode, rng, result):
    """This is mostly testing that AnimationThread.advance works as expected"""
    with make_thread(
        qtbot, fps=fps, nframes=nframes, frame_range=rng, playback_mode=mode
    ) as thread:
        current = thread.go()
    if rng:
        nrange = rng[1] - rng[0] + 1
        assert current == rng[0] + result(nframes, nrange)
    else:
        assert current == result(nframes, thread.nz)


def test_animation_thread_once(qtbot):
    """Single shot animation should stop when it reaches the last frame"""
    nframes = 13
    with make_thread(qtbot, nframes=nframes, playback_mode='once') as thread:
        with qtbot.waitSignal(thread.finished):
            thread.start()
    assert thread.current == thread.nz


@pytest.fixture()
def view(qtbot):
    """basic viewer with data that we will use a few times"""
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 10, 15))
    viewer.add_image(data)
    return view


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

    # that's not a valid playback_mode
    with pytest.raises(ValueError):
        view.dims.play(0, 20, playback_mode='enso')
        qtbot.wait(20)
        view.dims.stop()


def test_play_api(qtbot, view):
    """Test that the QtDims.play() function advances a few frames"""
    view.dims._frame = 0

    def increment(e):
        view.dims._frame = e.value  # this is provided by the axis event
        # if we don't "enable play" again, view.dims won't request a new frame
        view.dims._play_ready = True

    view.dims.dims.events.axis.connect(increment)

    view.dims.play(0, 20)
    # wait for the thread to start before timing...
    qtbot.waitSignal(view.dims._animation_thread.timer.timeout, timeout=10000)
    qtbot.wait(370)
    with qtbot.waitSignal(view.dims._animation_thread.finished):
        view.dims.stop()
    A = view.dims._frame
    assert A > 3

    # make sure the stop button actually worked
    qtbot.wait(150)
    assert A == view.dims._frame


def test_playing_hidden_slider_does_nothing(qtbot, view):
    """Make sure playing a dimension without a slider does nothing"""

    def increment(e):
        view.dims._frame = e.value  # this is provided by the axis event
        # if we don't "enable play" again, view.dims won't request a new frame
        view.dims._play_ready = True

    view.dims.dims.events.axis.connect(increment)

    view.dims.play(2, 20)
    assert not view.dims.is_playing
    assert not hasattr(view.dims, '_animation_thread')
