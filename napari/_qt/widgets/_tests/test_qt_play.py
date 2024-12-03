from contextlib import contextmanager
from weakref import ref

import numpy as np
import pytest

from napari._qt.widgets.qt_dims import QtDims
from napari._qt.widgets.qt_dims_slider import AnimationThread
from napari.components import Dims
from napari.settings._constants import LoopMode


@contextmanager
def make_worker(
    qtbot, nframes=8, fps=20, frame_range=None, loop_mode=LoopMode.LOOP
):
    # sets up an AnimationWorker ready for testing, and breaks down when done
    dims = Dims(ndim=4)
    qtdims = QtDims(dims)
    qtbot.addWidget(qtdims)
    nz = 8
    step = 1
    dims.set_range(0, (0, nz - 1, step))
    slider_widget = qtdims.slider_widgets[0]
    slider_widget.loop_mode = loop_mode
    slider_widget.fps = fps
    slider_widget.frame_range = frame_range

    worker = AnimationThread()
    worker.set_slider(slider_widget)
    worker._count = 0
    worker.nz = nz

    def bump(*args):
        if worker._count < nframes:
            worker._count += 1
        else:
            worker._stop()

    def count_reached():
        assert worker._count >= nframes

    def go():
        worker.work()
        qtbot.waitUntil(count_reached, timeout=6000)
        worker._stop()
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


@pytest.mark.slow
@pytest.mark.parametrize(
    ('nframes', 'fps', 'mode', 'rng', 'result'), CONDITIONS
)
def test_animation_thread_variants(qtbot, nframes, fps, mode, rng, result):
    """This is mostly testing that AnimationWorker.advance works as expected"""
    with make_worker(
        qtbot, fps=fps, nframes=nframes, frame_range=rng, loop_mode=mode
    ) as worker:
        current = worker.go()
    if rng:
        nrange = rng[1] - rng[0] + 1
        expected = rng[0] + result(nframes, nrange)
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
            worker.start()
    assert worker.current == worker.nz


@pytest.fixture
def ref_view(make_napari_viewer):
    """basic viewer with data that we will use a few times

    It is problematic to yield the qt_viewer directly as it will stick
    around in the generator frames and causes issues if we want to make sure
    there is only a single instance of QtViewer instantiated at all times during
    the test suite. Thus we yield a weak reference that we resolve immediately
    in the test suite.
    """

    viewer = make_napari_viewer()

    np.random.seed(0)
    data = np.random.random((2, 10, 10, 15))
    viewer.add_image(data)
    yield ref(viewer.window._qt_viewer)
    viewer.close()


def test_play_raises_index_errors(qtbot, ref_view):
    view = ref_view()
    # play axis is out of range
    with pytest.raises(IndexError):
        view.dims.play(5, 20)

    # data doesn't have 20 frames
    with pytest.raises(IndexError):
        view.dims.play(0, 20, frame_range=[2, 20])


def test_play_raises_value_errors(qtbot, ref_view):
    view = ref_view()
    with pytest.raises(ValueError, match='must be <='):
        view.dims.play(0, 20, frame_range=[2, 2])

    with pytest.raises(ValueError, match='loop_mode must be one of'):
        view.dims.play(0, 20, loop_mode=5)


def test_playing_hidden_slider_does_nothing(ref_view):
    """Make sure playing a dimension without a slider does nothing"""

    view = ref_view()

    def increment(e):
        view.dims._frame = e.value  # this is provided by the step event
        # if we don't "enable play" again, view.dims won't request a new frame
        view.dims._play_ready = True

    view.dims.dims.events.current_step.connect(increment)

    with pytest.warns(UserWarning):
        view.dims.play(2, 20)
    view.dims.dims.events.current_step.disconnect(increment)
    assert not view.dims.is_playing


def test_change_play_axis(ref_view, qtbot):
    """Make sure changing the play axis stops the current animation.

    Prior to https://github.com/napari/napari/pull/7158, starting a new play
    animation resulted in QThread warnings and could crash Python in
    some environments. In the future, we may want to allow multiple
    multiple simultaneous play axes [1]_, so this test should be changed
    or removed when we do that.

    ..[1] https://github.com/napari/napari/pull/6300#issuecomment-1757696072
    """
    view = ref_view()
    with qtbot.waitSignal(view.dims._animation_thread.started):
        view.dims.play(0, 20)
    qtbot.waitUntil(lambda: view.dims.is_playing)
    assert view.dims._animation_thread.slider.axis == 0
    view.dims.play(1, 20)
    assert view.dims._animation_thread.slider.axis == 1
    assert view.dims.is_playing
    with qtbot.waitSignal(view.dims._animation_thread.finished):
        view.dims.stop()


def test_change_play_fps(ref_view, qtbot):
    """Make sure changing the play fps stops the current animation"""
    view = ref_view()
    with qtbot.waitSignal(view.dims._animation_thread.started):
        view.dims.play(0, 20)
    qtbot.waitUntil(lambda: view.dims.is_playing)
    assert view.dims._animation_thread.slider.fps == 20
    view.dims.play(0, 30)
    assert view.dims._animation_thread.slider.fps == 30
    assert view.dims.is_playing
    with qtbot.waitSignal(view.dims._animation_thread.finished):
        view.dims.stop()
