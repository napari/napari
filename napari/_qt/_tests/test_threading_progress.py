import pytest

from napari._qt import qthreading
from napari._qt.widgets.qt_progress_bar import QtLabeledProgressBar

pytest.importorskip(
    'qtpy', reason='Cannot test threading progress without qtpy.'
)


def test_worker_with_progress(qtbot):
    test_val = [0]

    def func():
        yield 1
        yield 1

    def test_yield(v):
        test_val[0] += 1

    thread_func = qthreading.thread_worker(
        func,
        connect={'yielded': test_yield},
        progress={'total': 2},
        start_thread=False,
    )
    worker = thread_func()
    with qtbot.waitSignals([worker.yielded, worker.finished]):
        worker.start()
        assert worker.pbar.n == test_val[0]
    assert test_val[0] == 2


def test_function_worker_nonzero_total_warns():
    def not_a_generator():
        return

    with pytest.warns(RuntimeWarning):
        thread_func = qthreading.thread_worker(
            not_a_generator,
            progress={'total': 2},
            start_thread=False,
        )
        thread_func()


def test_worker_may_exceed_total(qtbot):
    test_val = [0]

    def func():
        yield 1
        yield 1

    def test_yield(v):
        test_val[0] += 1
        if test_val[0] < 2:
            assert worker.pbar.n == test_val[0]
        else:
            assert worker.pbar.total == 0

    thread_func = qthreading.thread_worker(
        func,
        progress={'total': 1},
        start_thread=False,
    )
    worker = thread_func()
    worker.yielded.connect(test_yield)
    with qtbot.waitSignals([worker.yielded, worker.finished]):
        worker.start()
    assert test_val[0] == 2


def test_generator_worker_with_description():
    def func():
        yield 1

    thread_func = qthreading.thread_worker(
        func,
        progress={'total': 1, 'desc': 'custom'},
        start_thread=False,
    )
    worker = thread_func()
    assert worker.pbar.desc == 'custom'


def test_function_worker_with_description():
    def func():
        for _ in range(10):
            pass

    thread_func = qthreading.thread_worker(
        func,
        progress={'total': 0, 'desc': 'custom'},
        start_thread=False,
    )
    worker = thread_func()
    assert worker.pbar.desc == 'custom'


def test_generator_worker_with_no_total():
    def func():
        yield 1

    thread_func = qthreading.thread_worker(
        func,
        progress=True,
        start_thread=False,
    )
    worker = thread_func()
    assert worker.pbar.total == 0


def test_function_worker_with_no_total():
    def func():
        for _ in range(10):
            pass

    thread_func = qthreading.thread_worker(
        func,
        progress=True,
        start_thread=False,
    )
    worker = thread_func()
    assert worker.pbar.total == 0


def test_function_worker_0_total():
    def func():
        for _ in range(10):
            pass

    thread_func = qthreading.thread_worker(
        func,
        progress={'total': 0},
        start_thread=False,
    )
    worker = thread_func()
    assert worker.pbar.total == 0


def test_unstarted_worker_no_widget(make_napari_viewer):
    viewer = make_napari_viewer()

    def func():
        for _ in range(5):
            yield

    thread_func = qthreading.thread_worker(
        func,
        progress={'total': 5},
        start_thread=False,
    )

    thread_func()
    assert not bool(
        viewer.window._qt_viewer.window()._activity_dialog.findChildren(
            QtLabeledProgressBar
        )
    )
