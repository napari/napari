import inspect
import time

import pytest

from napari._qt import qthreading


def test_as_generator_function():
    """Test we can convert a regular function to a generator function."""

    def func():
        return

    assert not inspect.isgeneratorfunction(func)

    newfunc = qthreading.as_generator_function(func)
    assert inspect.isgeneratorfunction(newfunc)
    assert list(newfunc()) == [None]


# qtbot is necessary for qthreading here.
# note: pytest-cov cannot check coverage of code run in the other thread.
def test_thread_worker(qtbot):
    """Test basic threadworker on a function"""

    def func():
        return 1

    def test(v):
        assert v == 1

    thread_func = qthreading.thread_worker(func, connect={'returned': test})
    worker = thread_func()
    assert isinstance(worker, qthreading.FunctionWorker)


def test_thread_generator_worker(qtbot):
    """Test basic threadworker on a generator"""

    def func():
        yield 1
        yield 1
        return 3

    def test_return(v):
        assert v == 3

    def test_yield(v):
        assert v == 1

    thread_func = qthreading.thread_worker(
        func, connect={'returned': test_return, 'yielded': test_yield}
    )
    worker = thread_func()
    assert isinstance(worker, qthreading.GeneratorWorker)


def test_thread_raises(qtbot):
    """Test exceptions get returned to main thread"""

    def func():
        yield 1
        yield 1
        raise ValueError('whoops')

    def handle_raise(e):
        assert isinstance(e, ValueError)
        assert str(e) == 'whoops'

    thread_func = qthreading.thread_worker(
        func, connect={'errored': handle_raise}
    )
    worker = thread_func()
    assert isinstance(worker, qthreading.GeneratorWorker)


def test_multiple_connections(qtbot):
    """Test the connect dict accepts a list of functions, and type checks"""

    def func():
        return 1

    def test1(v):
        assert v == 1

    def test2(v):
        assert v == 1

    thread_func = qthreading.thread_worker(
        func, connect={'returned': [test1, test2]}
    )
    worker = thread_func()
    assert isinstance(worker, qthreading.FunctionWorker)

    # they must all be functions
    with pytest.raises(TypeError):
        qthreading.thread_worker(
            func, connect={'returned': ['test1', test2]}
        )()

    # they must all be functions
    with pytest.raises(TypeError):
        qthreading.thread_worker(func, connect=test1)()


def test_create_worker():
    """Test directly calling create_worker."""

    def func(x, y):
        return x + y

    worker = qthreading.create_worker(func, 1, 2)
    assert isinstance(worker, qthreading.WorkerBase)

    with pytest.raises(TypeError):
        _ = qthreading.create_worker(func, 1, 2, _worker_class=object)


# note: pytest-cov cannot check coverage of code run in the other thread.
# this is just for the sake of coverage
def test_thread_worker_in_main_thread():
    """Test basic threadworker on a function"""

    def func(x):
        return x

    thread_func = qthreading.thread_worker(func)
    worker = thread_func(2)
    # NOTE: you shouldn't normally call worker.work()!  If you do, it will NOT
    # be run in a separate thread (as it would for worker.start().
    # This is for the sake of testing it in the main thread.
    assert worker.work() == 2


# note: pytest-cov cannot check coverage of code run in the other thread.
# this is just for the sake of coverage
def test_thread_generator_worker_in_main_thread():
    """Test basic threadworker on a generator in the main thread with methods.
    """

    def func():
        i = 0
        while i < 10:
            i += 1
            incoming = yield i
            i = incoming if incoming is not None else i
        return 3

    worker = qthreading.thread_worker(func, start_thread=False)()
    counter = 0

    def handle_pause():
        time.sleep(0.1)
        assert worker.is_paused
        worker.toggle_pause()

    def test_yield(v):
        nonlocal counter
        counter += 1
        if v == 2:
            assert not worker.is_paused
            worker.pause()
            assert not worker.is_paused
        if v == 3:
            worker.send(7)
        if v == 9:
            worker.quit()

    def handle_abort():
        assert counter == 5  # because we skipped a few by sending in 7

    worker.paused.connect(handle_pause)
    assert isinstance(worker, qthreading.GeneratorWorker)
    worker.yielded.connect(test_yield)
    worker.aborted.connect(handle_abort)
    # NOTE: you shouldn't normally call worker.work()!  If you do, it will NOT
    # be run in a separate thread (as it would for worker.start().
    # This is for the sake of testing it in the main thread.
    assert worker.work() is None  # because we aborted it
    assert not worker.is_paused
    assert counter == 5

    worker2 = qthreading.thread_worker(func, start_thread=False)()
    assert worker2.work() == 3
