from napari._qt import threading
import inspect
import pytest


def test_as_generator_function():
    def func():
        return

    assert not inspect.isgeneratorfunction(func)

    newfunc = threading.as_generator_function(func)
    assert inspect.isgeneratorfunction(newfunc)
    assert list(newfunc()) == [None]


# qtbot is necessary for qthreading here.
# note: pytest-cov cannot check coverage of code run in the other thread.
def test_thread_worker(qtbot):
    """test basic threadworker on a function"""

    def func():
        return 1

    def test(v):
        assert v == 1

    thread_func = threading.thread_worker(func, connect={'returned': test})
    worker = thread_func()
    assert isinstance(worker, threading.FunctionWorker)


def test_thread_generator_worker(qtbot):
    """test basic threadworker on a generator"""

    def func():
        yield 1
        yield 1
        return 3

    def test_return(v):
        assert v == 3

    def test_yield(v):
        assert v == 1

    thread_func = threading.thread_worker(
        func, connect={'returned': test_return, 'yielded': test_yield}
    )
    worker = thread_func()
    assert isinstance(worker, threading.GeneratorWorker)


def test_thread_raises(qtbot):
    """test exceptions get returned to main thread"""

    def func():
        yield 1
        yield 1
        raise ValueError('whoops')

    def handle_raise(e):
        assert isinstance(e, ValueError)
        assert str(e) == 'whoops'

    thread_func = threading.thread_worker(
        func, connect={'errored': handle_raise}
    )
    worker = thread_func()
    assert isinstance(worker, threading.GeneratorWorker)


def test_multiple_connections(qtbot):
    """test the connect dict accepts a list of functions"""

    def func():
        return 1

    def test1(v):
        assert v == 1

    def test2(v):
        assert v == 1

    thread_func = threading.thread_worker(
        func, connect={'returned': [test1, test2]}
    )
    worker = thread_func()
    assert isinstance(worker, threading.FunctionWorker)

    # they must all be functions
    with pytest.raises(TypeError):
        threading.thread_worker(func, connect={'returned': ['test1', test2]})()

    # they must all be functions
    with pytest.raises(TypeError):
        threading.thread_worker(func, connect=test1)()


def test_create_worker():
    def func(x, y):
        return x + y

    worker = threading.create_worker(func, 1, 2)
    assert isinstance(worker, threading.WorkerBase)

    with pytest.raises(TypeError):
        _ = threading.create_worker(func, 1, 2, _worker_class=object)


# note: pytest-cov cannot check coverage of code run in the other thread.
# this is just for the sake of coverage
def test_thread_worker_in_main_thread():
    """test basic threadworker on a function"""

    def func(x):
        return x

    thread_func = threading.thread_worker(func)
    worker = thread_func(2)
    assert worker.work() == 2


# note: pytest-cov cannot check coverage of code run in the other thread.
# this is just for the sake of coverage
def test_thread_generator_worker_in_main_thread(qtbot):
    """test basic threadworker on a generator"""

    def func():
        yield 1
        yield 1
        return 3

    def test_yield(v):
        assert v == 1

    worker = threading.thread_worker(func, start_thread=False)()
    assert isinstance(worker, threading.GeneratorWorker)
    worker.yielded.connect(test_yield)
    assert worker.work() == 3
    assert not worker.is_paused
