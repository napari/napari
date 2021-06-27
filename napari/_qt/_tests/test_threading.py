import inspect
import time
from functools import partial
from operator import eq

import pytest

from napari._qt import qthreading

equals_1 = partial(eq, 1)
equals_3 = partial(eq, 3)
skip = pytest.mark.skipif(True, reason="testing")


@pytest.mark.order(1)
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
@pytest.mark.order(2)
def test_thread_worker(qtbot):
    """Test basic threadworker on a function"""

    @qthreading.thread_worker
    def func():
        return 1

    wrkr = func()
    assert isinstance(wrkr, qthreading.FunctionWorker)

    signals = [wrkr.returned, wrkr.finished]
    checks = [equals_1, lambda: True]
    with qtbot.waitSignals(signals, check_params_cbs=checks, order="strict"):
        wrkr.start()


@pytest.mark.order(3)
def test_thread_generator_worker(qtbot):
    """Test basic threadworker on a generator"""

    @qthreading.thread_worker
    def func():
        yield 1
        yield 1
        return 3

    wrkr = func()
    assert isinstance(wrkr, qthreading.GeneratorWorker)

    signals = [wrkr.yielded, wrkr.yielded, wrkr.returned, wrkr.finished]
    checks = [equals_1, equals_1, equals_3, lambda: True]
    with qtbot.waitSignals(signals, check_params_cbs=checks, order="strict"):
        wrkr.start()

    qtbot.wait(500)


@pytest.mark.order(4)
def test_thread_raises2(qtbot):
    handle_val = [0]

    def handle_raise(e):
        handle_val[0] = 1
        assert isinstance(e, ValueError)
        assert str(e) == 'whoops'

    @qthreading.thread_worker(
        connect={'errored': handle_raise}, start_thread=False
    )
    def func():
        yield 1
        yield 1
        raise ValueError('whoops')

    wrkr = func()
    assert isinstance(wrkr, qthreading.GeneratorWorker)

    signals = [wrkr.yielded, wrkr.yielded, wrkr.errored, wrkr.finished]
    checks = [equals_1, equals_1, None, None]
    with qtbot.waitSignals(signals, check_params_cbs=checks):
        wrkr.start()
    assert handle_val[0] == 1


@pytest.mark.order(5)
def test_thread_warns(qtbot):
    """Test warnings get returned to main thread"""
    import warnings

    def check_warning(w):
        return str(w) == 'hey!'

    @qthreading.thread_worker(
        connect={'warned': check_warning}, start_thread=False
    )
    def func():
        yield 1
        warnings.warn('hey!')
        yield 3
        warnings.warn('hey!')
        return 1

    wrkr = func()
    assert isinstance(wrkr, qthreading.GeneratorWorker)

    signals = [wrkr.yielded, wrkr.warned, wrkr.yielded, wrkr.returned]
    checks = [equals_1, None, equals_3, equals_1]
    with qtbot.waitSignals(signals, check_params_cbs=checks):
        wrkr.start()


@pytest.mark.order(6)
def test_multiple_connections(qtbot):
    """Test the connect dict accepts a list of functions, and type checks"""

    test1_val = [0]
    test2_val = [0]

    def func():
        return 1

    def test1(v):
        test1_val[0] = 1
        assert v == 1

    def test2(v):
        test2_val[0] = 1
        assert v == 1

    thread_func = qthreading.thread_worker(
        func, connect={'returned': [test1, test2]}, start_thread=False
    )
    worker = thread_func()
    assert isinstance(worker, qthreading.FunctionWorker)
    with qtbot.waitSignal(worker.finished):
        worker.start()

    assert test1_val[0] == 1
    assert test2_val[0] == 1

    # they must all be functions
    with pytest.raises(TypeError):
        qthreading.thread_worker(
            func, connect={'returned': ['test1', test2]}
        )()

    # they must all be functions
    with pytest.raises(TypeError):
        qthreading.thread_worker(func, connect=test1)()


@pytest.mark.order(7)
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
@pytest.mark.order(8)
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
@pytest.mark.order(9)
def test_thread_generator_worker_in_main_thread():
    """Test basic threadworker on a generator in the main thread with methods."""

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


@pytest.mark.order(10)
def test_worker_base_attribute():
    obj = qthreading.WorkerBase()
    assert obj.started is not None
    assert obj.finished is not None
    assert obj.returned is not None
    assert obj.errored is not None
    with pytest.raises(AttributeError):
        obj.aa


@pytest.mark.order(11)
def test_abort_does_not_return(qtbot):
    loop_counter = 0

    def long_running_func():
        nonlocal loop_counter
        import time

        for i in range(5):
            yield loop_counter
            time.sleep(0.1)
            loop_counter += 1

    abort_counter = 0

    def count_abort():
        nonlocal abort_counter
        abort_counter += 1

    return_counter = 0

    def returned_handler(value):
        nonlocal return_counter
        return_counter += 1

    threaded_function = qthreading.thread_worker(
        long_running_func,
        connect={
            'returned': returned_handler,
            'aborted': count_abort,
        },
    )
    worker = threaded_function()
    worker.quit()
    qtbot.wait(600)
    assert loop_counter < 4
    assert abort_counter == 1
    assert return_counter == 0
