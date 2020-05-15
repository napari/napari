import inspect
import re
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Set, Type

import toolz as tz
from qtpy.QtCore import QObject, QRunnable, QThread, QThreadPool, Signal, Slot

#: A set of Workers.  Do not add directly, use ``start_worker``.`
_WORKERS: Set['WorkerBase'] = set()


def as_generator_function(func: Callable) -> Callable:
    """Turns a regular function (single return) into a generator function."""

    @wraps(func)
    def genwrapper(*args, **kwargs):
        yield
        return func(*args, **kwargs)

    return genwrapper


class WorkerBase(QRunnable, QObject):
    """Base class for creating a Worker that can run in another thread."""

    started = Signal()  # emitted when the work is started
    finished = Signal()  # emitted when the work is finished
    returned = Signal(object)  # emitted with return value
    errored = Signal(object)  # emitted with error object on Exception

    def __init__(self, *args, **kwargs) -> None:
        QRunnable.__init__(self)
        QObject.__init__(self)
        self._abort_requested = False
        self._running = False

    def quit(self) -> None:
        """Send a request to abort the worker."""
        self._abort_requested = True

    @property
    def abort_requested(self) -> bool:
        """Whether the worker has been requested to stop."""
        return self._abort_requested

    @property
    def is_running(self) -> bool:
        """Whether the worker has been started"""
        return self._running

    @Slot()
    def run(self):
        """Start the worker.

        The end-user should never need to call this function.
        But it cannot be made private or renamed, since it is called by Qt.

        The order of method calls when starting a worker is:

        .. code-block:: none

           calls start_worker -> QThreadPool.globalInstance().start(worker)
           |               triggered by the QThreadPool.start() method
           |               |             called by worker.run
           |               |             |
           V               V             V
           worker.start -> worker.run -> worker.work

        **This** is the function that actually gets called when calling
        :func:`QThreadPool.start(worker)`.  It simply wraps the :meth:`work`
        method, and emits a few signals.  Subclasses should NOT override this
        method (except with good reason), and instead should implement
        :meth:`work`.
        """
        self.started.emit()
        self._running = True
        try:
            result = self.work()
            self.returned.emit(result)
        except Exception as exc:
            self.errored.emit(exc)
        self.finished.emit()

    def work(self):
        """Main method to execute the worker.

        The end-user should never need to call this function.
        But subclasses must implement this method.  Minimally, it should check
        ``self.abort_requested`` periodically and exit if True.

        Example
        -------

        .. code-block:: python

            def work(self):
                i = 0
                while True:
                    if self.abort_requested:
                        self.aborted.emit()
                        break
                    i += 1
                    if i > max_iters:
                        break
                    time.sleep(0.5)

        """
        raise NotImplementedError(
            f'"{self.__class__.__name__}" failed to define work() method'
        )

    def start(self):
        """Start this worker in a thread and add it to the global threadpool.

        The order of method calls when starting a worker is:

        .. code-block:: none

           calls start_worker -> QThreadPool.globalInstance().start(worker)
           |               triggered by the QThreadPool.start() method
           |               |             called by worker.run
           |               |             |
           V               V             V
           worker.start -> worker.run -> worker.work
        """
        if self in _WORKERS:
            raise RuntimeError('This worker is already started!')

        # This will raise a RunTime error if the worker is already deleted
        repr(self)
        start_worker(self)


class FunctionWorker(WorkerBase):
    """QRunnable with signals that wraps a simple long-running function."""

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        if inspect.isgeneratorfunction(func):
            raise TypeError(
                f"Generator function {func} cannot be used with "
                "FunctionWorker, use GeneratorWorker instead"
            )

        self._func = func
        self._args = args
        self._kwargs = kwargs

    def work(self):
        return self._func(*self._args, **self._kwargs)


class GeneratorWorker(WorkerBase):
    """QRunnable with signals that wraps a long-running generator.

    When combined with :func:`new_worker_qthread`, provides a convenient way
    to run a function in another thread, while allowing 2-way communication
    between threads, using plain-python generator syntax in the original
    function.

    Parameters
    ----------
    func : callable
        The function being run in another thread.  May be a generator function.
    *args
        Will be passed to func on instantiation
    **kwargs
        Will be passed to func on instantiation
    """

    yielded = Signal(object)  # emitted with yielded values (if generator used)
    paused = Signal()  # emitted when a running job has successfully paused
    resumed = Signal()  # emitted when a paused job has successfully resumed
    aborted = Signal()  # emitted when a running job is successfully aborted

    def __init__(self, func: Callable, *args, init_yield=False, **kwargs):
        super().__init__()
        if not inspect.isgeneratorfunction(func):
            raise TypeError(
                f"Regular function {func} cannot be used with "
                "GeneratorWorker, use FunctionWorker instead"
            )

        self._gen = func(*args, **kwargs)
        self._incoming_value = None
        self._pause_requested = False
        self._resume_requested = False
        self._paused = False
        self._pause_interval = 0.01

        if init_yield:
            first_yield = next(self._gen)
            if isinstance(first_yield, dict):
                self.parse_first_yield(first_yield)

    def parse_first_yield(self, first_yield):
        self.__dict__.update(**first_yield)

    def work(self) -> None:
        """Core loop that calls the original function.  Enters a continual
        loop, yielding and returning from the original function.  Checks for
        various events (quit, pause, resume, etc...)
        """
        while True:
            if self.abort_requested:
                self.aborted.emit()
                break
            if self._paused:
                if self._resume_requested:
                    self._paused = False
                    self._resume_requested = False
                    self.resumed.emit()
                else:
                    time.sleep(self._pause_interval)
                    continue
            elif self._pause_requested:
                self._paused = True
                self._pause_requested = False
                self.paused.emit()
                continue
            try:
                self.pre_yield_hook()
                self.yielded.emit(self._gen.send(self._next_value()))
                self.post_yield_hook()
            except StopIteration as exc:
                return exc.value

    def send(self, value: Any):
        """Send a value into the function (if a generator was used)."""
        self._incoming_value = value

    def _next_value(self) -> Any:
        out = None
        if self._incoming_value is not None:
            out = self._incoming_value
            self._incoming_value = None
        return out

    @property
    def is_paused(self) -> bool:
        """Whether the worker is currently paused."""
        return self._paused

    def toggle_pause(self) -> None:
        """Send a request to pause the worker if playing or resumed if paused.
        """
        if self.is_paused:
            self._resume_requested = True
        else:
            self._pause_requested = True

    def pre_yield_hook(self):
        """Hook for subclasses. Called just before yielding from generator"""
        pass

    def post_yield_hook(self):
        """Hook for subclasses. Called just after yielding from generator"""
        pass


class ProgressWorker(GeneratorWorker):
    """A Worker that emits a progress update on each yield.

    See Worker docstring for details.  This simply counts the number of
    iterations, and emits a progress signal on every iteration.
    """

    # Will emit an integer between 0 - 100 every time a value is yielded
    progress = Signal(int)

    def __init__(self, func, *args, **kwargs):
        self._counter = 0
        self._length = None

        # look in the source code of the function for a yield statement
        # that yields a dict with a key named "__len__"
        # Note: this does NOT assert that it's actually the FIRST yield.
        # (That's harder to do, because the word "yield" may appear in the
        # docstring, etc...)
        source = inspect.getsource(func)
        match = re.search(
            r'yield\s\{([^}]*[\'"]__len__[\'"].*)\}', source, flags=re.DOTALL
        )
        if not match:
            raise ValueError(
                "ProgressWorker may only be used with a generator function "
                "whose first yield expression yields a dict with the key "
                "'__len__'"
            )

        super().__init__(func, *args, init_yield=True, **kwargs)

    def parse_first_yield(self, result):
        self._length = result.pop("__len__", None)
        if self._length is not None:
            if not isinstance(self._length, int) and self._length > 0:
                raise ValueError(
                    "If providing __len__, it must be a positive int"
                )
        super().parse_first_yield(result)

    def pre_yield_hook(self):
        self._counter += 1
        if self._length:
            self.progress.emit(round(100 * (self._counter - 1) / self._length))

    def set_counter(self, val) -> None:
        self._counter = val

    def __len__(self):
        if self._length is not None:
            return self._length
        raise TypeError("ProgressWorker was not provided with a length.")


############################################################################

# public API

# For now, the next three functions simply wrap the QThreadPool API, and allow
# us to track and cleanup all workers that were started with ``start_worker``,
# provided that ``wait_for_workers_to_quit`` is called at shutdown.
# In the future, this could wrap any API, or a pure python threadpool.


def start_worker(worker: WorkerBase) -> None:
    """Add a worker instance to the global ThreadPool and start it.

    Parameters
    ----------
    worker : WorkerBase
        A Worker instance to start and add to the threadpool.

    Raises
    ------
    TypeError
        If ``worker`` is not an instance of ``WorkerBase``.
    """
    if not isinstance(worker, WorkerBase):
        raise TypeError(
            'Using the `start_worker` API requires the the worker '
            'object be an instance of Worker'
        )
    _WORKERS.add(worker)
    worker.finished.connect(lambda: _WORKERS.discard(worker))
    QThreadPool.globalInstance().start(worker)


def set_max_thread_count(num: int):
    """Set the maximum number of threads used by the thread pool.

    Note: The thread pool will always use at least 1 thread, even if
    maxThreadCount limit is zero or negative.
    """
    QThreadPool.globalInstance().setMaxThreadCount(num)


def wait_for_workers_to_quit(msecs: int = -1) -> bool:
    """Ask all workers to quit, and wait up to `msec` for quit.

    Attempts to clean up all running workers.  (It is assumed that they have a
    ``quit()`` method, which they will if ``start_worker`` was used.)

    Parameters
    ----------
    msecs : int, optional
        Waits up to msecs milliseconds for all threads to exit and removes all
        threads from the thread pool. If msecs is -1 (the default), the timeout
        is ignored (waits for the last thread to exit).

    Returns
    -------
    bool
        True if all threads were removed; otherwise False.
    """
    for worker in _WORKERS:
        worker.quit()
    return QThreadPool.globalInstance().waitForDone(msecs)


def active_thread_count() -> int:
    """Return the number of active threads in the global ThreadPool."""
    return QThreadPool.globalInstance().activeThreadCount()


#############################################################################

# convenience functions for creating Worker instances


def create_worker(
    func: Callable,
    *args,
    _start_thread: Optional[bool] = None,
    _connect: Optional[Dict[str, Callable]] = None,
    _worker_class: Optional[Type[WorkerBase]] = None,
    _ignore_errors: bool = False,
    **kwargs,
) -> WorkerBase:
    """Convenience function to start a function in another thread.

    By default, uses :class:`Worker`, but a custom ``WorkerBase`` subclass may
    be provided.  If so, it must be a subclass of :class:`Worker`, which
    defines a standard set of signals and a run method.

    Parameters
    ----------
    func : Callable
        The function to call in another thread.
    _start_thread : bool, optional
        Whether to immediaetly start the thread.  If False, the returned worker
        must be manually started with ``worker.start()``. by default it will be
        ``False`` if the ``_connect`` argument is ``None``, otherwise ``True``.
    _connect : Dict[str, Callable], optional
        A mapping of ``"signal_name"`` -> ``callable``: callback functions to
        connect to the various signals offered by the worker class.
        by default None
    _worker_class : Type[WorkerBase], optional
        The :class`WorkerBase` to instantiate, by default
        :class:`FunctionWorker` will be used if ``func`` is a regular function,
        and :class:`GeneratorWorker` will be used if it is a generator.
    *args
        will be passed to ``func``
    **kwargs
        will be passed to ``func``

    Example
    -------

    .. code-block:: python

        def long_function(duration):
            import time
            time.sleep(duration)

        worker = create_worker(long_function, 10)

    Returns
    -------
    WorkerBase
        An instantiated worker.  If ``_start_thread`` was ``False``, the worker
        will have a `.start()` method that can be used to start the thread.

    Raises
    ------
    TypeError
        If a worker_class is provided that is not a subclass of WorkerBase.
    TypeError
        If _connect is provided and is not a dict of ``{str: callable}``
    """
    if not _worker_class:
        if inspect.isgeneratorfunction(func):
            _worker_class = GeneratorWorker
        else:
            _worker_class = FunctionWorker

    if inspect.isclass(_worker_class) and issubclass(
        WorkerBase, _worker_class
    ):
        raise TypeError(f'Worker {_worker_class} must be a subclass of Worker')

    worker = _worker_class(func, *args, **kwargs)

    if _connect is not None:
        if not isinstance(_connect, dict):
            raise TypeError("The '_connect' argument must be a dict")

        if _start_thread is None:
            _start_thread = True

        for key, val in _connect.items():
            if not callable(val):
                raise TypeError(
                    f'"_connect[{key!r}]" is not a callable function'
                )
            getattr(worker, key).connect(val)

    # if the user has not provided a default connection for the "errored"
    # signal... and they have not explicitly set ``ignore_errors=True``
    # Then rereaise any errors from the thread.
    if not _ignore_errors and not (_connect or {}).get('errored', False):

        def reraise(e):
            raise e

        worker.errored.connect(reraise)

    if _start_thread:
        worker.start()
    return worker


@tz.curry
def thread_worker(
    function: Callable,
    start_thread: Optional[bool] = None,
    connect: Optional[Dict[str, Callable]] = None,
    worker_class: Optional[Type[WorkerBase]] = None,
    ignore_errors: bool = False,
) -> Callable:
    """Decorator that runs a function in a seperate thread when called.

    When called, the decorated function returns a :class:`WorkerBase`.  See
    :func:`create_worker` for additional keyword arguments that can be used
    when calling the function.

    The returned worker will have these signals:

        - *started*: emitted when the work is started
        - *finished*: emitted when the work is finished
        - *returned*: emitted with return value
        - *errored*: emitted with error object on Exception

    If the decorated function is a generator, the (default) returned worker
    will also provide these signals:

        - *yielded*: emitted with yielded values (if generator used)
        - *paused*: emitted when a running job has successfully paused
        - *resumed*: emitted when a paused job has successfully resumed
        - *aborted*: emitted when a running job is successfully aborted

    And these methods:

        - *quit*: ask the thread to quit
        - *toggle_paused*: toggle the running state of the thread.
        - *send*: send a value into the generator.  (This requires that your
          decorator function uses the ``value = yield`` syntax)

    If you call your function with ``func(_start_thread=False)`` (the default)

        - *start*: it will also have a ``.start()`` method that can be used to
        start execution of the function in another thread.  (useful if you need
        to connect callbacks to signals prior to execution)

    Parameters
    ----------
    func : callable
        Function to call in another thread.  For communication between threads
        may be a generator function.

    Returns
    -------
    callable
        function that creates a worker, puts it in a new thread and returns
        the worker instance.

    Example
    -------

    .. code-block:: python

        @thread_worker
        def long_function(start, end):
            # do setup here.
            yield # Critical to yield before computation when using generator

            # do work, periodically yielding
            i = start
            while i <= end:
                time.sleep(0.1)
                yield i

            # do teardown
            return 'anything'

        # call the function to start running in another thread.
        worker = long_function(_start_thread=False)
        # connect signals if desired
        worker.start()
    """

    @wraps(function)
    def worker_function(*args, **kwargs):
        # decorator kwargs can be overridden at call time by using the
        # underscore-prefixed version of the kwarg.
        kwargs['_start_thread'] = kwargs.get('_start_thread', start_thread)
        kwargs['_connect'] = kwargs.get('_connect', connect)
        kwargs['_worker_class'] = kwargs.get('_worker_class', worker_class)
        kwargs['_ignore_errors'] = kwargs.get('_ignore_errors', ignore_errors)
        return create_worker(function, *args, **kwargs,)

    return worker_function


############################################################################

# This is a variant on the above pattern, it uses QThread instead of Qrunnable
# see https://doc.qt.io/qt-5/threads-technologies.html#comparison-of-solutions
# It provides more flexibility, but requires that the user manage the threads.
# With the runnable pattern above, we leverage the QThreadPool.globalInstance()
# which is wrapped in our own convenience method API.


def new_worker_qthread(
    Worker: Type[QObject],
    *args,
    _start_thread: bool = False,
    _connect: Dict[str, Callable] = None,
    **kwargs,
):
    """This is a convenience function to start a worker in a Qthread.

    In most cases, the @thread_worker decorator is sufficient and preferable.
    But this allows the user to completely customize the Worker object.
    However, they must then maintain control over the thread and clean up
    appropriately.

    It follows the pattern described here:
    https://www.qt.io/blog/2010/06/17/youre-doing-it-wrong
    and
    https://doc.qt.io/qt-5/qthread.html#details

    see also:
    https://mayaposch.wordpress.com/2011/11/01/how-to-really-truly-use-qthreads-the-full-explanation/

    A QThread object is not a thread! It should be thought of as a class to
    *manage* a thread, not as the actual code or object that runs in that
    thread.  The QThread object is created on the main thread and lives there.

    Worker objects which derive from QObject are the things that actually do
    the work. They can be moved to a QThread as is done here.

    .. note:: Mostly ignorable detail

        While the signals/slots syntax of the worker looks very similar to
        standard "single-threaded" signals & slots, note that inter-thread
        signals and slots (automatically) use an event-based QueuedConnection,
        while intra-thread signals use a DirectConnection. See `Signals and
        Slots Across Threads
        <https://doc.qt.io/qt-5/threads-qobject.html#signals-and-slots-across-threads>`_


    Parameters
    ----------
    Worker : QObject
        QObject type that implements a work() method.  The Worker should also
        emit a finished signal when the work is done.
    start_thread : bool
        If True, thread will be started immediately, otherwise, thread must
        be manually started with thread.start().
    connections: dict, optional
        Optional dictionary of {signal: function} to connect to the new worker.
        for instance:  connections = {'incremented': myfunc} will result in:
        worker.incremented.connect(myfunc)
    *args
        will be passed to the Worker class on instantiation.
    **kwargs
        will be passed to the Worker class on instantiation.

    Examples
    --------
    Create some QObject that has a long-running work method:

    .. code-block:: python

        class Worker(QObject):

            finished = Signal()
            increment = Signal(int)

            def __init__(self, argument):
                super().__init__()
                self.argument = argument

            @Slot()
            def work(self):
                # some long running task...
                import time
                for i in range(10):
                    time.sleep(1)
                    self.increment.emit(i)
                self.finished.emit()

        worker, thread = new_worker_qthread(
            Worker,
            'argument',
            start_thread=True,
            connections={'increment': print},
        )

    """

    if _connect and not isinstance(_connect, dict):
        raise TypeError('_connect parameter must be a dict')

    thread = QThread()
    worker = Worker(*args, **kwargs)
    worker.moveToThread(thread)
    thread.started.connect(worker.work)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    if _connect:
        [getattr(worker, key).connect(val) for key, val in _connect.items()]

    if _start_thread:
        thread.start()  # sometimes need to connect stuff before starting
    return worker, thread
