import inspect
from functools import wraps
from typing import Type, Dict, Callable, Any, Set, Optional
from qtpy.QtCore import QObject, QThread, Signal, Slot, QRunnable, QThreadPool
import time


def as_generatorfunction(func: Callable) -> Callable:
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
    yielded = Signal(object)  # emitted with yielded values (if generator used)
    returned = Signal(object)  # emitted with return value
    errored = Signal(object)  # emitted with error object on Exception
    paused = Signal()  # emitted when a running job has successfully paused
    resumed = Signal()  # emitted when a paused job has successfully resumed
    aborted = Signal()  # emitted when a running job is successfully aborted

    def __init__(self, *args, **kwargs) -> None:
        QRunnable.__init__(self)
        QObject.__init__(self)
        self._abort_requested = False

    @Slot()
    def run(self):
        self.started.emit()
        self._running = True
        self.work()
        self.finished.emit()

    def quit(self) -> None:
        """Send a message to abort the worker."""
        self._abort_requested = True

    @property
    def abort_requested(self) -> bool:
        """Whether the worker has been requested to stop."""
        return self._abort_requested

    @property
    def is_running(self) -> bool:
        """Whether the worker has been started"""
        return self._running

    def work(self):
        """Main method to execute the worker.

        Subclasses must implement this method.  Minimally, it should check
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


class Worker(WorkerBase):
    """QRunnable with signals that wraps a long-running function.

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

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        if inspect.isgeneratorfunction(func):
            self._gen = func(*args, **kwargs)
        else:
            self._gen = as_generatorfunction(func)(*args, **kwargs)

        self._incoming_value = None
        self._pause_requested = False
        self._resume_requested = False
        self._paused = False
        self._pause_interval = 0.01
        self._running = False

        first_yield = next(self._gen)
        if isinstance(first_yield, dict):
            self.parse_first_yield(first_yield)

    def parse_first_yield(self, first_yield):
        self.__dict__.update(**first_yield)

    def work(self) -> None:
        """Core loop that calls the runnable.

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
                self.returned.emit(exc.value)
                break
            except Exception as exc:
                self.errored.emit(exc)
                break

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


class ProgressWorker(Worker):
    """A Worker that emits a progress update on each yield.

    See Worker docstring for details.  This simply counts the number of
    iterations, and emits a progress signal on every iteration.
    """

    # emitted on yield ONLY if the function was a generator AND it yielded
    # dict in the first yield with a key "__len__" that defines the number of
    # total yields in the generator.  Will emit an integer between 0 - 100
    # every time a value is yielded
    progress = Signal(int)

    def __init__(self, *args, **kwargs):
        self._counter = 0
        self._length = None
        super().__init__(*args, **kwargs)

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

    def reset_counter(self) -> None:
        self._counter = -1


#: A set of Workers
_WORKERS: Set[WorkerBase] = set()


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


def wait_for_workers_to_quit(msecs: int = -1) -> bool:
    """Ask all workers to quit, and wait up to `msec` for quit.

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


def worker_factory(
    *args,
    start_thread: bool = True,
    connections: Dict[str, Callable] = None,
    worker_class: Type[WorkerBase] = Worker,
    **kwargs,
) -> WorkerBase:
    """Convenience function to start a function in a .

    By default, uses :class:`Worker`, but a custom ``WorkerBase`` subclass may
    be provided.  If so, it must be a subclass of :class:`Worker`, which
    defines a standard set of signals and a run method.

    Example
    -------

    .. code-block:: python

        def long_function(duration):
            import time
            time.sleep(duration)

        worker = worker_factory(long_function, 10)

    """

    if inspect.isclass(worker_class) and issubclass(WorkerBase, worker_class):
        raise TypeError(f'Worker {worker_class} must be a subclass of Worker')

    worker = worker_class(*args, **kwargs)

    if connections:
        if not isinstance(connections, dict):
            raise TypeError("The 'connections' argument must be a dict")

        for key, val in connections.items():
            if not callable(val):
                raise TypeError(
                    f'"connections[{key!r}]" is not a callable function'
                )
            getattr(worker, key).connect(val)

    def start():
        start_worker(worker)

    if start_thread:
        start()
    else:
        worker.start = start
    return worker


def thread_worker(
    function: Optional[Callable] = None,
    worker_class: Type[WorkerBase] = Worker,
) -> Callable:
    """Decorator that runs a function in a seperate thread when called.

    When called, the decorated function returns a :class:`Worker`.  See
    :func:`worker_factory` for additional keyword arguments that can be used
    when calling the function.

    The returned worker will have these signals:

        - *started*: emitted when the work is started
        - *finished*: emitted when the work is finished
        - *yielded*: emitted with yielded values (if generator used)
        - *returned*: emitted with return value
        - *errored*: emitted with error object on Exception
        - *paused*: emitted when a running job has successfully paused
        - *resumed*: emitted when a paused job has successfully resumed
        - *aborted*: emitted when a running job is successfully aborted

    If the decorated function is a generator, the (default) returned worker
    will also provide these methods:

        - *quit*: ask the thread to quit
        - *toggle_paused*: toggle the running state of the thread.
        - *send*: send a value into the generator.  (This requires that your
          decorator function uses the ``value = yield`` syntax)

        # optionally
        - *start*: if you call your function with ``func(start_thread=False)``,
          it will also have a `.start()` method that can be used to start
          execution of the function in another thread.  (useful if you need
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
        worker = long_function(start_thread=False)
        # connect signals if desired
        worker.start()
    """

    def inner_func(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return worker_factory(
                func, *args, worker_class=worker_class, **kwargs
            )

        return wrapper

    if function is None:
        return inner_func
    return inner_func(function)


############################################################################


# This is a variant on the above pattern, it uses QThread instead of Qrunnable
# see https://doc.qt.io/qt-5/threads-technologies.html#comparison-of-solutions
# It provides more flexibility, but requires that the user manage the threads.
# With the runnable pattern above, we leverage the QThreadPool.globalInstance()
# which is wrapped in our own convenience method API.


def new_worker_qthread(
    Worker: Type[QObject],
    *args,
    start_thread: bool = False,
    connections: Dict[str, Callable] = None,
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

    if connections and not isinstance(connections, dict):
        raise TypeError('connections parameter must be a dict')

    thread = QThread()
    worker = Worker(*args, **kwargs)
    worker.moveToThread(thread)
    thread.started.connect(worker.work)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    if connections:
        [getattr(worker, key).connect(val) for key, val in connections.items()]

    if start_thread:
        thread.start()  # sometimes need to connect stuff before starting
    return worker, thread
