import inspect
import time
import warnings
from functools import partial, wraps
from typing import Any, Callable, Dict, Optional, Sequence, Set, Type, Union

import toolz as tz
from qtpy.QtCore import (
    QObject,
    QRunnable,
    QThread,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)

from ..utils.translations import trans


def as_generator_function(func: Callable) -> Callable:
    """Turns a regular function (single return) into a generator function."""

    @wraps(func)
    def genwrapper(*args, **kwargs):
        yield
        return func(*args, **kwargs)

    return genwrapper


class WorkerBaseSignals(QObject):

    started = Signal()  # emitted when the work is started
    finished = Signal()  # emitted when the work is finished
    returned = Signal(object)  # emitted with return value
    errored = Signal(object)  # emitted with error object on Exception
    warned = Signal(tuple)  # emitted with showwarning args on warning


class WorkerBase(QRunnable):
    """Base class for creating a Worker that can run in another thread.

    Parameters
    ----------
    SignalsClass : type, optional
        A QObject subclass that contains signals, by default WorkerBaseSignals

    Attributes
    ----------
    signals: WorkerBaseSignals
        signal emitter object. To allow identify which worker thread emitted signal.
    """

    #: A set of Workers.  Add to set using :meth:`WorkerBase.start`
    _worker_set: Set['WorkerBase'] = set()

    def __init__(
        self,
        *args,
        SignalsClass: Type[WorkerBaseSignals] = WorkerBaseSignals,
        **kwargs,
    ) -> None:
        super().__init__()
        self._abort_requested = False
        self._running = False
        self.signals = SignalsClass()

        self.signals.errored.connect(self._relay_error)
        self.signals.warned.connect(self._relay_warning)

    def _relay_error(self, exc):
        from ..utils.notifications import notification_manager

        notification_manager.receive_error(type(exc), exc, exc.__traceback__)

    def _relay_warning(self, show_warn_args):
        from ..utils.notifications import notification_manager

        notification_manager.receive_warning(*show_warn_args)

    def __getattr__(self, name):
        """Pass through attr requests to signals to simplify connection API.

        The goal is to enable ``worker.yielded.connect`` instead of
        ``worker.signals.yielded.connect``. Because multiple inheritance of Qt
        classes is not well supported in PyQt, we have to use composition here
        (signals are provided by QObjects, and QRunnable is not a QObject). So
        this passthrough allows us to connect to signals on the ``_signals``
        object.
        """
        # the Signal object is actually a class attribute
        attr = getattr(self.signals.__class__, name, None)
        if isinstance(attr, Signal):
            # but what we need to connect to is the instantiated signal
            # (which is of type `SignalInstance` in PySide and
            # `pyqtBoundSignal` in PyQt)
            return getattr(self.signals, name)
        return super().__getattr__(name)

    def quit(self) -> None:
        """Send a request to abort the worker.

        .. note::

            It is entirely up to subclasses to honor this method by checking
            ``self.abort_requested`` periodically in their ``worker.work``
            method, and exiting if ``True``.
        """
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

           calls QThreadPool.globalInstance().start(worker)
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
            with warnings.catch_warnings():
                warnings.filterwarnings("always")
                warnings.showwarning = lambda *w: self.warned.emit(w)
                result = self.work()
            if isinstance(result, Exception):
                if isinstance(result, RuntimeError):
                    # The Worker object has likely been deleted.
                    # A deleted wrapped C/C++ object may result in a runtime
                    # error that will cause segfault if we try to do much other
                    # than simply notify the user.
                    warnings.warn(
                        trans._(
                            "RuntimeError in aborted thread: {result}",
                            result=result,
                        ),
                        RuntimeWarning,
                    )
                    return
                else:
                    raise result
            self.returned.emit(result)
        except Exception as exc:
            self.errored.emit(exc)
        self._running = False
        self.finished.emit()

    def work(self) -> Union[Exception, Any]:
        """Main method to execute the worker.

        The end-user should never need to call this function.
        But subclasses must implement this method (See
        :meth:`GeneratorFunction.work` for an example implementation).
        Minimally, it should check ``self.abort_requested`` periodically and
        exit if True.

        Examples
        --------
        .. code-block:: python

            class MyWorker(WorkerBase):

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
            trans._(
                '"{name}" failed to define work() method',
                deferred=True,
                name=self.__class__.__name__,
            )
        )

    def start(self):
        """Start this worker in a thread and add it to the global threadpool.

        The order of method calls when starting a worker is:

        .. code-block:: none

           calls QThreadPool.globalInstance().start(worker)
           |               triggered by the QThreadPool.start() method
           |               |             called by worker.run
           |               |             |
           V               V             V
           worker.start -> worker.run -> worker.work
        """
        if self in WorkerBase._worker_set:
            raise RuntimeError(
                trans._(
                    'This worker is already started!',
                    deferred=True,
                )
            )

        # This will raise a RunTimeError if the worker is already deleted
        repr(self)

        WorkerBase._worker_set.add(self)
        self.finished.connect(lambda: WorkerBase._worker_set.discard(self))
        start_ = partial(QThreadPool.globalInstance().start, self)
        QTimer.singleShot(10, start_)


class FunctionWorker(WorkerBase):
    """QRunnable with signals that wraps a simple long-running function.

    .. note::

        ``FunctionWorker`` does not provide a way to stop a very long-running
        function (e.g. ``time.sleep(10000)``).  So whenever possible, it is
        better to implement your long running function as a generator that
        yields periodically, and use the :class:`GeneratorWorker` instead.

    Parameters
    ----------
    func : Callable
        A function to call in another thread
    *args
        will be passed to the function
    **kwargs
        will be passed to the function

    Raises
    ------
    TypeError
        If ``func`` is a generator function and not a regular function.
    """

    def __init__(self, func: Callable, *args, **kwargs):
        if inspect.isgeneratorfunction(func):
            raise TypeError(
                trans._(
                    "Generator function {func} cannot be used with FunctionWorker, use GeneratorWorker instead",
                    deferred=True,
                    func=func,
                )
            )
        super().__init__()

        self._func = func
        self._args = args
        self._kwargs = kwargs

    def work(self):
        return self._func(*self._args, **self._kwargs)


class GeneratorWorkerSignals(WorkerBaseSignals):

    yielded = Signal(object)  # emitted with yielded values (if generator used)
    paused = Signal()  # emitted when a running job has successfully paused
    resumed = Signal()  # emitted when a paused job has successfully resumed
    aborted = Signal()  # emitted when a running job is successfully aborted


class GeneratorWorker(WorkerBase):
    """QRunnable with signals that wraps a long-running generator.

    Provides a convenient way to run a generator function in another thread,
    while allowing 2-way communication between threads, using plain-python
    generator syntax in the original function.

    Parameters
    ----------
    func : callable
        The function being run in another thread.  May be a generator function.
    SignalsClass : type, optional
        A QObject subclass that contains signals, by default
        GeneratorWorkerSignals
    *args
        Will be passed to func on instantiation
    **kwargs
        Will be passed to func on instantiation
    """

    def __init__(
        self,
        func: Callable,
        *args,
        SignalsClass: Type[WorkerBaseSignals] = GeneratorWorkerSignals,
        **kwargs,
    ):
        if not inspect.isgeneratorfunction(func):
            raise TypeError(
                trans._(
                    "Regular function {func} cannot be used with GeneratorWorker, use FunctionWorker instead",
                    deferred=True,
                    func=func,
                )
            )
        super().__init__(SignalsClass=SignalsClass)

        self._gen = func(*args, **kwargs)
        self._incoming_value = None
        self._pause_requested = False
        self._resume_requested = False
        self._paused = False
        # polling interval: ONLY relevant if the user paused a running worker
        self._pause_interval = 0.01

    def work(self):
        """Core event loop that calls the original function.

        Enters a continual loop, yielding and returning from the original
        function.  Checks for various events (quit, pause, resume, etc...).
        (To clarify: we are creating a rudimentary event loop here because
        there IS NO Qt event loop running in the other thread to hook into)
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
                input = self._next_value()
                output = self._gen.send(input)
                self.yielded.emit(output)
            except StopIteration as exc:
                return exc.value
            except RuntimeError as exc:
                # The worker has probably been deleted.  warning will be
                # emitted in ``WorkerBase.run``
                return exc

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
        """Request to pause the worker if playing or resume if paused."""
        if self.is_paused:
            self._resume_requested = True
        else:
            self._pause_requested = True

    def pause(self) -> None:
        """Request to pause the worker."""
        if not self.is_paused:
            self._pause_requested = True

    def resume(self) -> None:
        """Send a request to resume the worker."""
        if self.is_paused:
            self._resume_requested = True


############################################################################

# public API

# For now, the next three functions simply wrap the QThreadPool API, and allow
# us to track and cleanup all workers that were started with ``start_worker``,
# provided that ``wait_for_workers_to_quit`` is called at shutdown.
# In the future, this could wrap any API, or a pure python threadpool.


def set_max_thread_count(num: int):
    """Set the maximum number of threads used by the thread pool.

    Note: The thread pool will always use at least 1 thread, even if
    maxThreadCount limit is zero or negative.
    """
    QThreadPool.globalInstance().setMaxThreadCount(num)


def wait_for_workers_to_quit(msecs: int = None):
    """Ask all workers to quit, and wait up to `msec` for quit.

    Attempts to clean up all running workers by calling ``worker.quit()``
    method.  Any workers in the ``WorkerBase._worker_set`` set will have this
    method.

    By default, this function will block indefinitely, until worker threads
    finish.  If a timeout is provided, a ``RuntimeError`` will be raised if
    the workers do not gracefully exit in the time requests, but the threads
    will NOT be killed.  It is (currently) left to the user to use their OS
    to force-quit rogue threads.

    .. important::

        If the user does not put any yields in their function, and the function
        is super long, it will just hang... For instance, there's no graceful
        way to kill this thread in python:

        .. code-block:: python

            @thread_worker
            def ZZZzzz():
                time.sleep(10000000)

        This is why it's always advisable to use a generator that periodically
        yields for long-running computations in another thread.

        See `this stack-overflow post
        <https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread>`_
        for a good discussion on the difficulty of killing a rogue python thread:

    Parameters
    ----------
    msecs : int, optional
        Waits up to msecs milliseconds for all threads to exit and removes all
        threads from the thread pool. If msecs is `None` (the default), the
        timeout is ignored (waits for the last thread to exit).

    Raises
    ------
    RuntimeError
        If a timeout is provided and workers do not quit successfully within
        the time allotted.
    """
    for worker in WorkerBase._worker_set:
        worker.quit()

    msecs = msecs if msecs is not None else -1
    if not QThreadPool.globalInstance().waitForDone(msecs):
        raise RuntimeError(
            trans._(
                "Workers did not quit gracefully in the time allotted ({msecs} ms)",
                deferred=True,
                msecs=msecs,
            )
        )


def active_thread_count() -> int:
    """Return the number of active threads in the global ThreadPool."""
    return QThreadPool.globalInstance().activeThreadCount()


#############################################################################

# convenience functions for creating Worker instances


def create_worker(
    func: Callable,
    *args,
    _start_thread: Optional[bool] = None,
    _connect: Optional[Dict[str, Union[Callable, Sequence[Callable]]]] = None,
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
    _connect : Dict[str, Union[Callable, Sequence]], optional
        A mapping of ``"signal_name"`` -> ``callable`` or list of ``callable``:
        callback functions to connect to the various signals offered by the
        worker class. by default None
    _worker_class : Type[WorkerBase], optional
        The :class`WorkerBase` to instantiate, by default
        :class:`FunctionWorker` will be used if ``func`` is a regular function,
        and :class:`GeneratorWorker` will be used if it is a generator.
    _ignore_errors : bool, optional
        If ``False`` (the default), errors raised in the other thread will be
        reraised in the main thread (makes debugging significantly easier).
    *args
        will be passed to ``func``
    **kwargs
        will be passed to ``func``

    Returns
    -------
    worker : WorkerBase
        An instantiated worker.  If ``_start_thread`` was ``False``, the worker
        will have a `.start()` method that can be used to start the thread.

    Raises
    ------
    TypeError
        If a worker_class is provided that is not a subclass of WorkerBase.
    TypeError
        If _connect is provided and is not a dict of ``{str: callable}``

    Examples
    --------
    .. code-block:: python

        def long_function(duration):
            import time
            time.sleep(duration)

        worker = create_worker(long_function, 10)

    """
    if not _worker_class:
        if inspect.isgeneratorfunction(func):
            _worker_class = GeneratorWorker
        else:
            _worker_class = FunctionWorker

    if not (
        inspect.isclass(_worker_class)
        and issubclass(_worker_class, WorkerBase)
    ):
        raise TypeError(
            trans._(
                'Worker {_worker_class} must be a subclass of WorkerBase',
                deferred=True,
                _worker_class=_worker_class,
            )
        )

    worker = _worker_class(func, *args, **kwargs)

    if _connect is not None:
        if not isinstance(_connect, dict):
            raise TypeError(
                trans._(
                    "The '_connect' argument must be a dict",
                    deferred=True,
                )
            )

        if _start_thread is None:
            _start_thread = True

        for key, val in _connect.items():
            _val = val if isinstance(val, (tuple, list)) else [val]
            for v in _val:
                if not callable(v):
                    raise TypeError(
                        trans._(
                            '"_connect[{key!r}]" must be a function or sequence of functions',
                            deferred=True,
                            key=key,
                        )
                    )
                getattr(worker, key).connect(v)

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
    connect: Optional[Dict[str, Union[Callable, Sequence[Callable]]]] = None,
    worker_class: Optional[Type[WorkerBase]] = None,
    ignore_errors: bool = False,
) -> Callable:
    """Decorator that runs a function in a separate thread when called.

    When called, the decorated function returns a :class:`WorkerBase`.  See
    :func:`create_worker` for additional keyword arguments that can be used
    when calling the function.

    The returned worker will have these signals:

        - *started*: emitted when the work is started
        - *finished*: emitted when the work is finished
        - *returned*: emitted with return value
        - *errored*: emitted with error object on Exception

    It will also have a ``worker.start()`` method that can be used to start
    execution of the function in another thread. (useful if you need to connect
    callbacks to signals prior to execution)

    If the decorated function is a generator, the returned worker will also
    provide these signals:

        - *yielded*: emitted with yielded values
        - *paused*: emitted when a running job has successfully paused
        - *resumed*: emitted when a paused job has successfully resumed
        - *aborted*: emitted when a running job is successfully aborted

    And these methods:

        - *quit*: ask the thread to quit
        - *toggle_paused*: toggle the running state of the thread.
        - *send*: send a value into the generator.  (This requires that your
          decorator function uses the ``value = yield`` syntax)

    Parameters
    ----------
    function : callable
        Function to call in another thread.  For communication between threads
        may be a generator function.
    start_thread : bool, optional
        Whether to immediaetly start the thread.  If False, the returned worker
        must be manually started with ``worker.start()``. by default it will be
        ``False`` if the ``_connect`` argument is ``None``, otherwise ``True``.
    connect : Dict[str, Union[Callable, Sequence]], optional
        A mapping of ``"signal_name"`` -> ``callable`` or list of ``callable``:
        callback functions to connect to the various signals offered by the
        worker class. by default None
    worker_class : Type[WorkerBase], optional
        The :class`WorkerBase` to instantiate, by default
        :class:`FunctionWorker` will be used if ``func`` is a regular function,
        and :class:`GeneratorWorker` will be used if it is a generator.
    ignore_errors : bool, optional
        If ``False`` (the default), errors raised in the other thread will be
        reraised in the main thread (makes debugging significantly easier).

    Returns
    -------
    callable
        function that creates a worker, puts it in a new thread and returns
        the worker instance.

    Examples
    --------
    .. code-block:: python

        @thread_worker
        def long_function(start, end):
            # do work, periodically yielding
            i = start
            while i <= end:
                time.sleep(0.1)
                yield i

            # do teardown
            return 'anything'

        # call the function to start running in another thread.
        worker = long_function()
        # connect signals here if desired... or they may be added using the
        # `connect` argument in the `@thread_worker` decorator... in which
        # case the worker will start immediately when long_function() is called
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
        return create_worker(
            function,
            *args,
            **kwargs,
        )

    return worker_function


############################################################################

# This is a variant on the above pattern, it uses QThread instead of Qrunnable
# see https://doc.qt.io/qt-5/threads-technologies.html#comparison-of-solutions
# (it appears from that table that QRunnable cannot emit or receive signals,
# but we circumvent that here with our WorkerBase class that also inherits from
# QObject... providing signals/slots).
#
# A benefit of the QRunnable pattern is that Qt manages the threads for you,
# in the QThreadPool.globalInstance() ... making it easier to reuse threads,
# and reduce overhead.
#
# However, a disadvantage is that you have no access to (and therefore less
# control over) the QThread itself.  See for example all of the methods
# provided on the QThread object: https://doc.qt.io/qt-5/qthread.html


# TODO: potentially remove this altogether, by refactoring the dims
# AnimationWorker to subclass WorkerBase


def _new_worker_qthread(
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
    _start_thread : bool
        If True, thread will be started immediately, otherwise, thread must
        be manually started with thread.start().
    _connect : dict, optional
        Optional dictionary of {signal: function} to connect to the new worker.
        for instance:  connections = {'incremented': myfunc} will result in:
        worker.incremented.connect(myfunc)
    *args
        will be passed to the Worker class on instantiation.
    **kwargs
        will be passed to the Worker class on instantiation.

    Returns
    -------
    worker : WorkerBase
        The created worker.
    thread : QThread
        The thread on which the worker is running.

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

        worker, thread = _new_worker_qthread(
            Worker,
            'argument',
            start_thread=True,
            connections={'increment': print},
        )

    """

    if _connect and not isinstance(_connect, dict):
        raise TypeError(
            trans._('_connect parameter must be a dict', deferred=True)
        )

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
