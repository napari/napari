"""LoaderPool class.

ChunkLoader has one or more of these. They load data in worker pools.
"""
from __future__ import annotations

import logging
from concurrent.futures import (
    CancelledError,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

# Executor for either a thread pool or a process pool.
PoolExecutor = Union[ThreadPoolExecutor, ProcessPoolExecutor]

LOGGER = logging.getLogger("napari.loader")

DoneCallback = Optional[Callable[[Future], None]]

if TYPE_CHECKING:
    from napari.components.experimental.chunk._request import ChunkRequest


class LoaderPool:
    """Loads chunks asynchronously in worker threads or processes.

    We cannot call np.asarray() in the GUI thread because it might block on
    IO or a computation. So we call np.asarray() in _chunk_loader_worker()
    instead.

    Parameters
    ----------
    config : dict
        Our configuration, see napari.utils._octree.py for format.
    on_done_loader : Callable[[Future], None]
        Called when a future finishes.

    Attributes
    ----------
    force_synchronous : bool
        If True all requests are loaded synchronously.
    num_workers : int
        The number of workers.
    use_processes | bool
        Use processess as workers, otherwise use threads.
    _executor : PoolExecutor
        The thread or process pool executor.
    _futures : Dict[ChunkRequest, Future]
        In progress futures for each layer (data_id).
    _delay_queue : DelayQueue
        Requests sit in here for a bit before submission.
    """

    def __init__(self, config: dict, on_done_loader: DoneCallback = None):
        from napari.components.experimental.chunk._delay_queue import (
            DelayQueue,
        )

        self.config = config
        self._on_done_loader = on_done_loader

        self.num_workers: int = int(config['num_workers'])
        self.use_processes: bool = bool(config.get('use_processes', False))

        self._executor: PoolExecutor = _create_executor(
            self.use_processes, self.num_workers
        )
        self._futures: Dict[ChunkRequest, Future] = {}
        self._delay_queue = DelayQueue(config['delay_queue_ms'], self._submit)

    def load_async(self, request: ChunkRequest) -> None:
        """Load this request asynchronously.

        Parameters
        ----------
        request : ChunkRequest
            The request to load.
        """
        # Add to the DelayQueue which will call our self._submit() method
        # right away, if zero delay, or after the configured delay.
        self._delay_queue.add(request)

    def cancel_requests(
        self, should_cancel: Callable[[ChunkRequest], bool]
    ) -> List[ChunkRequest]:
        """Cancel pending requests based on the given filter.

        Parameters
        ----------
        should_cancel : Callable[[ChunkRequest], bool]
            Cancel the request if this returns True.

        Returns
        -------
        List[ChunkRequests]
            The requests that were cancelled, if any.
        """
        # Cancelling requests in the delay queue is fast and easy.
        cancelled = self._delay_queue.cancel_requests(should_cancel)

        num_before = len(self._futures)

        # Cancelling futures may or may not work. Future.cancel() will
        # return False if the worker is already loading the request and it
        # cannot be cancelled.
        for request in list(self._futures.keys()):
            if self._futures[request].cancel():
                del self._futures[request]
                cancelled.append(request)

        num_after = len(self._futures)
        num_cancelled = num_before - num_after

        LOGGER.debug(
            "cancel_requests: %d -> %d futures (cancelled %d)",
            num_before,
            num_after,
            num_cancelled,
        )

        return cancelled

    def _submit(self, request: ChunkRequest) -> Optional[Future]:
        """Initiate an asynchronous load of the given request.

        Parameters
        ----------
        request : ChunkRequest
            Contains the arrays to load.
        """
        # Submit the future. Have it call self._done when finished.
        future = self._executor.submit(_chunk_loader_worker, request)
        future.add_done_callback(self._on_done)
        self._futures[request] = future

        LOGGER.debug(
            "_submit_async: %s elapsed=%.3fms num_futures=%d",
            request.location,
            request.elapsed_ms,
            len(self._futures),
        )

        return future

    def _on_done(self, future: Future) -> None:
        """Called when a future finishes.

        future : Future
            This is the future that finished.
        """
        try:
            request = self._get_request(future)
        except ValueError:
            return  # Pool not running? App exit in progress?

        if request is None:
            return  # Future was cancelled, nothing to do.

        # Tell the loader this request finished.
        if self._on_done_loader is not None:
            self._on_done_loader(request)

    def shutdown(self) -> None:
        """Shutdown the pool."""
        # Avoid crashes or hangs on exit.
        self._delay_queue.shutdown()
        self._executor.shutdown(wait=True)

    @staticmethod
    def _get_request(future: Future) -> Optional[ChunkRequest]:
        """Return the ChunkRequest for this future.

        Parameters
        ----------
        future : Future
            Get the request from this future.

        Returns
        -------
        Optional[ChunkRequest]
            The ChunkRequest or None if the future was cancelled.
        """
        try:
            # Our future has already finished since this is being
            # called from Chunk_Request._done(), so result() will
            # never block. But we can see if it finished or was
            # cancelled. Although we don't care right now.
            return future.result()
        except CancelledError:
            return None


def _create_executor(use_processes: bool, num_workers: int) -> PoolExecutor:
    """Return the thread or process pool executor.

    Parameters
    ----------
    use_processes : bool
        If True use processes, otherwise threads.
    num_workers : int
        The number of worker threads or processes.
    """
    if use_processes:
        LOGGER.debug("Process pool num_workers=%d", num_workers)
        return ProcessPoolExecutor(max_workers=num_workers)

    LOGGER.debug("Thread pool num_workers=%d", num_workers)
    return ThreadPoolExecutor(max_workers=num_workers)


def _chunk_loader_worker(request: ChunkRequest) -> ChunkRequest:
    """This is the worker thread or process that loads the array.

    We call np.asarray() in a worker because it might lead to IO or
    computation which would block the GUI thread.

    Parameters
    ----------
    request : ChunkRequest
        The request to load.
    """
    request.load_chunks()  # loads all chunks in the request
    return request
