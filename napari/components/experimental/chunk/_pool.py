"""LoaderPool class.

ChunkLoader has one or more of these. They load data in worker pools.
"""
import logging
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Dict, List, Union

from ._delay_queue import DelayQueue
from ._request import ChunkRequest

# Executor for either a thread pool or a process pool.
PoolExecutor = Union[ThreadPoolExecutor, ProcessPoolExecutor]

LOGGER = logging.getLogger("napari.loader")


class LoaderPool:
    """Loads chunks asynchronously in worker threads or processes.

    We cannot call np.asarray() in the GUI thread because it might block on
    IO or a computation. So we call np.asarray() in _chunk_loader_worker()
    instead.

    Parameters
    ----------
    config : dict
        Our configuration, see napari.utils._octree.py for format.
    done_callback : Callable[[Future], None]
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
    _futures : Dict[int, List[Future]]
        In progress futures for each layer (data_id).
    _delay_queue : DelayQueue
        Requests sit in here for a bit before submission.
    """

    def __init__(self, config: dict, done_callback: Callable[[Future], None]):
        self.config = config
        self._done = done_callback

        self.num_workers: int = int(config['num_workers'])
        self.use_processes: bool = bool(config['use_processes'])

        self._executor: PoolExecutor = _create_executor(
            self.use_processes, self.num_workers
        )
        self._futures: Dict[int, List[Future]] = {}
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

    def _submit(self, request: ChunkRequest) -> None:
        """Initiate an asynchronous load of the given request.

        Parameters
        ----------
        request : ChunkRequest
            Contains the arrays to load.
        """
        # Submit the future. Have it call self._done when finished.
        future = self._executor.submit(_chunk_loader_worker, request)
        future.add_done_callback(self._done)

        # Store the future in case we need to cancel it.
        future_list = self._futures.setdefault(request.data_id, [])
        future_list.append(future)

        LOGGER.debug(
            "_submit_async: %s elapsed=%.3fms num_futures=%d",
            request.key.location,
            request.elapsed_ms,
            len(future_list),
        )

        return future

    def clear_pending(self, data_id: int) -> None:
        """Clear any pending requests for this data_id.

        Parameters
        ----------
        data_id : int
            Clear all requests associated with this data_id.
        """
        LOGGER.debug("_clear_pending %d", data_id)

        # Clear delay queue first. These requests are trivial to clear
        # because they have not even been submitted to the worker pool.
        self._delay_queue.clear(data_id)

        # Get list of futures we submitted to the pool.
        future_list = self._futures.setdefault(data_id, [])

        # Try to cancel all futures in the list, but cancel() will return
        # False if the task already started running.
        num_before = len(future_list)
        future_list[:] = [x for x in future_list if x.cancel()]
        num_after = len(future_list)
        num_cleared = num_before - num_after

        # Delete the list entirely if empty
        if num_after == 0:
            del self._futures[data_id]

        # Log what we did.
        if num_before == 0:
            LOGGER.debug("_clear_pending: empty")
        else:
            LOGGER.debug(
                "_clear_pending: %d of %d cleared -> %d remain",
                num_cleared,
                num_before,
                num_after,
            )


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
