"""ChunkDelayQueue used by the ChunkLoader
"""
import threading
import time

from ._config import async_config


class ChunkQueueEntry:
    def __init__(self, request, submit_time):
        self.request = request
        self.submit_time = submit_time


class ChunkDelayQueue(threading.Thread):
    """A threaded queue that delays request submission a bit.

    We delay submitting requests so that if you are zipping through slices
    we don't end up starting a bunch of workers for stale slices where no
    one will use the results. We can't cancel workers once they've started,
    but we can trivially cancel requests still in this queue.

    For example if the user is continuously move the slider requests will
    got into this queue and then get immediately canceled as the user
    goes to the next slice. Only when they pause by self.delay_seconds
    will the request actually get submitting the worker start loading.

    Attributes
    ----------
    executor : Union[ThreadPoolExecutor, ProcessPoolExecutor]
        Our thread or process pool executor.
    use_processes : bool
        If True use a process pool for the workers, otherwise use a thread pool.
    num_workers : int
        Create this many worker threads or processes.
    """

    def __init__(self, delay_seconds, submit_func):
        super().__init__(daemon=True)
        self.delay_seconds = delay_seconds
        self.submit_func = submit_func
        self.use_processes: bool = async_config.use_processes
        self.num_workers: int = async_config.num_workers
        self.entries = []
        self.start()

    def add(self, request) -> None:
        """Insert the request into the queue.

        Parameters
        ----------
        request : ChunkRequest
            Insert this request into the queue.
        """
        if self.delay_seconds == 0:
            # Submit with no delay.
            self.submit_func(request)
        else:
            # Add to the queue for a short period of time.
            submit_time = time.time() + self.delay_seconds
            self.entries.append(ChunkQueueEntry(request, submit_time))

    def submit(self, entry, now):
        """Submit the return if its time.

        Parameters
        ----------
        entry : ChunkQueueEntry
            The entry to potentially submit.
        now : int
            Current time in seconds.

        Return
        ------
        bool
            True if the entry was submitted.
        """
        if entry.submit_time < now:
            self.submit_func(entry.request)
            return True
        return False

    def clear(self, data_id):
        """Remove any entires for this data_id.

        Parameters
        ----------
        data_id : int
            Remove entries for this data_id.
        """
        self.entries = [
            x for x in self.entries if x.request.data_id != data_id
        ]

    def run(self):
        """The thread's main method.

        Submit requests after their delay is up.
        """
        while True:
            # Submit all entires which are due
            now = time.time()
            self.entries = [x for x in self.entries if not self.submit(x, now)]
            time.sleep(self.delay_seconds)
