"""ChunkDelayQueue used by the ChunkLoader
"""
from collections import namedtuple
import threading
import time
from typing import List

from ._config import async_config

# Each queue entry contains 1 request and the time we should submit it.
QueueEntry = namedtuple('QueueEntry', ["request", "submit_time"])


class ChunkDelayQueue(threading.Thread):
    """A threaded queue that delays request submission.

    We delay submitting requests so that if you are zipping through slices
    we don't end up starting a bunch of workers to load slices we will not
    even need. We can't cancel workers once they've started, but we can
    trivially cancel requests still in this delay queue.

    For example if the user is continuously move the slider requests will
    got into this queue and then get immediately canceled as the user goes
    to the next slice. Only when the users pauses for more than
    `delay_seconds` will the request actually get submitted the worker.

    TODO_ASYNC: when user click previous/next buttons we could have it
    skip the delay. The delay is really only for the slider. This would
    speed up previous/next by 100ms.

    Attributes
    ----------
    delay_seconds : float
        Delay each request by this many seconds.
    submit_func
        Call this function to submit the request.
    use_processes : bool
        If True use a process pool, otherwise use a thread pool.
    num_workers : int
        Create this many worker threads or processes.
    entries : List[QueueEntry]
        The entries in the queue
    """

    def __init__(self, delay_seconds, submit_func):
        super().__init__(daemon=True)
        self.delay_seconds = delay_seconds
        self.submit_func = submit_func
        self.use_processes: bool = async_config.use_processes
        self.num_workers: int = async_config.num_workers
        self.entries: List[QueueEntry] = []

        # If we sleep exactly delay_seconds then sometimes by chance we'll
        # delay for twice that long. Instead we only sleep for 1/5th of the
        # delay time, then our max over-shoot is 20% of delay_seconds. Note
        # that we never submit an entry early, all requests are delayed by
        # at least delay_seconds.
        self.sleep_time_seconds = delay_seconds / 4

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
            self.entries.append(QueueEntry(request, submit_time))

    def submit(self, entry: QueueEntry, now: float) -> bool:
        """Submit and return True if entry is ready to be submitted.

        Parameters
        ----------
        entry : QueueEntry
            The entry to potentially submit.
        now : float
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

    def clear(self, data_id: int) -> None:
        """Remove any entires for this data_id.

        Parameters
        ----------
        data_id : int
            Remove entries for this data_id.
        """
        self.entries = [
            x for x in self.entries if x.request.key.data_id != data_id
        ]

    def run(self):
        """The thread's main method.

        Submit requests after their delay is up.
        """
        while True:
            # Submit all entires which are due, then wait.
            now = time.time()
            self.entries = [x for x in self.entries if not self.submit(x, now)]
            time.sleep(self.sleep_time_seconds)
