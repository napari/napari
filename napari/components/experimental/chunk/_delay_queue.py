"""DelayQueue class.
"""
import logging
import threading
import time
from collections import namedtuple
from typing import List, Optional

from ....utils.perf import add_counter_event

LOGGER = logging.getLogger("napari.async")

# Each queue entry contains the request we are going to submit, and the
# time in seconds when it should be submitted.
QueueEntry = namedtuple('QueueEntry', ["request", "submit_time"])


class DelayQueue(threading.Thread):
    """A threaded queue that delays request submission.

    The DelayQueue exists because it's hard to cancel requests which
    have been given to the thread or process pool. But it's trivial
    to cancel requests from the DelayQueue.

    We could just throw away the stale results, but that could generate
    a lot of unnecessary network traffic and a lot of unnecessary
    background computation in some cases.

    So the GUI thread calls DelayQueue.clear() every time the user switches
    to a new slice. This trivially clears the requests still in this queue.

    The net result with much less wasted background effort. And because of
    this when the user does pause, there are generally resources available
    to immediately start on that load.

    Parameters
    ----------
    delay_queue_ms : float
        Delay the request for this many milliseconds before submission.
    submit_func
        Call this function to submit the request.

    Attributes
    ----------
    delay_seconds : float
        Delay each request by this many seconds.
    submit_func
        We call this function to submit the request.
    entries : List[QueueEntry]
        The entries in the queue.
    lock : threading.Lock
        Lock access to the self.entires queue.
    event : threading.Event
        Event we signal to wake up the worker.
    """

    def __init__(self, delay_queue_ms: float, submit_func):
        super().__init__(daemon=True)
        self.delay_seconds: float = (delay_queue_ms / 1000)
        self.submit_func = submit_func

        self.entries: List[QueueEntry] = []
        self.lock = threading.Lock()
        self.event = threading.Event()

        self.start()

    def add(self, request) -> None:
        """Insert the request into the queue.

        Parameters
        ----------
        request : ChunkRequest
            Insert this request into the queue.
        """
        if self.delay_seconds == 0:
            self.submit_func(request)  # Submit with no delay.
            return

        LOGGER.info("DelayQueue.add: data_id=%d", request.data_id)

        # Create entry with the time to submit it.
        submit_time = time.time() + self.delay_seconds
        entry = QueueEntry(request, submit_time)

        with self.lock:
            self.entries.append(entry)
            num_entries = len(self.entries)

        add_counter_event("delay_queue", entries=num_entries)

        if num_entries == 1:
            self.event.set()  # The list was empty so wake up the worker.

    def clear(self, data_id: int) -> None:
        """Remove any entires for this data_id.

        Parameters
        ----------
        data_id : int
            Remove entries for this data_id.
        """
        LOGGER.info("DelayQueue.clear: data_id=%d", data_id)

        with self.lock:
            self.entries[:] = [
                x for x in self.entries if x.request.key.data_id != data_id
            ]

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
        # If entry is due to be submitted.
        if entry.submit_time < now:
            LOGGER.info(
                "DelayQueue.submit: data_id=%d", entry.request.data_id,
            )
            self.submit_func(entry.request)
            return True  # We submitted this request.
        return False

    def run(self):
        """The DelayQueue thread's main method.

        Submit all due entires, then sleep or wait on self.event
        for new entries.
        """
        while True:
            now = time.time()

            with self.lock:
                seconds = self._submit_due_entries(now)
                num_entries = len(self.entries)

            add_counter_event("delay_queue", entries=num_entries)

            if seconds is None:
                # There were no entries left, so wait until there is one.
                self.event.wait()
                self.event.clear()
            else:
                # Sleep until the next entry is due. This will tend to
                # oversleep by a few milliseconds, but close enough for our
                # purposes. Once we wake up we'll submit all due entries.
                # So we won't miss any.
                time.sleep(seconds)

    def _submit_due_entries(self, now: float) -> Optional[float]:
        """Submit all due entries, oldest to newest.

        Parameters
        ----------
        now : float
            Current time in seconds.

        Returns
        -------
        Optional[float]
            Seconds until next entry is due, or None if no next entry.
        """
        while self.entries:
            # Submit the oldest entry if it's due.
            if self.submit(self.entries[0], now):
                self.entries.pop(0)  # Remove the one we just submitted.
            else:
                # Oldest entry is not due, return time until it is.
                return self.entries[0].submit_time - now

        return None  # There are no more entries.

    def flush(self):
        """Submit all entries right now."""
        with self.lock:
            for entry in self.entries:
                self.submit_func(entry.request)
            self.entries = []
