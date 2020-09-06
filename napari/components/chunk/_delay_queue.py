"""ChunkDelayQueue used by the ChunkLoader
"""
import logging
import threading
import time
from collections import namedtuple
from typing import List, Optional

from ...utils.perf import add_counter_event

LOGGER = logging.getLogger("ChunkLoader")

# Each queue entry contains 1 request and the time we should submit it.
QueueEntry = namedtuple('QueueEntry', ["request", "submit_time"])


class DelayQueue(threading.Thread):
    """A threaded queue that delays request submission.

    We have a DelayQueue because once a worker is handling a request we
    cannot cancel or abort it. But if the user is rapidly changing slices,
    requests need to be cancelled often, because the requests for previous
    slices will quickly become stale.

    We could load those stale requests and just throw away the results, but
    that will hammer the remote server with bogus requests, and it would
    mean there might not be an available worker when the user finally does
    settle on a slice they want to load.

    So the GUI thread calls DelayQueue.clear() everytime the user switches
    to a new slice and we trivially clear quests still in this queue. This
    greatly reduces requests the remote server, and keeps the worker pool
    from being overly busy.

    Parameters
    ----------
    delay_queue_ms : float
        Delay the request for this many milliseconds before submission.
    sumbit_func
        We call this function to submit the request.

    Attributes
    ----------
    delay_seconds : float
        Delay each request by this many seconds.
    submit_func
        We call this function to submit the request.
    entries : List[QueueEntry]
        The entries in the queue
    lock : threading.Lock
        Guard access to the list of entries.
    event : threading.Event()
        Signal the worker there are entires in the queue.
    """

    def __init__(self, delay_queue_ms: float, submit_func):
        super().__init__(daemon=True)
        self.delay_seconds: float = (delay_queue_ms / 1000)
        self.submit_func = submit_func

        # The entries waiting to be submitted.
        self.entries: List[QueueEntry] = []

        # Lock access to self.entries. If we are careful we can probably rely
        # on the just the GIL, but it was getting a little complicated because
        # of the delay, this should be iron clad and not much slower.
        self.lock = threading.Lock()

        # Worker blocks on this if the queue is empty, so it can politely
        # wait for entries. It seems like maybe threading.Queue could
        # replace the lock and the event, but again with the delay that was
        # complicated.
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

        LOGGER.info("DelayQueue.add: data_id=%d", request.key.data_id)

        # Create entry with the time to submit it.
        submit_time = time.time() + self.delay_seconds
        entry = QueueEntry(request, submit_time)

        with self.lock:
            self.entries.append(entry)
            num_entries = len(self.entries)

        add_counter_event("delay_queue", entries=num_entries)

        # If the list was previously empty.
        if num_entries == 1:
            self.event.set()  # Wake up the worker

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
                "DelayQueue.submit: data_id=%d", entry.request.key.data_id
            )
            self.submit_func(entry.request)
            return True  # We submitted this request.
        return False

    def run(self):
        """The DelayQueue thread's main method.

        Submit all due entires, then sleep or wait on self.event
        for new entries.

        We have a lock since DelayQueue.add() and DelayQueue.clear()
        can be called out from under us in the GUI thread. We could
        probably do it without a lock if we were very careful, but
        this makes it bullet-proof and is probably not much slower.

        It seems like threading.Queue might be suitable here, but it
        was not obvious how to get the delay working with it.
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
        else:
            return None  # There are no more entries.
