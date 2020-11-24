"""Monitor API.
"""
import os
from multiprocessing.managers import SharedMemoryManager
from queue import Empty, Queue

from ...layerlist import LayerList

queue = Queue()


def get_queue():
    return queue


class MonitorApi:
    """The Monitor API that client's can access.

    The "API" right now is just shared dict. In the dict the key
    'commands' is a list. Clients can append to that list and we
    can consume those commands.
    """

    def __init__(self, layers: LayerList):
        self._log("starting")
        self.layers = None

        SharedMemoryManager.register('command_queue', callable=get_queue)

        # Manager is set later once it is created.
        self.manager = None

    def set_manager(self, manager: SharedMemoryManager) -> None:
        """Set the shared manager we will get values from.

        Parameters
        ----------
        manager : SharedMemoryManager
            Get values from this manager.
        """
        self.manager = manager

    def poll(self):
        """Poll the shared dict for new commands."""
        if self.manager is None:
            return  # Nothing to poll yet.

        self._log("Checking command queue")
        command_queue = self.manager.command_queue()
        while True:
            try:
                command = command_queue.get_nowait()
                self._process_command(command)
            except Empty:
                self._log("EMPTY")
                return

    def _process_command(self, command: dict):
        self._log(f"command={command}")

    def _log(self, msg: str) -> None:
        # print for now but change to logging
        print(f"MonitorApi: process={os.getpid()} {msg}")
