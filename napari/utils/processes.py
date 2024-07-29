import datetime
import uuid
from enum import auto
from typing import Optional

from napari.utils.misc import StringEnum


class ProcessStatus(StringEnum):
    BUSY = auto()
    IDLE = auto()


class ProcessStatusItem:
    def __init__(
        self,
        status: ProcessStatus,
        description: str,
        id_: Optional[uuid.UUID] = None,
        timestamp: Optional[str] = None,
    ):
        self.id = id_ or uuid.uuid4()
        self.timestamp = timestamp or datetime.datetime.now().isoformat()
        self.status = status
        self.description = description

    def __str__(self) -> str:
        return f'ProcessStatusItem({self.status}, {self.description}, {self.id}, {self.timestamp})'


class ProcessStatusManager:
    """
    A process manager, to store status of long running processes.

    Only one instance is in general available through napari.

    napari methods and plugins can use it to register and unregister
    long running processes
    """

    _processes: list[ProcessStatusItem]
    _process_map: dict[uuid.UUID, ProcessStatusItem]

    def __init__(self) -> None:
        self._processes: list = []
        self._process_map: dict = {}

    def register_process_status(
        self, process_status: ProcessStatus, description: str
    ) -> uuid.UUID:
        item = ProcessStatusItem(process_status, description)
        self._processes.append(item)
        self._process_map[item.id] = item
        return item.id

    def unregister_process_status(self, process_status_id: uuid.UUID) -> bool:
        if process_status_id in self._process_map:
            self._process_map.pop(process_status_id)
            return True

        return False

    def is_busy(self) -> bool:
        return len(self._process_map) > 0

    def get_status(self) -> list[str]:
        messages = []
        for _, item in self._process_map.items():
            if item.status == ProcessStatus.BUSY:
                messages.append(item.description)

        return messages


process_status_manager = ProcessStatusManager()


def register_process_status(
    process_status: ProcessStatus, description: str
) -> uuid.UUID:
    """
    Register a long running process.
    """
    return process_status_manager.register_process_status(
        process_status, description
    )


def unregister_process_status(process_status_id: uuid.UUID) -> bool:
    """
    Unregister a long running process.
    """
    return process_status_manager.unregister_process_status(process_status_id)
