import datetime
import uuid
from enum import auto
from typing import Optional

from napari.utils.misc import Callable, StringEnum


class Status(StringEnum):
    PENDING = auto()
    BUSY = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    FAILED = auto()
    START_FAILED = auto()


class TaskStatusItem:
    def __init__(
        self,
        provider: str,
        status: Status,
        description: str,
        cancel_callback: Optional[Callable] = None,
    ) -> None:
        self.id: uuid.UUID = uuid.uuid4()
        self._provider = provider
        self._timestamp = [self._timestap()]
        self._status = [status]
        self._description = [description]
        self._cancel_callback = cancel_callback

    def _timestap(self) -> str:
        return datetime.datetime.now().isoformat()

    def __str__(self) -> str:
        return f'TaskStatusItem: ({self._provider}, {self.id}, {self._timestamp[-1]}, {self._status[-1]}, {self._description[-1]})'

    def update(self, status: Status, description: str) -> None:
        self._timestamp.append(self._timestap())
        self._status.append(status)
        self._description.append(description)

    def cancel(self) -> bool:
        self.update(Status.CANCELLED, '')
        if self._cancel_callback is not None:
            return self._cancel_callback()
        return False

    def state(self) -> tuple[str, str, Status, str]:
        return (
            self._provider,
            self._timestamp[-1],
            self._status[-1],
            self._description[-1],
        )


class TaskStatusManager:
    """
    A task status manager, to store status of long running processes/tasks.

    Only one instance is in general available through napari.

    napari methods and plugins can use it to register and update
    long running tasks.
    """

    _tasks: dict[uuid.UUID, TaskStatusItem]

    def __init__(self) -> None:
        # Note: we are using a dict here that may not be thread-safe; however
        # given the that the values from it are added/updated using an UUID
        # collision chances are low and it should be ok as long as operations
        # that require its iteration (`is_busy`, `get_status`, `cancel_all`)
        # are done when no task status additions are scheduled (i.e when
        # closing the application).
        self._tasks: dict[uuid.UUID, TaskStatusItem] = {}

    def register_task_status(
        self,
        provider: str,
        task_status: Status,
        description: str,
        cancel_callback: Optional[Callable] = None,
    ) -> uuid.UUID:
        item = TaskStatusItem(
            provider, task_status, description, cancel_callback
        )
        self._tasks[item.id] = item
        return item.id

    def update_task_status(
        self,
        status_id: uuid.UUID,
        task_status: Status,
        description: str = '',
    ) -> bool:
        if status_id in self._tasks:
            item = self._tasks[status_id]
            item.update(task_status, description)
            return True

        return False

    def is_busy(self) -> bool:
        for _, item in self._tasks.items():
            if item.state()[2] in {Status.PENDING, Status.BUSY}:
                return True
        return False

    def get_status(self) -> list[str]:
        messages = []
        for _, item in self._tasks.items():
            provider, _ts, status, description = item.state()
            if status in {Status.PENDING, Status.BUSY}:
                messages.append(f'{provider}: {description}')

        return messages

    def cancel_all(self) -> None:
        for _, item in self._tasks.items():
            item.cancel()
