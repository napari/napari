from concurrent.futures import Future
from typing import Generic, Optional, TypeVar

R = TypeVar("R")


class MockFuture(Future, Generic[R]):
    """Synchronous Future that mocks the API, but requires result upfront.

    This object is useful when we want to eventually default to asynchronous
    behavior, but need to start synchronously, as it will allow us to change
    later without breaking any API.
    """

    def __init__(
        self, result: R, exception: Optional[BaseException] = None
    ) -> None:
        self._result = result
        self._exception = None

    def __repr__(self) -> str:
        name = self.__class__.__name__
        resultT = self._result.__class__.__name__
        return f'<{name} at {id(self):#x} state=finished returned {resultT}>'

    def result(self, timeout: Optional[float] = None) -> R:
        if self._exception:
            raise self._exception
        return self._result

    def exception(
        self, timeout: Optional[float] = None
    ) -> Optional[BaseException]:
        return self._exception

    def cancel(self) -> bool:
        return False

    def cancelled(self) -> bool:
        return False

    def running(self) -> bool:
        return False

    def done(self) -> bool:
        return True

    def set_exception(self, exception: Optional[BaseException]) -> None:
        return

    def set_result(self, result) -> None:
        return

    def set_running_or_notify_cancel(self) -> bool:
        return False
