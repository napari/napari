"""Monitor API.
"""
from multiprocessing.managers import SharedMemoryManager

from ...layerlist import LayerList


class MonitorApi:
    """The Monitor API that client's can access.

    Monitor clients can access shared memory resources using an API. The
    API mechanism is implemented by BaseManager, where SharedMemoryManager
    is derived from BaseManager.

    While shared memory resources are implemented by shared memory, the
    BaseManager API uses sockets and pickle serialization. Some comments
    in SharedMemoryManager seem to imply share memory is use, but that
    doesn't seem true since it requires a socket connection.

    At any rate, best practice seems to be use the API for small-ish data,
    but direct the client to a shared memory resources for bulk data.

    The two should work nicely together. For instance an API for "get
    images" might return a dict with indexes into a shared memory resource.
    The client makes the API call, then it can access the shared memory.
    """

    def __init__(self, layers: LayerList):
        self.layers = None
        self._register()

    def _register(self) -> None:
        print("Monitor: registering API")
        SharedMemoryManager.register(
            'test_callable', callable=self._test_callable
        )

    def _test_callable(self, input_value) -> int:
        print(f"test_callable: {input_value}")
        return 1234
