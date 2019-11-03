from dask.distributed import Pub, Sub, Client
from contextlib import contextmanager
import asyncio
from ..util.event import EmitterGroup, Event


class RemoteTopic:
    def __init__(self, topic, address):
        super().__init__()

        self.address = address
        self.client = Client(self.address)
        self.topic = topic
        self.pub = Pub(self.topic)
        self._block = False
        self.events = EmitterGroup(source=self, auto_connect=True, value=Event)

    @contextmanager
    def block_put(self):
        self._block = True
        yield
        self._block = False

    def put(self, value):
        if self._block is False:
            print('putting', value)
            self.pub.put(value)

    async def sub(self):
        async with Client(self.address, asynchronous=True) as client:
            sub = Sub(self.topic)
            while True:
                value = await sub.get()
                print('gotten', value)
                with self.block_put():
                    self.events.value(value=value)

    def connect(self):
        # Connect to event loop
        asyncio.get_event_loop().create_task(self.sub())
