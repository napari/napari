from .util import RemoteTopic


class DaskRemoteViewer:
    def __init__(self, viewer, address):
        super().__init__()

        self.viewer = viewer
        self.address = address

        self.title = RemoteTopic('viewer_title', self.address)
        self.viewer.events.title.connect(
            lambda e: self.title.put(self.viewer.title)
        )
        self.title.events.value.connect(self._on_title_update)

    def _on_title_update(self, event):
        print('on_update', event.value)
        self.viewer.title = event.value

    def connect(self):
        self.title.connect()
