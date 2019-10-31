from qtpy.QtCore import QThread


class QtUpdateUI(QThread):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        self.fn()
