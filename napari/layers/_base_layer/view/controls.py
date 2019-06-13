from qtpy.QtWidgets import QFrame


class QtLayerControls(QFrame):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
