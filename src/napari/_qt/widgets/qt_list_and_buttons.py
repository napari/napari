from qtpy.QtCore import QSize
from qtpy.QtWidgets import QWIDGETSIZE_MAX, QVBoxLayout, QWidget


class QtLayerListAndButtons(QWidget):
    def __init__(self, layer_buttons, layer_list, viewer_buttons):
        super().__init__()
        self.setObjectName('layerList')
        layerListLayout = QVBoxLayout()
        layerListLayout.addWidget(layer_buttons)
        layerListLayout.addWidget(layer_list)
        layerListLayout.addWidget(viewer_buttons)
        layerListLayout.setContentsMargins(8, 4, 8, 6)
        self.setLayout(layerListLayout)

    def sizeHint(self):
        # because we use Maximum as dock widget size policy;
        # without this, this widget would take just enough space
        # instead of all the available space
        return QSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
