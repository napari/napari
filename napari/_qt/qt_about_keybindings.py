from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QComboBox,
)
import napari
from ..util.misc import get_keybindings_summary


class QtAboutKeybindings(QWidget):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # stacked keybindings widgets
        self.textEditBox = QTextEdit()
        self.textEditBox.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.keybindings_strs = {'Currently active': ''}
        layers = [
            napari.layers.Image,
            napari.layers.Labels,
            napari.layers.Points,
            napari.layers.Shapes,
            napari.layers.Surface,
            napari.layers.Vectors,
        ]
        for layer in layers:
            if len(layer.class_keymap) == 0:
                text = 'No keybindings'
            else:
                text = get_keybindings_summary(layer.class_keymap)
            self.keybindings_strs[str(layer.__name__)] = text

        # layer type selection
        self.layerTypeComboBox = QComboBox()
        for name in self.keybindings_strs.keys():
            self.layerTypeComboBox.addItem(name)
        self.layerTypeComboBox.activated[str].connect(
            lambda text=self.layerTypeComboBox: self.change_layer_type(text)
        )
        current_layer = 'Currently active'
        self.layerTypeComboBox.setCurrentText(current_layer)
        # self.change_layer_type(current_layer)
        layer_type_layout = QHBoxLayout()
        layer_type_layout.addWidget(QLabel('Layer type:'))
        layer_type_layout.addWidget(self.layerTypeComboBox)
        layer_type_layout.addStretch(1)
        layer_type_layout.setSpacing(0)
        self.layout.addLayout(layer_type_layout)
        self.layout.addWidget(self.textEditBox, 1)

        self.viewer.events.active_layer.connect(self.update_active_layer)
        self.update_active_layer(None)

    def change_layer_type(self, text):
        self.textEditBox.setText(self.keybindings_strs[text])

    def update_active_layer(self, event):
        text = ''
        # Add class and instance viewer keybindings
        text += get_keybindings_summary(self.viewer.class_keymap)
        text += get_keybindings_summary(self.viewer.keymap)

        layer = self.viewer.active_layer
        if layer is not None:
            # Add class and instance layer keybindings for the active layer
            text += get_keybindings_summary(layer.class_keymap)
            text += get_keybindings_summary(layer.keymap)

        # Do updates
        self.keybindings_strs['Currently active'] = text
        if self.layerTypeComboBox.currentText() == 'Currently active':
            self.textEditBox.setText(text)
