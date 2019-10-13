from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QLabel,
    QDialog,
    QFrame,
)
import napari
from ..util.misc import get_keybindings_summary


class QtAboutKeybindings(QTabWidget):
    def __init__(self, viewer, parent):
        super(QtAboutKeybindings, self).__init__(parent)

        self.viewer = viewer

        self.addTab(QtActiveKeybindings(self.viewer), 'Currently active')
        self.addTab(QtLayerKeybindings(napari.layers.Image), 'Image')
        self.addTab(QtLayerKeybindings(napari.layers.Labels), 'Labels')
        self.addTab(QtLayerKeybindings(napari.layers.Points), 'Labels')
        self.addTab(QtLayerKeybindings(napari.layers.Shapes), 'Shapes')
        self.addTab(QtLayerKeybindings(napari.layers.Surface), 'Surface')
        self.addTab(QtLayerKeybindings(napari.layers.Vectors), 'Vectors')

    @staticmethod
    def showAbout(qt_viewer):
        d = QDialog()
        d.setObjectName('QtAboutKeybindings')
        d.setStyleSheet(qt_viewer.styleSheet())
        d.setGeometry(150, 150, 350, 400)
        d.setFixedSize(600, 700)
        QtAboutKeybindings(qt_viewer.viewer, d)
        d.setWindowTitle('Keybindings')
        d.setWindowModality(Qt.ApplicationModal)
        d.exec_()


class QtActiveKeybindings(QWidget):
    def __init__(self, viewer):
        super().__init__()

        self.layout = QVBoxLayout()

        keybindings_str = ''
        # Add class and instance viewer keybindings
        keybindings_str += get_keybindings_summary(viewer.class_keymap)
        keybindings_str += get_keybindings_summary(viewer.keymap)

        layer = viewer.active_layer
        if layer is not None:
            # Add class and instance layer keybindings for the active layer
            keybindings_str += get_keybindings_summary(layer.class_keymap)
            keybindings_str += get_keybindings_summary(layer.keymap)

        active_label = QLabel(keybindings_str)
        active_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        active_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(active_label)
        self.setLayout(self.layout)


class QtLayerKeybindings(QWidget):
    def __init__(self, layer):
        super().__init__()

        self.layout = QVBoxLayout()

        # Add class keybindings for the layer
        if len(layer.class_keymap) == 0:
            keybindings_str = 'No keybindings'
        else:
            keybindings_str = get_keybindings_summary(layer.class_keymap)

        layer_label = QLabel(keybindings_str)
        layer_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layer_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(layer_label)
        self.setLayout(self.layout)
