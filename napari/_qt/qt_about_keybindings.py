from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QLabel,
    QDialog,
    QFrame,
    QScrollArea,
    QSizePolicy,
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
        d.setWindowTitle('Keybindings')
        qt_viewer._about_keybindings = QtAboutKeybindings(qt_viewer.viewer, d)
        d.show()
        d.setWindowModality(Qt.NonModal)
        d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        qt_viewer._about_keybindings_dialog = d


class QtActiveKeybindings(QScrollArea):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.active_label = QLabel()
        self.active_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.active_label.setAlignment(Qt.AlignLeft)
        self.active_label.setContentsMargins(10, 10, 10, 10)
        self.active_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        self.update_text(None)

        scroll_widget = QWidget()
        scroll_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        scroll_widget.setContentsMargins(10, 10, 10, 10)
        scroll_layout = QVBoxLayout()
        scroll_layout.addWidget(self.active_label)
        scroll_layout.addStretch(1)
        scroll_widget.setLayout(scroll_layout)
        self.setWidget(scroll_widget)

        self.viewer.events.active_layer.connect(self.update_text)

    def update_text(self, event):
        keybindings_str = ''
        # Add class and instance viewer keybindings
        keybindings_str += get_keybindings_summary(self.viewer.class_keymap)
        keybindings_str += get_keybindings_summary(self.viewer.keymap)

        layer = self.viewer.active_layer
        if layer is not None:
            # Add class and instance layer keybindings for the active layer
            keybindings_str += get_keybindings_summary(layer.class_keymap)
            keybindings_str += get_keybindings_summary(layer.keymap)
        self.active_label.setText(keybindings_str)


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
        # layer_label.setAlignment(Qt.AlignLeft)
        layer_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(layer_label)
        self.setLayout(self.layout)
