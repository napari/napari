from collections import OrderedDict

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QTextEdit,
    QVBoxLayout,
)

import napari

from ...utils.interactions import get_key_bindings_summary
from ...utils.theme import get_theme
from ...utils.translations import trans


class QtAboutKeyBindings(QDialog):
    """Qt dialog window for displaying keybinding information.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.

    key_map_handler : napari.utils.key_bindings.KeyMapHandler
        Handler for key mapping and calling functionality.

    Attributes
    ----------
    key_bindings_strs : collections.OrderedDict
        Ordered dictionary of hotkey shortcuts and associated key bindings.
        Dictionary keys include:
        - 'All active key bindings'
        - 'Image layer'
        - 'Labels layer'
        - 'Points layer'
        - 'Shapes layer'
        - 'Surface layer'
        - 'Vectors layer'
    layout : qtpy.QtWidgets.QVBoxLayout
        Layout of the widget.
    layerTypeComboBox : qtpy.QtWidgets.QComboBox
        Dropdown menu to select layer type.
    textEditBox : qtpy.QtWidgets.QTextEdit
        Text box widget containing table of key bindings information.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    """

    ALL_ACTIVE_KEYBINDINGS = trans._('All active key bindings')

    def __init__(self, viewer, key_map_handler, parent=None):
        super().__init__(parent=parent)

        self.viewer = viewer
        self.layout = QVBoxLayout()

        self.setWindowTitle(trans._('Keybindings'))
        self.setWindowModality(Qt.NonModal)
        self.setLayout(self.layout)

        # stacked key bindings widgets
        self.textEditBox = QTextEdit()
        self.textEditBox.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.textEditBox.setMinimumWidth(360)
        # Can switch to a normal dict when our minimum Python is 3.7
        self.key_bindings_strs = OrderedDict()
        self.key_bindings_strs[self.ALL_ACTIVE_KEYBINDINGS] = ''
        self.key_map_handler = key_map_handler
        theme = get_theme(self.viewer.theme)
        col = theme['secondary']
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
                text = trans._('No key bindings')
            else:
                text = get_key_bindings_summary(layer.class_keymap, col=col)
            self.key_bindings_strs[f"{layer.__name__} layer"] = text

        # layer type selection
        self.layerTypeComboBox = QComboBox()
        self.layerTypeComboBox.addItems(list(self.key_bindings_strs))
        self.layerTypeComboBox.activated[str].connect(self.change_layer_type)
        self.layerTypeComboBox.setCurrentText(self.ALL_ACTIVE_KEYBINDINGS)
        # self.change_layer_type(current_layer)
        layer_type_layout = QHBoxLayout()
        layer_type_layout.setContentsMargins(10, 5, 0, 0)
        layer_type_layout.addWidget(self.layerTypeComboBox)
        layer_type_layout.addStretch(1)
        layer_type_layout.setSpacing(0)
        self.layout.addLayout(layer_type_layout)
        self.layout.addWidget(self.textEditBox, 1)

        self.viewer.events.theme.connect(self.update_active_layer)
        self.update_active_layer()

    def change_layer_type(self, text):
        """Change layer type selected in dropdown menu.

        Parameters
        ----------
        text : str
            Dictionary key to access key bindings associated with the layer.
            Available keys include:
            - 'All active key bindings'
            - 'Image layer'
            - 'Labels layer'
            - 'Points layer'
            - 'Shapes layer'
            - 'Surface layer'
            - 'Vectors layer'
        """
        self.textEditBox.setHtml(self.key_bindings_strs[text])

    def update_active_layer(self, event=None):
        """Update the active layer and display key bindings for that layer type.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        theme = get_theme(self.viewer.theme)
        col = theme['secondary']
        # Add class and instance viewer key bindings
        text = get_key_bindings_summary(
            self.key_map_handler.active_keymap, col=col
        )
        # Update layer speficic key bindings if all active are displayed
        self.key_bindings_strs[self.ALL_ACTIVE_KEYBINDINGS] = text
        if self.layerTypeComboBox.currentText() == self.ALL_ACTIVE_KEYBINDINGS:
            self.textEditBox.setHtml(text)
