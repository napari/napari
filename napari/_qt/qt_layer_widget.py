from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QFrame,
    QLineEdit,
    QLabel,
    QCheckBox,
    QHBoxLayout,
)
from copy import copy
from ..layers.base._base_layer_interface import BaseLayerInterface
from ..utils.event import Event, EmitterGroup


class QtLayerWidget(QFrame, BaseLayerInterface):
    """Qt view for Layer model.

    Attributes
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.
    layout : QVBoxLayout
        Layout of the widget.
    nameTextBox : QLineEdit
        Textbox for layer name.
    thumbnailLabel : QLabel
        Label of layer thumbnail.
    typeLabel : QLabel
        Label of layer type.
    visibleCheckBox : QCheckBox
        Checkbox to toggle layer visibility.
    """

    def __init__(self, layer):
        super().__init__()

        self.layer = layer

        self.layer.event_handler.register_component_to_update(self)
        self.events = EmitterGroup(
            source=self,
            selected=Event,
            name=Event,
            name_unique=Event,
            visible=Event,
            event_handler_callback=self.layer.event_handler.on_change,
        )

        self.setObjectName('layer')

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        tb = QLabel(self)
        tb.setObjectName('thumbnail')
        tb.setToolTip('Layer thumbnail')
        self.thumbnailLabel = tb
        self._on_thumbnail_change(self.layer.thumbnail)
        self.layout.addWidget(tb)

        cb = QCheckBox(self)
        cb.setObjectName('visibility')
        cb.setToolTip('Layer visibility')
        cb.setChecked(self.layer.visible)
        cb.setProperty('mode', 'visibility')
        cb.stateChanged[int].connect(self.events.visible)
        self.visibleCheckBox = cb
        self.layout.addWidget(cb)

        textbox = QLineEdit(self)
        textbox.setText(layer.name)
        textbox.home(False)
        textbox.setToolTip(self.layer.name)
        textbox.setAcceptDrops(False)
        textbox.setEnabled(True)
        textbox.editingFinished.connect(self.changeText)
        self.nameTextBox = textbox
        self.layout.addWidget(textbox)

        ltb = QLabel(self)
        layer_type = type(layer).__name__
        ltb.setObjectName(layer_type)
        ltb.setProperty('layer_type_label', True)
        ltb.setToolTip('Layer type')
        self.typeLabel = ltb
        self.layout.addWidget(ltb)

        msg = 'Click to select\nDrag to rearrange'
        self.setToolTip(msg)
        self._on_selected_change(self.layer.selected)

    def _on_selected_change(self, state):
        """Select layer widget.

        Parameters
        ----------
        state : bool
        """
        self.setProperty('selected', state)
        self.nameTextBox.setEnabled(state)
        self.style().unpolish(self)
        self.style().polish(self)

    def changeText(self):
        """Update layer name attribute using layer name textbox contents."""
        value = self.nameTextBox.text()
        if self.layer.name == value:
            return
        old_name = copy(self.layer.name)
        self.events.name_unique(value=(old_name, value))
        if self.layer.name == old_name:
            self.events.name(value=value)
        self.nameTextBox.setToolTip(self.layer.name)
        # Prevent retriggering during clearing of focus
        self.nameTextBox.blockSignals(True)
        self.nameTextBox.clearFocus()
        self.setFocus()
        self.nameTextBox.blockSignals(False)

    def mouseReleaseEvent(self, event):
        """Ignores mouse release event.

        Allows the event to pass through a parent widget to its child widget
        without doing anything. If we did not use event.ignore() then the
        parent widget would catch the event and not pass it on to the child.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        event.ignore()

    def mousePressEvent(self, event):
        """Ignores mouse press event.

        Allows the event to pass through a parent widget to its child widget
        without doing anything. If we did not use event.ignore() then the
        parent widget would catch the event and not pass it on to the child.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        event.ignore()

    def mouseMoveEvent(self, event):
        """Ignores mouse move event.

        Allows the event to pass through a parent widget to its child widget
        without doing anything. If we did not use event.ignore() then the
        parent widget would catch the event and not pass it on to the child.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        event.ignore()

    def _on_name_change(self, text):
        """Update text displaying name of layer.

        Parameters
        ----------
        text : str
            Name of the layer.
        """
        self.nameTextBox.setText(text)
        self.nameTextBox.home(False)

    def _on_visible_change(self, state):
        """Toggle visibility of the layer.

        Parameters
        ----------
        state : bool
            Layer visibility.
        """
        self.visibleCheckBox.setChecked(state)

    def _on_thumbnail_change(self, thumbnail):
        """Update thumbnail image on the layer widget.

        Parameters
        ----------
        thumbnail : ndarray
            Thumbnail in RGBA unit8 format.
        """
        # Note that QImage expects the image width followed by height
        image = QImage(
            thumbnail,
            thumbnail.shape[1],
            thumbnail.shape[0],
            QImage.Format_RGBA8888,
        )
        self.thumbnailLabel.setPixmap(QPixmap.fromImage(image))
