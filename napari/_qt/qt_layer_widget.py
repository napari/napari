from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFrame,
    QLineEdit,
    QLabel,
    QCheckBox,
    QHBoxLayout,
)
from qtpy.QtGui import QImage, QPixmap
from .utils import qt_signals_blocked
from ..utils.event import Event, EmitterGroup


class QtLayerWidget(QFrame):
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

    def __init__(self, layer, parent=None):
        super().__init__(parent=parent)

        self.events = EmitterGroup(
            source=self,
            auto_connect=False,
            selected=Event,
            name=Event,
            name_unique=Event,
            visible=Event,
        )

        # When the EVH refactor #1376 is done we might not even need the layer
        # attribute anymore as all data updates will be through the handler.
        # At that point we could remove the attribute and do the registering
        # and connecting outside this class and never even need to pass the
        # layer to this class.
        self.layer = layer
        self.layer.event_handler.register_listener(self)
        self.events.connect(self.layer.event_handler.on_change)

        self.setObjectName('layer')

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        tb = QLabel(self)
        tb.setObjectName('thumbnail')
        tb.setToolTip('Layer thumbnail')
        self.thumbnailLabel = tb
        self.layout.addWidget(tb)

        cb = QCheckBox(self)
        cb.setObjectName('visibility')
        cb.setToolTip('Layer visibility')
        cb.setProperty('mode', 'visibility')
        cb.stateChanged[int].connect(self.events.visible)
        self.visibleCheckBox = cb
        self.layout.addWidget(cb)

        self.nameTextBox = QLineEditDraggable(self)
        self.nameTextBox.setAcceptDrops(False)
        self.nameTextBox.editingFinished.connect(self.changeText)
        self.nameTextBox.old_name = ''
        self.layout.addWidget(self.nameTextBox)

        ltb = QLabel(self)
        ltb.setProperty('layer_type_label', True)
        ltb.setToolTip('Layer type')
        self.typeLabel = ltb
        self.layout.addWidget(ltb)

        msg = 'Click to select\nDrag to rearrange'
        self.setToolTip(msg)

        # Once EVH refactor is done, these can be moved to an initialization
        # outside of this object. This initialization could be done in the
        # EVH itself or the object that creates this one.
        self._set_layer_type(type(self.layer).__name__)
        self._on_selected_change(self.layer.selected)
        self._on_thumbnail_change(self.layer.thumbnail)
        self._on_visible_change(self.layer.visible)
        self._on_name_change(self.layer.name)

    def _set_layer_type(self, layer_type):
        """Set layer type.

        Parameters
        ----------
        layer_type : str
            Type of layer, must be one of the napari supported layer types.
        """
        self.typeLabel.setObjectName(layer_type)

    def _on_selected_change(self, selected):
        """Update selected status of the layer widget.

        Parameters
        ----------
        selected : bool
            Layer selected status.
        """
        self.setProperty('selected', selected)
        self.nameTextBox.setEnabled(selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def changeText(self):
        """Update layer name attribute using layer name textbox contents."""
        name = self.nameTextBox.text()
        old_name = self.nameTextBox.old_name
        if old_name == name:
            return
        self.events.name_unique((old_name, name))
        self.nameTextBox.old_name = self.nameTextBox.text()

        # Prevent retriggering during clearing of focus
        with qt_signals_blocked(self.nameTextBox):
            self.nameTextBox.clearFocus()
            self.setFocus()

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

    def _on_name_change(self, name):
        """Update text displaying name of layer.

        Parameters
        ----------
        text : str
            Name of the layer.
        """
        self.nameTextBox.setText(name)
        self.nameTextBox.setToolTip(name)
        self.nameTextBox.home(False)
        self.nameTextBox.old_name = name

    def _on_visible_change(self, visible):
        """Toggle visibility of the layer.

        Parameters
        ----------
        visible : bool
            Layer visibility.
        """
        self.visibleCheckBox.setChecked(visible)

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


class QLineEditDraggable(QLineEdit):
    """Draggable QLineEdit for better use inside a QListWidget."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFocusPolicy(Qt.NoFocus)
        # Use the _pressed property to only transfer focus to the QLineEdit
        # on a release event after there has been a click event on the widget.
        # This extra step allows for the initial drag on the QLineEdit to move
        # the widget instead of select the text. After release the text would
        # become selectable if there was no drag and drop.
        self._pressed = False

    def mousePressEvent(self, event):
        if self.hasFocus() and self._pressed:
            super().mousePressEvent(event)
        else:
            event.accept()
        self._pressed = True

    def mouseReleaseEvent(self, event):
        if self._pressed:
            self.setFocus()
        self._pressed = False
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self.hasFocus() and self._pressed:
            super().mouseMoveEvent(event)
        else:
            event.ignore()
