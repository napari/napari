from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QFrame, QGridLayout, QSlider

from ...layers.base._base_constants import BLENDING_TRANSLATIONS
from ...utils.events import disconnect_events


class QtLayerControls(QFrame):
    """Superclass for all the other LayerControl classes.

    This class is never directly instantiated anywhere.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    blendComboBox : qtpy.QtWidgets.QComboBox
        Drowpdown widget to select blending mode of layer.
    grid_layout : qtpy.QtWidgets.QGridLayout
        Layout of Qt widget controls for the layer.
    layer : napari.layers.Layer
        An instance of a napari layer.
    opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    """

    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.layer.events.blending.connect(self._on_blending_change)
        self.layer.events.opacity.connect(self._on_opacity_change)

        self.setAttribute(Qt.WA_DeleteOnClose)

        self.setObjectName('layer')
        self.setMouseTracking(True)

        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(2)
        self.grid_layout.setColumnMinimumWidth(0, 86)
        self.grid_layout.setColumnStretch(1, 1)
        self.setLayout(self.grid_layout)

        sld = QSlider(Qt.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.valueChanged.connect(self.changeOpacity)
        self.opacitySlider = sld
        self._on_opacity_change()

        blend_comboBox = QComboBox(self)
        for index, (data, text) in enumerate(BLENDING_TRANSLATIONS.items()):
            data = data.value
            blend_comboBox.addItem(text, data)
            if data == self.layer.blending:
                blend_comboBox.setCurrentIndex(index)

        blend_comboBox.activated[str].connect(self.changeBlending)
        self.blendComboBox = blend_comboBox

    def changeOpacity(self, value):
        """Change opacity value on the layer model.

        Parameters
        ----------
        value : float
            Opacity value for shapes.
            Input range 0 - 100 (transparent to fully opaque).
        """
        with self.layer.events.blocker(self._on_opacity_change):
            self.layer.opacity = value / 100

    def changeBlending(self, text):
        """Change blending mode on the layer model.

        Parameters
        ----------
        text : str
            Name of blending mode, eg: 'translucent', 'additive', 'opaque'.
        """
        self.layer.blending = self.blendComboBox.currentData()

    def _on_opacity_change(self, event=None):
        """Receive layer model opacity change event and update opacity slider.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        with self.layer.events.opacity.blocker():
            self.opacitySlider.setValue(int(self.layer.opacity * 100))

    def _on_blending_change(self, event=None):
        """Receive layer model blending mode change event and update slider.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        with self.layer.events.blending.blocker():
            self.blendComboBox.setCurrentIndex(
                self.blendComboBox.findData(self.layer.blending)
            )

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.events, self)
        for child in self.children():
            close_method = getattr(child, 'close', None)
            if close_method is not None:
                close_method()
        super().close()
