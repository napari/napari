from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QWidget,
)
from superqt import QLabeledDoubleSlider

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import attr_to_settr
from napari.layers.base._base_constants import BLENDING_TRANSLATIONS, Blending
from napari.layers.base.base import Layer
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans

# opaque and minimum blending do not support changing alpha (opacity)
NO_OPACITY_BLENDING_MODES = {str(Blending.MINIMUM), str(Blending.OPAQUE)}


class QtOpacityBlendingControls(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between opacity/blending
    layer attributes and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    blend_combobox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select blending mode of layer.
    blend_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the blending combobox widget.
    opacity_slider : superqt.QLabeledDoubleSlider
        Slider controlling opacity of the layer.
    opacity_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the opacity slider widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.blending.connect(self._on_blending_change)

        # Setup widgets
        sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent=parent)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(1)
        sld.setSingleStep(0.01)
        sld.setValue(self._layer.opacity)
        self.opacity_slider = sld
        connect_setattr(
            self.opacity_slider.valueChanged, self._layer, 'opacity'
        )
        self._callbacks.append(
            attr_to_settr(
                self._layer, 'opacity', self.opacity_slider, 'setValue'
            )
        )
        self.opacity_label = QtWrappedLabel(trans._('opacity:'))

        blend_combobox = QComboBox(parent)
        for index, (data, text) in enumerate(BLENDING_TRANSLATIONS.items()):
            data = data.value
            blend_combobox.addItem(text, data)
            if data == self._layer.blending:
                blend_combobox.setCurrentIndex(index)

        blend_combobox.currentTextChanged.connect(self.change_blending)
        self.blend_combobox = blend_combobox
        self.blend_label = QtWrappedLabel(trans._('blending:'))

        # opaque and minimum blending do not support changing alpha
        self.opacity_slider.setEnabled(
            self._layer.blending not in NO_OPACITY_BLENDING_MODES
        )
        self.opacity_label.setEnabled(
            self._layer.blending not in NO_OPACITY_BLENDING_MODES
        )

    def change_blending(self, text: str) -> None:
        """Change blending mode on the layer model.

        Parameters
        ----------
        text : str
            Name of blending mode, eg: 'translucent', 'additive', 'opaque'.
        """
        self._layer.blending = self.blend_combobox.currentData()
        # opaque and minimum blending do not support changing alpha
        self.opacity_slider.setEnabled(
            self._layer.blending not in NO_OPACITY_BLENDING_MODES
        )
        self.opacity_label.setEnabled(
            self._layer.blending not in NO_OPACITY_BLENDING_MODES
        )

        blending_tooltip = ''
        if self._layer.blending == Blending.MINIMUM:
            blending_tooltip = trans._(
                '`minimum` blending mode works best with inverted colormaps with a white background.',
            )
        self.blend_combobox.setToolTip(blending_tooltip)
        self._layer.help = blending_tooltip

    def _on_blending_change(self) -> None:
        """Receive layer model blending mode change event and update slider."""
        with self._layer.events.blending.blocker():
            self.blend_combobox.setCurrentIndex(
                self.blend_combobox.findData(self._layer.blending)
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.opacity_label, self.opacity_slider),
            (self.blend_label, self.blend_combobox),
        ]
