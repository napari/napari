from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QWidget
from superqt import QLabeledDoubleSlider

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import attr_to_settr, qt_signals_blocked
from napari.layers import Image
from napari.layers.image._image_constants import (
    ImageRendering,
)
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtImageRenderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer attribute for
    the method to render an image and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Image
        An instance of a napari Image layer.

    Attributes
    ----------
    render_combobox : qtpy.QtWidgets.QComboBox
        Combobox to control labels render method.
    render_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the way labels should be rendered chooser widget.
    iso_threshold_slider : superqt.QLabeledDoubleSlider
        Slider controlling the isosurface threshold value for rendering.
    iso_threshold_label : qtpy.QtWidgets.QLabel
        Label for the isosurface threshold slider widget.
    attenuation_slider : superqt.QLabeledDoubleSlider
        Slider controlling attenuation rate for `attenuated_mip` mode.
    attenuation_label : qtpy.QtWidgets.QLabel
        Label for the attenuation slider widget.
    """

    def __init__(self, parent: QWidget, layer: Image) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.rendering.connect(self._on_rendering_change)
        self._layer.events.contrast_limits.connect(
            self._on_contrast_limits_change
        )

        # Setup widgets
        render_combobox = QComboBox(parent)
        rendering_options = [i.value for i in ImageRendering]
        render_combobox.addItems(rendering_options)
        index = render_combobox.findText(
            self._layer.rendering, Qt.MatchFlag.MatchFixedString
        )
        render_combobox.setCurrentIndex(index)
        render_combobox.currentTextChanged.connect(self.change_rendering)
        self.render_combobox = render_combobox

        self.render_label = QtWrappedLabel(trans._('rendering:'))

        sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent=parent)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        cmin, cmax = self._layer.contrast_limits_range
        sld.setMinimum(cmin)
        sld.setMaximum(cmax)
        sld.setValue(self._layer.iso_threshold)
        connect_setattr(sld.valueChanged, self._layer, 'iso_threshold')
        self._callbacks.append(
            attr_to_settr(self._layer, 'iso_threshold', sld, 'setValue')
        )
        self.iso_threshold_slider = sld

        self.iso_threshold_label = QtWrappedLabel(trans._('iso threshold:'))

        sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent=parent)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(0.5)
        sld.setSingleStep(0.001)
        sld.setValue(self._layer.attenuation)
        sld.setDecimals(3)
        connect_setattr(sld.valueChanged, self._layer, 'attenuation')
        self._callbacks.append(
            attr_to_settr(self._layer, 'attenuation', sld, 'setValue')
        )
        self.attenuation_slider = sld

        self.attenuation_label = QtWrappedLabel(trans._('attenuation:'))

    def change_rendering(self, text):
        """Change rendering mode for image display.

        Parameters
        ----------
        text : str
            Rendering mode used by vispy.
            Selects a preset rendering mode in vispy that determines how
            volume is displayed:
            * translucent: voxel colors are blended along the view ray until
              the result is opaque.
            * mip: maximum intensity projection. Cast a ray and display the
              maximum value that was encountered.
            * additive: voxel colors are added along the view ray until
              the result is saturated.
            * iso: isosurface. Cast a ray until a certain threshold is
              encountered. At that location, lighning calculations are
              performed to give the visual appearance of a surface.
            * attenuated_mip: attenuated maximum intensity projection. Cast a
              ray and attenuate values based on integral of encountered values,
              display the maximum value that was encountered after attenuation.
              This will make nearer objects appear more prominent.
        """
        self._layer.rendering = text
        self._update_rendering_parameter_visibility()

    def _on_rendering_change(self):
        """Receive layer model rendering change event and update dropdown menu."""
        with qt_signals_blocked(self.render_combobox):
            index = self.render_combobox.findText(
                self._layer.rendering, Qt.MatchFlag.MatchFixedString
            )
            self.render_combobox.setCurrentIndex(index)
            self._update_rendering_parameter_visibility()

    def _on_contrast_limits_change(self):
        with qt_signals_blocked(self.iso_threshold_slider):
            cmin, cmax = self._layer.contrast_limits_range
            self.iso_threshold_slider.setMinimum(cmin)
            self.iso_threshold_slider.setMaximum(cmax)

    def _on_display_change_hide(self):
        self.iso_threshold_slider.hide()
        self.iso_threshold_label.hide()
        self.attenuation_slider.hide()
        self.attenuation_label.hide()
        self.render_combobox.hide()
        self.render_label.hide()

    def _on_display_change_show(self):
        self.render_combobox.show()
        self.render_label.show()
        self._update_rendering_parameter_visibility()

    def _update_rendering_parameter_visibility(self):
        """Hide isosurface rendering parameters if they aren't needed."""
        rendering = ImageRendering(self._layer.rendering)
        iso_threshold_visible = rendering == ImageRendering.ISO
        self.iso_threshold_label.setVisible(iso_threshold_visible)
        self.iso_threshold_slider.setVisible(iso_threshold_visible)
        attenuation_visible = rendering == ImageRendering.ATTENUATED_MIP
        self.attenuation_slider.setVisible(attenuation_visible)
        self.attenuation_label.setVisible(attenuation_visible)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.render_label, self.render_combobox),
            (self.iso_threshold_label, self.iso_threshold_slider),
            (self.attenuation_label, self.attenuation_slider),
        ]
