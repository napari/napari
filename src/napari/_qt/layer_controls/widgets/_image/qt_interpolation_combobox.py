from qtpy.QtWidgets import (
    QComboBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Image
from napari.layers.image._image_constants import Interpolation
from napari.utils.translations import trans


class QtInterpolationComboBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer interpolation
    mode attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Image
        An instance of a napari Image layer.

    Attributes
    ----------
    interpolation_combobox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current shading value of the layer.
    interpolation_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the shading value chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Image) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.interpolation2d.connect(
            self._on_interpolation_change
        )
        self._layer.events.interpolation3d.connect(
            self._on_interpolation_change
        )

        # Setup widgets
        self.interpolation_combobox = QComboBox(parent)
        self.interpolation_combobox.currentTextChanged.connect(
            self.change_interpolation
        )
        self.interpolation_combobox.setToolTip(
            trans._(
                'Texture interpolation for display.\nnearest and linear are most performant.'
            )
        )

        self.interpolation_combobox_label = QtWrappedLabel(
            trans._('interpolation:')
        )

    def change_interpolation(self, text: str) -> None:
        """Change interpolation mode for image display.

        Parameters
        ----------
        text : str
            Interpolation mode used by vispy. Must be one of our supported
            modes:
            'bessel', 'bicubic', 'linear', 'blackman', 'catrom', 'gaussian',
            'hamming', 'hanning', 'hermite', 'kaiser', 'lanczos', 'mitchell',
            'nearest', 'spline16', 'spline36'
        """
        # TODO: Better way to handle the ndisplay value?
        if self.parent().ndisplay == 2:
            self._layer.interpolation2d = text
        else:
            self._layer.interpolation3d = text

    def _on_interpolation_change(self, event) -> None:
        """Receive layer interpolation change event and update dropdown menu.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        interp_string = event.value.value

        with qt_signals_blocked(self.interpolation_combobox):
            if self.interpolation_combobox.findText(interp_string) == -1:
                self.interpolation_combobox.addItem(interp_string)
            self.interpolation_combobox.setCurrentText(interp_string)

    def _update_interpolation_combo(self, ndisplay: int) -> None:
        interp_names = [i.value for i in Interpolation.view_subset()]
        interp = (
            self._layer.interpolation2d
            if ndisplay == 2
            else self._layer.interpolation3d
        )
        with qt_signals_blocked(self.interpolation_combobox):
            self.interpolation_combobox.clear()
            self.interpolation_combobox.addItems(interp_names)
            self.interpolation_combobox.setCurrentText(interp)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.interpolation_combobox_label, self.interpolation_combobox)
        ]
