from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
)
from superqt import QLargeIntSpinBox

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import attr_to_settr
from napari.layers import Labels
from napari.layers.labels._labels_utils import get_dtype
from napari.utils._dtype import get_dtype_limits
from napari.utils.translations import trans


class QtContourSpinBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer contour
    thickness attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    contour_spinbox : superqt.QLargeSpinBox
        Spinbox to control the layer contour thickness.
    contour_spinbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the layer contour thickness chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Labels) -> None:
        super().__init__(parent, layer)
        # Setup widgets
        self.contour_spinbox = QLargeIntSpinBox()
        dtype_lims = get_dtype_limits(get_dtype(layer))
        self.contour_spinbox.setRange(0, dtype_lims[1])
        self.contour_spinbox.setToolTip(
            trans._('Set width of displayed label contours')
        )
        self.contour_spinbox.setValue(self._layer.contour)
        self.contour_spinbox.valueChanged.connect(self.change_contour)
        self.contour_spinbox.setKeyboardTracking(False)
        self.contour_spinbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._callbacks.append(
            attr_to_settr(
                self._layer,
                'contour',
                self.contour_spinbox,
                'setValue',
            )
        )

        self.contour_spinbox_label = QtWrappedLabel(trans._('contour:'))

    def change_contour(self, value: int) -> None:
        """Change contour thickness.
        Parameters
        ----------
        value : int
            Thickness of contour.
        """
        self._layer.contour = value
        self.contour_spinbox.clearFocus()
        self.parent().setFocus()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.contour_spinbox_label, self.contour_spinbox)]
