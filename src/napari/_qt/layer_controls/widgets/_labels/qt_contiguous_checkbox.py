from qtpy.QtWidgets import (
    QCheckBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import attr_to_settr, checked_to_bool
from napari.layers import Labels
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtContiguousCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer contiguous
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    contiguous_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if label layer is contiguous.
    contiguous_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the contiguous model chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Labels) -> None:
        super().__init__(parent, layer)
        # Setup widgets
        contig_cb = QCheckBox()
        contig_cb.setToolTip(trans._('Contiguous editing'))
        self._callbacks.append(
            attr_to_settr(
                self._layer,
                'contiguous',
                contig_cb,
                'setChecked',
            )
        )
        connect_setattr(
            contig_cb.stateChanged,
            layer,
            'contiguous',
            convert_fun=checked_to_bool,
        )
        self.contiguous_checkbox = contig_cb

        self.contiguous_checkbox_label = QtWrappedLabel(trans._('contiguous:'))

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.contiguous_checkbox_label, self.contiguous_checkbox)]
