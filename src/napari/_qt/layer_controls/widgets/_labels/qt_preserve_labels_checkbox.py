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


class QtPreserveLabelsCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer attribute to
    preserve existing labels and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    preserve_labels_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if existing labels are preserved.
    preserve_labels_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the layer should preserve labels chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Labels) -> None:
        super().__init__(parent, layer)
        # Setup widgets
        preserve_labels_cb = QCheckBox()
        preserve_labels_cb.setToolTip(
            trans._('Preserve existing labels while painting')
        )
        preserve_labels_cb.setChecked(self._layer.preserve_labels)
        self._callbacks.append(
            attr_to_settr(
                self._layer,
                'preserve_labels',
                preserve_labels_cb,
                'setChecked',
            )
        )
        connect_setattr(
            preserve_labels_cb.stateChanged,
            layer,
            'preserve_labels',
            convert_fun=checked_to_bool,
        )
        self.preserve_labels_checkbox = preserve_labels_cb

        self.preserve_labels_checkbox_label = QtWrappedLabel(
            trans._('preserve\nlabels:')
        )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (
                self.preserve_labels_checkbox_label,
                self.preserve_labels_checkbox,
            )
        ]
