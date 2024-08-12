from qtpy.QtWidgets import QGridLayout, QLabel, QWidget

from napari._qt.containers import QtListView
from napari._qt.containers.qt_axis_model import AxisList, AxisModel
from napari._qt.widgets.qt_tooltip import QtToolTipLabel
from napari.components import Dims
from napari.utils.translations import trans


def set_dims_order(dims: Dims, order: tuple[int, ...]):
    """Set dimension order of Dims object to order.

    Parameters
    ----------
    dims : napari.components.dims.Dims
        Dims object.
    order : tuple of int
        New dimension order.
    """
    if type(order[0]) is AxisModel:
        order = [a.axis for a in order]
    dims.order = order


class QtDimsSorter(QWidget):
    """Qt widget for dimension / axis reordering and locking.

    Modified from:
    https://github.com/jni/zarpaint/blob/main/zarpaint/_dims_chooser.py

    Parameters
    ----------
    viewer : napari.Viewer
        Main napari viewer instance.
    parent : QWidget
        QWidget that holds this widget.

    Attributes
    ----------
    dims : napari.components.Dims
        Dimensions object of the current viewer, modeling slicing and displaying.
    axis_list : napari._qt.containers.qt_axis_model.AxisList
        Selectable evented list representing the viewer axes.
    """

    def __init__(self, dims: Dims, parent: QWidget) -> None:
        super().__init__(parent=parent)
        self.dims = dims
        self.axis_list = AxisList.from_dims(self.dims)

        self.view = QtListView(self.axis_list)
        if len(self.axis_list) <= 2:
            # prevent excess space in popup
            self.view.setSizeAdjustPolicy(QtListView.AdjustToContents)

        layout = QGridLayout()
        self.setLayout(layout)

        widget_tooltip = QtToolTipLabel(self)
        widget_tooltip.setObjectName('help_label')
        widget_tooltip.setToolTip(
            trans._(
                'Drag dimensions to reorder, click lock icon to lock dimension in place.'
            )
        )

        widget_title = QLabel(trans._('Dims. Ordering'), self)

        self.layout().addWidget(widget_title, 0, 0)
        self.layout().addWidget(widget_tooltip, 0, 1)
        self.layout().addWidget(self.view, 1, 0, 1, 2)

        # connect axis_list and dims
        self.axis_list.events.reordered.connect(
            self._axis_list_reorder_callback,
        )
        self.dims.events.order.connect(
            self._dims_order_callback,
        )

    def _axis_list_reorder_callback(self, event):
        set_dims_order(self.dims, event.value)

    def _dims_order_callback(self, event):
        # Regenerate AxisList upon Dims side order changes for easy cleanup
        self.axis_list = AxisList.from_dims(self.dims)
        self.view.setRoot(self.axis_list)
