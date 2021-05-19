import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QWidget,
)

from ...layers.labels._labels_constants import (
    LABEL_COLOR_MODE_TRANSLATIONS,
    Mode,
)
from ...layers.labels._labels_utils import get_dtype
from ...utils.events import disconnect_events
from ...utils.interactions import Shortcut
from ...utils.translations import trans
from ..utils import disable_with_opacity
from ..widgets.qt_large_int_spinbox import QtLargeIntSpinBox
from ..widgets.qt_mode_buttons import QtModePushButton, QtModeRadioButton
from .qt_layer_controls_base import QtLayerControls

INT32_MAX = 2 ** 31 - 1


class QtLabelsControls(QtLayerControls):
    """Qt view and controls for the napari Labels layer.

    Parameters
    ----------
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group of labels layer modes: PAN_ZOOM, PICKER, PAINT, ERASE, or
        FILL.
    colormapUpdate : qtpy.QtWidgets.QPushButton
        Button to update colormap of label layer.
    contigCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if label layer is contiguous.
    fill_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select FILL mode on Labels layer.
    grid_layout : qtpy.QtWidgets.QGridLayout
        Layout of Qt widget controls for the layer.
    layer : napari.layers.Labels
        An instance of a napari Labels layer.
    ndimCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if label layer is n-dimensional.
    paint_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select PAINT mode on Labels layer.
    panzoom_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select PAN_ZOOM mode on Labels layer.
    pick_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select PICKER mode on Labels layer.
    erase_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select ERASE mode on Labels layer.
    selectionSpinBox : napari._qt.widgets.qt_large_int_spinbox.QtLargeIntSpinBox
        Widget to select a specific label by its index.
        N.B. cannot represent labels > 2**53.

    Raises
    ------
    ValueError
        Raise error if label mode is not PAN_ZOOM, PICKER, PAINT, ERASE, or
        FILL.
    """

    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.mode.connect(self._on_mode_change)
        self.layer.events.selected_label.connect(
            self._on_selected_label_change
        )
        self.layer.events.brush_size.connect(self._on_brush_size_change)
        self.layer.events.contiguous.connect(self._on_contiguous_change)
        self.layer.events.n_edit_dimensions.connect(
            self._on_n_edit_dimensions_change
        )
        self.layer.events.contour.connect(self._on_contour_change)
        self.layer.events.editable.connect(self._on_editable_change)
        self.layer.events.preserve_labels.connect(
            self._on_preserve_labels_change
        )
        self.layer.events.color_mode.connect(self._on_color_mode_change)

        # selection spinbox
        self.selectionSpinBox = QtLargeIntSpinBox()
        layer_dtype = get_dtype(layer)
        self.selectionSpinBox.set_dtype(layer_dtype)
        self.selectionSpinBox.setKeyboardTracking(False)
        self.selectionSpinBox.valueChanged.connect(self.changeSelection)
        self.selectionSpinBox.setAlignment(Qt.AlignCenter)
        self._on_selected_label_change()

        sld = QSlider(Qt.Horizontal)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(1)
        sld.setMaximum(40)
        sld.setSingleStep(1)
        sld.valueChanged.connect(self.changeSize)
        self.brushSizeSlider = sld
        self._on_brush_size_change()

        contig_cb = QCheckBox()
        contig_cb.setToolTip(trans._('contiguous editing'))
        contig_cb.stateChanged.connect(self.change_contig)
        self.contigCheckBox = contig_cb
        self._on_contiguous_change()

        ndim_sb = QSpinBox()
        self.ndimSpinBox = ndim_sb
        ndim_sb.setToolTip(trans._('number of dimensions for label editing'))
        ndim_sb.valueChanged.connect(self.change_n_edit_dim)
        ndim_sb.setMinimum(2)
        ndim_sb.setMaximum(self.layer.ndim)
        ndim_sb.setSingleStep(1)
        ndim_sb.setAlignment(Qt.AlignCenter)
        self._on_n_edit_dimensions_change()

        self.contourSpinBox = QtLargeIntSpinBox()
        self.contourSpinBox.set_dtype(layer_dtype)
        self.contourSpinBox.setToolTip(trans._('display contours of labels'))
        self.contourSpinBox.valueChanged.connect(self.change_contour)
        self.contourSpinBox.setKeyboardTracking(False)
        self.contourSpinBox.setAlignment(Qt.AlignCenter)
        self._on_contour_change()

        preserve_labels_cb = QCheckBox()
        preserve_labels_cb.setToolTip(
            trans._('preserve existing labels while painting')
        )
        preserve_labels_cb.stateChanged.connect(self.change_preserve_labels)
        self.preserveLabelsCheckBox = preserve_labels_cb
        self._on_preserve_labels_change()

        selectedColorCheckbox = QCheckBox()
        selectedColorCheckbox.setToolTip(
            trans._("Display only selected label")
        )
        selectedColorCheckbox.stateChanged.connect(self.toggle_selected_mode)
        self.selectedColorCheckbox = selectedColorCheckbox

        # shuffle colormap button
        self.colormapUpdate = QtModePushButton(
            None,
            'shuffle',
            slot=self.changeColor,
            tooltip=trans._('shuffle colors'),
        )

        self.panzoom_button = QtModeRadioButton(
            layer,
            'zoom',
            Mode.PAN_ZOOM,
            tooltip=trans._('Pan/zoom mode (Space)'),
            checked=True,
        )
        self.pick_button = QtModeRadioButton(
            layer, 'picker', Mode.PICK, tooltip=trans._('Pick mode (L)')
        )
        self.paint_button = QtModeRadioButton(
            layer, 'paint', Mode.PAINT, tooltip=trans._('Paint mode (P)')
        )
        self.fill_button = QtModeRadioButton(
            layer,
            'fill',
            Mode.FILL,
            tooltip=trans._(
                "Fill mode (F) \nToggle with {shortcut}",
                shortcut=Shortcut("Control"),
            ),
        )
        self.erase_button = QtModeRadioButton(
            layer,
            'erase',
            Mode.ERASE,
            tooltip=trans._(
                "Erase mode (E) \nToggle with {shortcut}",
                shortcut=Shortcut("Alt"),
            ),
        )

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.panzoom_button)
        self.button_group.addButton(self.paint_button)
        self.button_group.addButton(self.pick_button)
        self.button_group.addButton(self.fill_button)
        self.button_group.addButton(self.erase_button)
        self._on_editable_change()

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.colormapUpdate)
        button_row.addWidget(self.erase_button)
        button_row.addWidget(self.fill_button)
        button_row.addWidget(self.paint_button)
        button_row.addWidget(self.pick_button)
        button_row.addWidget(self.panzoom_button)
        button_row.setSpacing(4)
        button_row.setContentsMargins(0, 0, 0, 5)

        color_mode_comboBox = QComboBox(self)
        for index, (data, text) in enumerate(
            LABEL_COLOR_MODE_TRANSLATIONS.items()
        ):
            data = data.value
            color_mode_comboBox.addItem(text, data)

            if self.layer.color_mode == data:
                color_mode_comboBox.setCurrentIndex(index)

        color_mode_comboBox.activated[str].connect(self.change_color_mode)
        self.colorModeComboBox = color_mode_comboBox
        self._on_color_mode_change()

        color_layout = QHBoxLayout()
        self.colorBox = QtColorBox(layer)
        color_layout.addWidget(self.colorBox)
        color_layout.addWidget(self.selectionSpinBox)

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addLayout(button_row, 0, 0, 1, 4)
        self.grid_layout.addWidget(QLabel(trans._('label:')), 1, 0, 1, 1)
        self.grid_layout.addLayout(color_layout, 1, 1, 1, 3)
        self.grid_layout.addWidget(QLabel(trans._('opacity:')), 2, 0, 1, 1)
        self.grid_layout.addWidget(self.opacitySlider, 2, 1, 1, 3)
        self.grid_layout.addWidget(QLabel(trans._('brush size:')), 3, 0, 1, 1)
        self.grid_layout.addWidget(self.brushSizeSlider, 3, 1, 1, 3)
        self.grid_layout.addWidget(QLabel(trans._('blending:')), 5, 0, 1, 1)
        self.grid_layout.addWidget(self.blendComboBox, 5, 1, 1, 3)
        self.grid_layout.addWidget(QLabel(trans._('color mode:')), 6, 0, 1, 1)
        self.grid_layout.addWidget(self.colorModeComboBox, 6, 1, 1, 3)
        self.grid_layout.addWidget(QLabel(trans._('contour:')), 7, 0, 1, 1)
        self.grid_layout.addWidget(self.contourSpinBox, 7, 1, 1, 1)
        self.grid_layout.addWidget(QLabel(trans._('n edit dim:')), 8, 0, 1, 1)
        self.grid_layout.addWidget(self.ndimSpinBox, 8, 1, 1, 1)
        self.grid_layout.addWidget(QLabel(trans._('contiguous:')), 9, 0, 1, 1)
        self.grid_layout.addWidget(self.contigCheckBox, 9, 1, 1, 1)
        self.grid_layout.addWidget(
            QLabel(trans._('preserve labels:')), 10, 0, 1, 2
        )
        self.grid_layout.addWidget(self.preserveLabelsCheckBox, 10, 1, 1, 1)
        self.grid_layout.addWidget(
            QLabel(trans._('show selected:')), 10, 2, 1, 1
        )
        self.grid_layout.addWidget(self.selectedColorCheckbox, 10, 3, 1, 1)
        self.grid_layout.setRowStretch(10, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setSpacing(4)

    def _on_mode_change(self, event):
        """Receive layer model mode change event and update checkbox ticks.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not PAN_ZOOM, PICK, PAINT, ERASE, or
            FILL
        """
        mode = event.mode
        if mode == Mode.PAN_ZOOM:
            self.panzoom_button.setChecked(True)
        elif mode == Mode.PICK:
            self.pick_button.setChecked(True)
        elif mode == Mode.PAINT:
            self.paint_button.setChecked(True)
        elif mode == Mode.FILL:
            self.fill_button.setChecked(True)
        elif mode == Mode.ERASE:
            self.erase_button.setChecked(True)
        else:
            raise ValueError(trans._("Mode not recognized"))

    def changeColor(self):
        """Change colormap of the label layer."""
        self.layer.new_colormap()

    def changeSelection(self, value):
        """Change currently selected label.

        Parameters
        ----------
        value : int
            Index of label to select.
        """
        self.layer.selected_label = value
        self.selectionSpinBox.clearFocus()
        self.setFocus()

    def toggle_selected_mode(self, state):
        if state == Qt.Checked:
            self.layer.show_selected_label = True
        else:
            self.layer.show_selected_label = False

    def changeSize(self, value):
        """Change paint brush size.

        Parameters
        ----------
        value : float
            Size of the paint brush.
        """
        self.layer.brush_size = value

    def change_contig(self, state):
        """Toggle contiguous state of label layer.

        Parameters
        ----------
        state : QCheckBox
            Checkbox indicating if labels are contiguous.
        """
        if state == Qt.Checked:
            self.layer.contiguous = True
        else:
            self.layer.contiguous = False

    def change_n_edit_dim(self, value):
        """Change the number of editable dimensions of label layer.

        Parameters
        ----------
        value : int
            The number of editable dimensions to set.
        """
        self.layer.n_edit_dimensions = value
        self.ndimSpinBox.clearFocus()
        self.setFocus()

    def change_contour(self, value):
        """Change contour thickness.

        Parameters
        ----------
        value : int
            Thickness of contour.
        """
        self.layer.contour = value
        self.contourSpinBox.clearFocus()
        self.setFocus()

    def change_preserve_labels(self, state):
        """Toggle preserve_labels state of label layer.

        Parameters
        ----------
        state : QCheckBox
            Checkbox indicating if overwriting label is enabled.
        """
        if state == Qt.Checked:
            self.layer.preserve_labels = True
        else:
            self.layer.preserve_labels = False

    def change_color_mode(self, new_mode):
        """Change color mode of label layer.

        Parameters
        ----------
        new_mode : str
            AUTO (default) allows color to be set via a hash function with a seed.
            DIRECT allows color of each label to be set directly by a color dictionary.
        """
        self.layer.color_mode = self.colorModeComboBox.currentData()

    def _on_contour_change(self, event=None):
        """Receive layer model contour value change event and update spinbox.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        with self.layer.events.contour.blocker():
            value = self.layer.contour
            self.contourSpinBox.setValue(value)

    def _on_selected_label_change(self, event=None):
        """Receive layer model label selection change event and update spinbox.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        with self.layer.events.selected_label.blocker():
            value = self.layer.selected_label
            self.selectionSpinBox.setValue(value)

    def _on_brush_size_change(self, event=None):
        """Receive layer model brush size change event and update the slider.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        with self.layer.events.brush_size.blocker():
            value = self.layer.brush_size
            value = np.clip(int(value), 1, 40)
            self.brushSizeSlider.setValue(value)

    def _on_n_edit_dimensions_change(self, event=None):
        """Receive layer model n-dim mode change event and update the checkbox.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        with self.layer.events.n_edit_dimensions.blocker():
            value = self.layer.n_edit_dimensions
            self.ndimSpinBox.setValue(int(value))

    def _on_contiguous_change(self, event=None):
        """Receive layer model contiguous change event and update the checkbox.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        with self.layer.events.contiguous.blocker():
            self.contigCheckBox.setChecked(self.layer.contiguous)

    def _on_preserve_labels_change(self, event=None):
        """Receive layer model preserve_labels event and update the checkbox.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        with self.layer.events.preserve_labels.blocker():
            self.preserveLabelsCheckBox.setChecked(self.layer.preserve_labels)

    def _on_color_mode_change(self, event=None):
        """Receive layer model color.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        with self.layer.events.color_mode.blocker():
            self.colorModeComboBox.setCurrentIndex(
                self.colorModeComboBox.findData(self.layer.color_mode)
            )

    def _on_editable_change(self, event=None):
        """Receive layer model editable change event & enable/disable buttons.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        disable_with_opacity(
            self,
            ['pick_button', 'paint_button', 'fill_button'],
            self.layer.editable,
        )


class QtColorBox(QWidget):
    """A widget that shows a square with the current label color.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.
    """

    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.layer.events.selected_label.connect(
            self._on_selected_label_change
        )
        self.layer.events.opacity.connect(self._on_opacity_change)

        self.setAttribute(Qt.WA_DeleteOnClose)

        self._height = 24
        self.setFixedWidth(self._height)
        self.setFixedHeight(self._height)
        self.setToolTip(trans._('Selected label color'))

    def _on_selected_label_change(self, event):
        """Receive layer model label selection change event & update colorbox.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        self.update()

    def _on_opacity_change(self, event):
        """Receive layer model label selection change event & update colorbox.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        self.update()

    def paintEvent(self, event):
        """Paint the colorbox.  If no color, display a checkerboard pattern.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        painter = QPainter(self)
        if self.layer._selected_color is None:
            for i in range(self._height // 4):
                for j in range(self._height // 4):
                    if (i % 2 == 0 and j % 2 == 0) or (
                        i % 2 == 1 and j % 2 == 1
                    ):
                        painter.setPen(QColor(230, 230, 230))
                        painter.setBrush(QColor(230, 230, 230))
                    else:
                        painter.setPen(QColor(25, 25, 25))
                        painter.setBrush(QColor(25, 25, 25))
                    painter.drawRect(i * 4, j * 4, 5, 5)
        else:
            color = np.multiply(self.layer._selected_color, self.layer.opacity)
            color = np.round(255 * color).astype(int)
            painter.setPen(QColor(*list(color)))
            painter.setBrush(QColor(*list(color)))
            painter.drawRect(0, 0, self._height, self._height)

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.events, self)
        super().close()
