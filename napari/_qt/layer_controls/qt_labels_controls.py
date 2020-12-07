import sys

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
    LabelBrushShape,
    LabelColorMode,
    Mode,
)
from ...utils.events import disconnect_events
from ..utils import disable_with_opacity
from ..widgets.qt_mode_buttons import QtModePushButton, QtModeRadioButton
from .qt_layer_controls_base import QtLayerControls


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
    selectionSpinBox : qtpy.QtWidgets.QSpinBox
        Widget to select a specific label by its index.

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
        self.layer.events.n_dimensional.connect(self._on_n_dimensional_change)
        self.layer.events.editable.connect(self._on_editable_change)
        self.layer.events.preserve_labels.connect(
            self._on_preserve_labels_change
        )
        self.layer.events.color_mode.connect(self._on_color_mode_change)

        # selection spinbox
        self.selectionSpinBox = QSpinBox()
        self.selectionSpinBox.setKeyboardTracking(False)
        self.selectionSpinBox.setSingleStep(1)
        self.selectionSpinBox.setMinimum(0)
        self.selectionSpinBox.setMaximum(2147483647)
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
        contig_cb.setToolTip('contiguous editing')
        contig_cb.stateChanged.connect(self.change_contig)
        self.contigCheckBox = contig_cb
        self._on_contiguous_change()

        ndim_cb = QCheckBox()
        ndim_cb.setToolTip('edit all dimensions')
        ndim_cb.stateChanged.connect(self.change_ndim)
        self.ndimCheckBox = ndim_cb
        self._on_n_dimensional_change()

        preserve_labels_cb = QCheckBox()
        preserve_labels_cb.setToolTip(
            'preserve existing labels while painting'
        )
        preserve_labels_cb.stateChanged.connect(self.change_preserve_labels)
        self.preserveLabelsCheckBox = preserve_labels_cb
        self._on_preserve_labels_change()

        selectedColorCheckbox = QCheckBox()
        selectedColorCheckbox.setToolTip("Display only selected label")
        selectedColorCheckbox.stateChanged.connect(self.toggle_selected_mode)
        self.selectedColorCheckbox = selectedColorCheckbox

        # shuffle colormap button
        self.colormapUpdate = QtModePushButton(
            None, 'shuffle', slot=self.changeColor, tooltip='shuffle colors',
        )

        self.panzoom_button = QtModeRadioButton(
            layer,
            'zoom',
            Mode.PAN_ZOOM,
            tooltip='Pan/zoom mode (Space)',
            checked=True,
        )
        self.pick_button = QtModeRadioButton(
            layer, 'picker', Mode.PICK, tooltip='Pick mode'
        )
        self.paint_button = QtModeRadioButton(
            layer, 'paint', Mode.PAINT, tooltip='Paint mode'
        )
        btn = 'Cmd' if sys.platform == 'darwin' else 'Ctrl'
        self.fill_button = QtModeRadioButton(
            layer, 'fill', Mode.FILL, tooltip=f'Fill mode ({btn})'
        )
        self.erase_button = QtModeRadioButton(
            layer, 'erase', Mode.ERASE, tooltip='Erase mode (Alt)'
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

        brush_shape_comboBox = QComboBox(self)
        brush_shape_comboBox.addItems(LabelBrushShape.keys())
        index = brush_shape_comboBox.findText(
            self.layer.brush_shape, Qt.MatchFixedString
        )
        brush_shape_comboBox.setCurrentIndex(index)
        brush_shape_comboBox.activated[str].connect(self.change_brush_shape)
        self.brushShapeComboBox = brush_shape_comboBox
        self._on_brush_shape_change()

        color_mode_comboBox = QComboBox(self)
        color_mode_comboBox.addItems(LabelColorMode.keys())
        index = color_mode_comboBox.findText(
            self.layer.color_mode, Qt.MatchFixedString
        )
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
        self.grid_layout.addWidget(QLabel('label:'), 1, 0, 1, 1)
        self.grid_layout.addLayout(color_layout, 1, 1, 1, 3)
        self.grid_layout.addWidget(QLabel('opacity:'), 2, 0, 1, 1)
        self.grid_layout.addWidget(self.opacitySlider, 2, 1, 1, 3)
        self.grid_layout.addWidget(QLabel('brush size:'), 3, 0, 1, 1)
        self.grid_layout.addWidget(self.brushSizeSlider, 3, 1, 1, 3)
        self.grid_layout.addWidget(QLabel('brush shape:'), 4, 0, 1, 1)
        self.grid_layout.addWidget(self.brushShapeComboBox, 4, 1, 1, 3)
        self.grid_layout.addWidget(QLabel('blending:'), 5, 0, 1, 1)
        self.grid_layout.addWidget(self.blendComboBox, 5, 1, 1, 3)
        self.grid_layout.addWidget(QLabel('color mode:'), 6, 0, 1, 1)
        self.grid_layout.addWidget(self.colorModeComboBox, 6, 1, 1, 3)
        self.grid_layout.addWidget(QLabel('contiguous:'), 7, 0, 1, 1)
        self.grid_layout.addWidget(self.contigCheckBox, 7, 1, 1, 1)
        self.grid_layout.addWidget(QLabel('n-dim:'), 7, 2, 1, 1)
        self.grid_layout.addWidget(self.ndimCheckBox, 7, 3, 1, 1)
        self.grid_layout.addWidget(QLabel('preserve labels:'), 8, 0, 1, 2)
        self.grid_layout.addWidget(self.preserveLabelsCheckBox, 8, 1, 1, 1)
        self.grid_layout.addWidget(QLabel('show selected:'), 8, 2, 1, 1)
        self.grid_layout.addWidget(self.selectedColorCheckbox, 8, 3, 1, 1)
        self.grid_layout.setRowStretch(9, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setSpacing(4)

    def mouseMoveEvent(self, event):
        """On mouse move, set layer status equal to the current selected mode.

        Available mode options are: PAN_ZOOM, PICKER, PAINT, ERASE or FILL

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self.layer.status = str(self.layer.mode)

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
            raise ValueError("Mode not recognized")

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

    def change_ndim(self, state):
        """Toggle n-dimensional state of label layer.

        Parameters
        ----------
        state : QCheckBox
            Checkbox indicating if label layer is n-dimensional.
        """
        if state == Qt.Checked:
            self.layer.n_dimensional = True
        else:
            self.layer.n_dimensional = False

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
        self.layer.color_mode = new_mode

    def change_brush_shape(self, brush_shape):
        """Change paintbrush shape of label layer.

        Parameters
        ----------
        brush_shape : str
            CIRCLE (default) uses circle paintbrush (case insensitive).
            SQUARE uses square paintbrush (case insensitive).
        """
        self.layer.brush_shape = brush_shape

    def _on_selected_label_change(self, event=None):
        """Receive layer model label selection change event and update spinbox.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        with self.layer.events.selected_label.blocker():
            value = self.layer.selected_label
            self.selectionSpinBox.setValue(int(value))

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

    def _on_n_dimensional_change(self, event=None):
        """Receive layer model n-dim mode change event and update the checkbox.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        with self.layer.events.n_dimensional.blocker():
            self.ndimCheckBox.setChecked(self.layer.n_dimensional)

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
            index = self.colorModeComboBox.findText(
                self.layer.color_mode, Qt.MatchFixedString
            )
            self.colorModeComboBox.setCurrentIndex(index)

    def _on_brush_shape_change(self, event=None):
        """Receive brush shape change event and update dropdown menu.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        with self.layer.events.brush_shape.blocker():
            index = self.brushShapeComboBox.findText(
                self.layer.brush_shape, Qt.MatchFixedString
            )
            self.brushShapeComboBox.setCurrentIndex(index)

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
        self.setToolTip('Selected label color')

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
