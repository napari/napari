from collections.abc import Iterable

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGridLayout,
    QLabel,
    QSlider,
)

from ...layers.shapes._shapes_constants import Mode
from ...utils.events import disconnect_events
from ..utils import disable_with_opacity, qt_signals_blocked
from ..widgets.qt_color_swatch import QColorSwatchEdit
from ..widgets.qt_mode_buttons import QtModePushButton, QtModeRadioButton
from .qt_layer_controls_base import QtLayerControls


class QtShapesControls(QtLayerControls):
    """Qt view and controls for the napari Shapes layer.

    Parameters
    ----------
    layer : napari.layers.Shapes
        An instance of a napari Shapes layer.

    Attributes
    ----------
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for shapes layer modes
        (SELECT, DIRECT, PAN_ZOOM, ADD_RECTANGLE, ADD_ELLIPSE, ADD_LINE,
        ADD_PATH, ADD_POLYGON, VERTEX_INSERT, VERTEX_REMOVE).
    delete_button : qtpy.QtWidgets.QtModePushButton
        Button to delete selected shapes
    direct_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select individual vertices in shapes.
    edgeColorSwatch : qtpy.QtWidgets.QFrame
        Thumbnail display of points edge color.
    edgeComboBox : qtpy.QtWidgets.QComboBox
        Drop down list allowing user to set edge color of points.
    ellipse_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add ellipses to shapes layer.
    faceColorSwatch : qtpy.QtWidgets.QFrame
        Thumbnail display of points face color.
    faceComboBox : qtpy.QtWidgets.QComboBox
        Drop down list allowing user to set face color of points.
    grid_layout : qtpy.QtWidgets.QGridLayout
        Layout of Qt widget controls for the layer.
    layer : napari.layers.Shapes
        An instance of a napari Shapes layer.
    line_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add lines to shapes layer.
    move_back_button : qtpy.QtWidgets.QtModePushButton
        Button to move selected shape(s) to the back.
    move_front_button : qtpy.QtWidgets.QtModePushButton
        Button to move shape(s) to the front.
    panzoom_button : qtpy.QtWidgets.QtModeRadioButton
        Button to pan/zoom shapes layer.
    path_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add paths to shapes layer.
    polygon_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add polygons to shapes layer.
    rectangle_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add rectangles to shapes layer.
    select_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select shapes.
    vertex_insert_button : qtpy.QtWidgets.QtModeRadioButton
        Button to insert vertex into shape.
    vertex_remove_button : qtpy.QtWidgets.QtModeRadioButton
        Button to remove vertex from shapes.
    widthSlider : qtpy.QtWidgets.QSlider
        Slider controlling line edge width of shapes.

    Raises
    ------
    ValueError
        Raise error if shapes mode is not recognized.
    """

    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.mode.connect(self._on_mode_change)
        self.layer.events.edge_width.connect(self._on_edge_width_change)
        self.layer.events.current_edge_color.connect(
            self._on_current_edge_color_change
        )
        self.layer.events.current_face_color.connect(
            self._on_current_face_color_change
        )
        self.layer.events.editable.connect(self._on_editable_change)
        self.layer.text.events.visible.connect(self._on_text_visibility_change)

        sld = QSlider(Qt.Horizontal)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(40)
        sld.setSingleStep(1)
        value = self.layer.current_edge_width
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()
        sld.setValue(int(value))
        sld.valueChanged.connect(self.changeWidth)
        self.widthSlider = sld

        self.select_button = QtModeRadioButton(
            layer, 'select', Mode.SELECT, tooltip='Select shapes'
        )
        self.direct_button = QtModeRadioButton(
            layer, 'direct', Mode.DIRECT, tooltip='Select vertices'
        )
        self.panzoom_button = QtModeRadioButton(
            layer, 'zoom', Mode.PAN_ZOOM, tooltip='Pan/zoom', checked=True
        )
        self.rectangle_button = QtModeRadioButton(
            layer, 'rectangle', Mode.ADD_RECTANGLE, tooltip='Add rectangles'
        )
        self.ellipse_button = QtModeRadioButton(
            layer, 'ellipse', Mode.ADD_ELLIPSE, tooltip='Add ellipses'
        )
        self.line_button = QtModeRadioButton(
            layer, 'line', Mode.ADD_LINE, tooltip='Add lines'
        )
        self.path_button = QtModeRadioButton(
            layer, 'path', Mode.ADD_PATH, tooltip='Add paths'
        )
        self.polygon_button = QtModeRadioButton(
            layer, 'polygon', Mode.ADD_POLYGON, tooltip='Add polygons'
        )
        self.vertex_insert_button = QtModeRadioButton(
            layer, 'vertex_insert', Mode.VERTEX_INSERT, tooltip='Insert vertex'
        )
        self.vertex_remove_button = QtModeRadioButton(
            layer, 'vertex_remove', Mode.VERTEX_REMOVE, tooltip='Remove vertex'
        )

        self.move_front_button = QtModePushButton(
            layer,
            'move_front',
            slot=self.layer.move_to_front,
            tooltip='Move to front',
        )
        self.move_back_button = QtModePushButton(
            layer,
            'move_back',
            slot=self.layer.move_to_back,
            tooltip='Move to back',
        )
        self.delete_button = QtModePushButton(
            layer,
            'delete_shape',
            slot=self.layer.remove_selected,
            tooltip='Delete selected shapes',
        )

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.select_button)
        self.button_group.addButton(self.direct_button)
        self.button_group.addButton(self.panzoom_button)
        self.button_group.addButton(self.rectangle_button)
        self.button_group.addButton(self.ellipse_button)
        self.button_group.addButton(self.line_button)
        self.button_group.addButton(self.path_button)
        self.button_group.addButton(self.polygon_button)
        self.button_group.addButton(self.vertex_insert_button)
        self.button_group.addButton(self.vertex_remove_button)

        button_grid = QGridLayout()
        button_grid.addWidget(self.vertex_remove_button, 0, 2)
        button_grid.addWidget(self.vertex_insert_button, 0, 3)
        button_grid.addWidget(self.delete_button, 0, 4)
        button_grid.addWidget(self.direct_button, 0, 5)
        button_grid.addWidget(self.select_button, 0, 6)
        button_grid.addWidget(self.panzoom_button, 0, 7)
        button_grid.addWidget(self.move_back_button, 1, 1)
        button_grid.addWidget(self.move_front_button, 1, 2)
        button_grid.addWidget(self.ellipse_button, 1, 3)
        button_grid.addWidget(self.rectangle_button, 1, 4)
        button_grid.addWidget(self.polygon_button, 1, 5)
        button_grid.addWidget(self.line_button, 1, 6)
        button_grid.addWidget(self.path_button, 1, 7)
        button_grid.setContentsMargins(5, 0, 0, 5)
        button_grid.setColumnStretch(0, 1)
        button_grid.setSpacing(4)

        self.faceColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_face_color,
            tooltip='click to set current face color',
        )
        self._on_current_face_color_change()
        self.edgeColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_edge_color,
            tooltip='click to set current edge color',
        )
        self._on_current_edge_color_change()
        self.faceColorEdit.color_changed.connect(self.changeFaceColor)
        self.edgeColorEdit.color_changed.connect(self.changeEdgeColor)

        text_disp_cb = QCheckBox()
        text_disp_cb.setToolTip('toggle text visibility')
        text_disp_cb.setChecked(self.layer.text.visible)
        text_disp_cb.stateChanged.connect(self.change_text_visibility)
        self.textDispCheckBox = text_disp_cb

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addLayout(button_grid, 0, 0, 1, 2)
        self.grid_layout.addWidget(QLabel('opacity:'), 1, 0)
        self.grid_layout.addWidget(self.opacitySlider, 1, 1)
        self.grid_layout.addWidget(QLabel('edge width:'), 2, 0)
        self.grid_layout.addWidget(self.widthSlider, 2, 1)
        self.grid_layout.addWidget(QLabel('blending:'), 3, 0)
        self.grid_layout.addWidget(self.blendComboBox, 3, 1)
        self.grid_layout.addWidget(QLabel('face color:'), 4, 0)
        self.grid_layout.addWidget(self.faceColorEdit, 4, 1)
        self.grid_layout.addWidget(QLabel('edge color:'), 5, 0)
        self.grid_layout.addWidget(self.edgeColorEdit, 5, 1)
        self.grid_layout.addWidget(QLabel('display text:'), 6, 0)
        self.grid_layout.addWidget(self.textDispCheckBox, 6, 1)
        self.grid_layout.setRowStretch(7, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setSpacing(4)

    def mouseMoveEvent(self, event):
        """On mouse move, update layer mode status.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self.layer.status = str(self.layer.mode)

    def _on_mode_change(self, event):
        """"Update ticks in checkbox widgets when shapes layer mode changed.

        Available modes for shapes layer are:
        * SELECT
        * DIRECT
        * PAN_ZOOM
        * ADD_RECTANGLE
        * ADD_ELLIPSE
        * ADD_LINE
        * ADD_PATH
        * ADD_POLYGON
        * VERTEX_INSERT
        * VERTEX_REMOVE

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not ADD, PAN_ZOOM, or SELECT.
        """
        mode_buttons = {
            Mode.SELECT: self.select_button,
            Mode.DIRECT: self.direct_button,
            Mode.PAN_ZOOM: self.panzoom_button,
            Mode.ADD_RECTANGLE: self.rectangle_button,
            Mode.ADD_ELLIPSE: self.ellipse_button,
            Mode.ADD_LINE: self.line_button,
            Mode.ADD_PATH: self.path_button,
            Mode.ADD_POLYGON: self.polygon_button,
            Mode.VERTEX_INSERT: self.vertex_insert_button,
            Mode.VERTEX_REMOVE: self.vertex_remove_button,
        }

        if event.mode in mode_buttons:
            mode_buttons[event.mode].setChecked(True)
        else:
            raise ValueError(f"Mode '{event.mode}'not recognized")

    def changeFaceColor(self, color: np.ndarray):
        """Change face color of shapes.

        Parameters
        ----------
        color : np.ndarray
            Face color for shapes, color name or hex string.
            Eg: 'white', 'red', 'blue', '#00ff00', etc.
        """
        with self.layer.events.current_face_color.blocker():
            self.layer.current_face_color = color

    def changeEdgeColor(self, color: np.ndarray):
        """Change edge color of shapes.

        Parameters
        ----------
        color : np.ndarray
            Edge color for shapes, color name or hex string.
            Eg: 'white', 'red', 'blue', '#00ff00', etc.
        """
        with self.layer.events.current_edge_color.blocker():
            self.layer.current_edge_color = color

    def changeWidth(self, value):
        """Change edge line width of shapes on the layer model.

        Parameters
        ----------
        value : float
            Line width of shapes.
        """
        self.layer.current_edge_width = float(value) / 2

    def changeOpacity(self, value):
        """Change opacity value of shapes on the layer model.

        Parameters
        ----------
        value : float
            Opacity value for shapes.
            Input range 0 - 100 (transparent to fully opaque).
        """
        with self.layer.events.blocker(self._on_opacity_change):
            self.layer.opacity = value / 100

    def change_text_visibility(self, state):
        """Toggle the visibiltiy of the text.

        Parameters
        ----------
        state : QCheckBox
            Checkbox indicating if text is visible.
        """
        if state == Qt.Checked:
            self.layer.text.visible = True
        else:
            self.layer.text.visible = False

    def _on_text_visibility_change(self, event):
        """Receive layer model text visibiltiy change change event and update checkbox.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        with self.layer.text.events.visible.blocker():
            self.textDispCheckBox.setChecked(self.layer.text.visible)

    def _on_edge_width_change(self, event=None):
        """Receive layer model edge line width change event and update slider.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        with self.layer.events.edge_width.blocker():
            value = self.layer.current_edge_width
            value = np.clip(int(2 * value), 0, 40)
            self.widthSlider.setValue(value)

    def _on_current_edge_color_change(self, event=None):
        """Receive layer model edge color change event and update color swatch.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        with qt_signals_blocked(self.edgeColorEdit):
            self.edgeColorEdit.setColor(self.layer.current_edge_color)

    def _on_current_face_color_change(self, event=None):
        """Receive layer model face color change event and update color swatch.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        with qt_signals_blocked(self.faceColorEdit):
            self.faceColorEdit.setColor(self.layer.current_face_color)

    def _on_editable_change(self, event=None):
        """Receive layer model editable change event & enable/disable buttons.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        disable_with_opacity(
            self,
            [
                'select_button',
                'direct_button',
                'rectangle_button',
                'ellipse_button',
                'line_button',
                'path_button',
                'polygon_button',
                'vertex_remove_button',
                'vertex_insert_button',
                'delete_button',
                'move_back_button',
                'move_front_button',
            ],
            self.layer.editable,
        )

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()
