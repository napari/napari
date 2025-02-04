from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox
from superqt import QLabeledSlider

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.layers.shapes._shapes_constants import Mode
from napari.utils.action_manager import action_manager
from napari.utils.events import disconnect_events
from napari.utils.interactions import Shortcut
from napari.utils.translations import trans

if TYPE_CHECKING:
    import napari.layers


class QtShapesControls(QtLayerControls):
    """Qt view and controls for the napari Shapes layer.

    Parameters
    ----------
    layer : napari.layers.Shapes
        An instance of a napari Shapes layer.

    Attributes
    ----------
    layer : napari.layers.Shapes
        An instance of a napari Shapes layer.
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for shapes layer modes
        (SELECT, DIRECT, PAN_ZOOM, ADD_RECTANGLE, ADD_ELLIPSE, ADD_LINE,
        ADD_POLYLINE, ADD_PATH, ADD_POLYGON, VERTEX_INSERT, VERTEX_REMOVE,
        TRANSFORM).
    delete_button : qtpy.QtWidgets.QtModePushButton
        Button to delete selected shapes
    direct_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select individual vertices in shapes.
    edgeColorEdit : QColorSwatchEdit
        Widget allowing user to set edge color of points.
    ellipse_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add ellipses to shapes layer.
    faceColorEdit : QColorSwatchEdit
        Widget allowing user to set face color of points.
    line_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add lines to shapes layer.
    polyline_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add polylines to shapes layer.
    move_back_button : qtpy.QtWidgets.QtModePushButton
        Button to move selected shape(s) to the back.
    move_front_button : qtpy.QtWidgets.QtModePushButton
        Button to move shape(s) to the front.
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to pan/zoom shapes layer.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to transform shapes layer.
    path_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add paths to shapes layer.
    polygon_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add polygons to shapes layer.
    polygon_lasso_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add polygons to shapes layer with a lasso tool.
    rectangle_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add rectangles to shapes layer.
    select_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select shapes.
    textDispCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if text should be displayed
    vertex_insert_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to insert vertex into shape.
    vertex_remove_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to remove vertex from shapes.
    widthSlider : qtpy.QtWidgets.QSlider
        Slider controlling line edge width of shapes.

    Raises
    ------
    ValueError
        Raise error if shapes mode is not recognized.
    """

    layer: 'napari.layers.Shapes'
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_shapes_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_shapes_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        self.layer.events.edge_width.connect(self._on_edge_width_change)
        self.layer.events.current_edge_color.connect(
            self._on_current_edge_color_change
        )
        self.layer.events.current_face_color.connect(
            self._on_current_face_color_change
        )
        self.layer.text.events.visible.connect(self._on_text_visibility_change)

        sld = QLabeledSlider(Qt.Orientation.Horizontal)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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
        self.widthSlider.setToolTip(
            trans._(
                'Set the edge width of currently selected shapes and any added afterwards.'
            )
        )

        self.select_button = self._radio_button(
            layer, 'select', Mode.SELECT, True, 'activate_select_mode'
        )
        self.direct_button = self._radio_button(
            layer, 'direct', Mode.DIRECT, True, 'activate_direct_mode'
        )
        self.rectangle_button = self._radio_button(
            layer,
            'rectangle',
            Mode.ADD_RECTANGLE,
            True,
            'activate_add_rectangle_mode',
        )
        self.ellipse_button = self._radio_button(
            layer,
            'ellipse',
            Mode.ADD_ELLIPSE,
            True,
            'activate_add_ellipse_mode',
        )
        self.line_button = self._radio_button(
            layer, 'line', Mode.ADD_LINE, True, 'activate_add_line_mode'
        )
        self.polyline_button = self._radio_button(
            layer,
            'polyline',
            Mode.ADD_POLYLINE,
            True,
            'activate_add_polyline_mode',
        )
        self.path_button = self._radio_button(
            layer, 'path', Mode.ADD_PATH, True, 'activate_add_path_mode'
        )
        self.polygon_button = self._radio_button(
            layer,
            'polygon',
            Mode.ADD_POLYGON,
            True,
            'activate_add_polygon_mode',
        )
        self.polygon_lasso_button = self._radio_button(
            layer,
            'polygon_lasso',
            Mode.ADD_POLYGON_LASSO,
            True,
            'activate_add_polygon_lasso_mode',
        )
        self.vertex_insert_button = self._radio_button(
            layer,
            'vertex_insert',
            Mode.VERTEX_INSERT,
            True,
            'activate_vertex_insert_mode',
        )
        self.vertex_remove_button = self._radio_button(
            layer,
            'vertex_remove',
            Mode.VERTEX_REMOVE,
            True,
            'activate_vertex_remove_mode',
        )

        self.move_front_button = QtModePushButton(
            layer,
            'move_front',
            slot=self.layer.move_to_front,
            tooltip=trans._('Move to front'),
        )
        action_manager.bind_button(
            'napari:move_shapes_selection_to_front', self.move_front_button
        )
        self.move_back_button = QtModePushButton(
            layer,
            'move_back',
            slot=self.layer.move_to_back,
            tooltip=trans._('Move to back'),
        )
        action_manager.bind_button(
            'napari:move_shapes_selection_to_back', self.move_back_button
        )
        self.delete_button = QtModePushButton(
            layer,
            'delete_shape',
            slot=self.layer.remove_selected,
            tooltip=trans._(
                'Delete selected shapes ({shortcut})',
                shortcut=Shortcut('Backspace').platform,
            ),
        )
        self._EDIT_BUTTONS += (
            self.delete_button,
            self.move_back_button,
            self.move_front_button,
        )
        self._on_editable_or_visible_change()

        self.button_grid.addWidget(self.move_back_button, 0, 0)
        self.button_grid.addWidget(self.vertex_remove_button, 0, 1)
        self.button_grid.addWidget(self.vertex_insert_button, 0, 2)
        self.button_grid.addWidget(self.delete_button, 0, 3)
        self.button_grid.addWidget(self.direct_button, 0, 4)
        self.button_grid.addWidget(self.select_button, 0, 5)
        self.button_grid.addWidget(self.move_front_button, 1, 0)
        self.button_grid.addWidget(self.ellipse_button, 1, 1)
        self.button_grid.addWidget(self.rectangle_button, 1, 2)
        self.button_grid.addWidget(self.polygon_button, 1, 3)
        self.button_grid.addWidget(self.polygon_lasso_button, 1, 4)
        self.button_grid.addWidget(self.line_button, 1, 5)
        self.button_grid.addWidget(self.polyline_button, 1, 6)
        self.button_grid.addWidget(self.path_button, 1, 7)
        self.button_grid.setContentsMargins(5, 0, 0, 5)
        self.button_grid.setColumnStretch(0, 1)
        self.button_grid.setSpacing(4)

        self.faceColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_face_color,
            tooltip=trans._(
                'Click to set the face color of currently selected shapes and any added afterwards.'
            ),
        )
        self._on_current_face_color_change()
        self.edgeColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_edge_color,
            tooltip=trans._(
                'Click to set the edge color of currently selected shapes and any added afterwards'
            ),
        )
        self._on_current_edge_color_change()
        self.faceColorEdit.color_changed.connect(self.changeFaceColor)
        self.edgeColorEdit.color_changed.connect(self.changeEdgeColor)

        text_disp_cb = QCheckBox()
        text_disp_cb.setToolTip(trans._('Toggle text visibility'))
        text_disp_cb.setChecked(self.layer.text.visible)
        text_disp_cb.stateChanged.connect(self.change_text_visibility)
        self.textDispCheckBox = text_disp_cb

        self.layout().addRow(self.button_grid)
        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(trans._('blending:'), self.blendComboBox)
        self.layout().addRow(trans._('edge width:'), self.widthSlider)
        self.layout().addRow(trans._('face color:'), self.faceColorEdit)
        self.layout().addRow(trans._('edge color:'), self.edgeColorEdit)
        self.layout().addRow(trans._('display text:'), self.textDispCheckBox)

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
        self.layer.current_edge_width = float(value)

    def change_text_visibility(self, state):
        """Toggle the visibility of the text.

        Parameters
        ----------
        state : int
            Integer value of Qt.CheckState that indicates the check state of textDispCheckBox
        """
        self.layer.text.visible = Qt.CheckState(state) == Qt.CheckState.Checked

    def _on_mode_change(self, event):
        """Update ticks in checkbox widgets when shapes layer mode changed.

        Available modes for shapes layer are:
        * SELECT
        * DIRECT
        * PAN_ZOOM
        * ADD_RECTANGLE
        * ADD_ELLIPSE
        * ADD_LINE
        * ADD_POLYLINE
        * ADD_PATH
        * ADD_POLYGON
        * ADD_POLYGON_LASSO
        * VERTEX_INSERT
        * VERTEX_REMOVE
        * TRANSFORM

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not one of the available modes.
        """
        super()._on_mode_change(event)

    def _on_text_visibility_change(self):
        """Receive layer model text visibiltiy change change event and update checkbox."""
        with self.layer.text.events.visible.blocker():
            self.textDispCheckBox.setChecked(self.layer.text.visible)

    def _on_edge_width_change(self):
        """Receive layer model edge line width change event and update slider."""
        with self.layer.events.edge_width.blocker():
            value = self.layer.current_edge_width
            value = np.clip(int(value), 0, 40)
            self.widthSlider.setValue(value)

    def _on_current_edge_color_change(self):
        """Receive layer model edge color change event and update color swatch."""
        with qt_signals_blocked(self.edgeColorEdit):
            self.edgeColorEdit.setColor(self.layer.current_edge_color)

    def _on_current_face_color_change(self):
        """Receive layer model face color change event and update color swatch."""
        with qt_signals_blocked(self.faceColorEdit):
            self.faceColorEdit.setColor(self.layer.current_face_color)

    def _on_ndisplay_changed(self):
        self.layer.editable = self.ndisplay == 2
        super()._on_ndisplay_changed()

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()
