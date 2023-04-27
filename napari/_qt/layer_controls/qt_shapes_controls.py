from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QButtonGroup, QCheckBox, QGridLayout

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.utils import (
    qt_signals_blocked,
    set_widgets_enabled_with_opacity,
)
from napari._qt.widgets._slider_compat import QSlider
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari._qt.widgets.qt_mode_buttons import (
    QtModePushButton,
    QtModeRadioButton,
)
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
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for shapes layer modes
        (SELECT, DIRECT, PAN_ZOOM, ADD_RECTANGLE, ADD_ELLIPSE, ADD_LINE,
        ADD_PATH, ADD_POLYGON, VERTEX_INSERT, VERTEX_REMOVE).
    delete_button : qtpy.QtWidgets.QtModePushButton
        Button to delete selected shapes
    direct_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select individual vertices in shapes.
    edgeColorEdit : QColorSwatchEdit
        Widget allowing user to set edge color of points.
    ellipse_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add ellipses to shapes layer.
    faceColorEdit : QColorSwatchEdit
        Widget allowing user to set face color of points.
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
    polygon_lasso_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add polygons to shapes layer with a lasso tool.
    rectangle_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add rectangles to shapes layer.
    select_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select shapes.
    textDispCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if text should be displayed
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

    layer: 'napari.layers.Shapes'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        self.layer.events.mode.connect(self._on_mode_change)
        self.layer.events.edge_width.connect(self._on_edge_width_change)
        self.layer.events.current_edge_color.connect(
            self._on_current_edge_color_change
        )
        self.layer.events.current_face_color.connect(
            self._on_current_face_color_change
        )
        self.layer.events.editable.connect(self._on_editable_or_visible_change)
        self.layer.events.visible.connect(self._on_editable_or_visible_change)
        self.layer.text.events.visible.connect(self._on_text_visibility_change)

        sld = QSlider(Qt.Orientation.Horizontal)
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

        def _radio_button(
            parent,
            btn_name,
            mode,
            action_name,
            extra_tooltip_text='',
            **kwargs,
        ):
            """
            Convenience local function to create a RadioButton and bind it to
            an action at the same time.

            Parameters
            ----------
            parent : Any
                Parent of the generated QtModeRadioButton
            btn_name : str
                name fo the button
            mode : Enum
                Value Associated to current button
            action_name : str
                Action triggered when button pressed
            extra_tooltip_text : str
                Text you want added after the automatic tooltip set by the
                action manager
            **kwargs:
                Passed to QtModeRadioButton

            Returns
            -------
            button: QtModeRadioButton
                button bound (or that will be bound to) to action `action_name`

            Notes
            -----
            When shortcuts are modifed/added/removed via the action manager, the
            tooltip will be updated to reflect the new shortcut.
            """
            action_name = f'napari:{action_name}'
            btn = QtModeRadioButton(parent, btn_name, mode, **kwargs)
            action_manager.bind_button(
                action_name,
                btn,
                extra_tooltip_text='',
            )
            return btn

        self.select_button = _radio_button(
            layer, 'select', Mode.SELECT, "activate_select_mode"
        )

        self.direct_button = _radio_button(
            layer, 'direct', Mode.DIRECT, "activate_direct_mode"
        )

        self.panzoom_button = _radio_button(
            layer,
            'pan',
            Mode.PAN_ZOOM,
            "activate_shapes_pan_zoom_mode",
            extra_tooltip_text=trans._('(or hold Space)'),
            checked=True,
        )

        self.rectangle_button = _radio_button(
            layer,
            'rectangle',
            Mode.ADD_RECTANGLE,
            "activate_add_rectangle_mode",
        )
        self.ellipse_button = _radio_button(
            layer,
            'ellipse',
            Mode.ADD_ELLIPSE,
            "activate_add_ellipse_mode",
        )

        self.line_button = _radio_button(
            layer, 'line', Mode.ADD_LINE, "activate_add_line_mode"
        )
        self.path_button = _radio_button(
            layer, 'path', Mode.ADD_PATH, "activate_add_path_mode"
        )
        self.polygon_button = _radio_button(
            layer,
            'polygon',
            Mode.ADD_POLYGON,
            "activate_add_polygon_mode",
        )
        self.polygon_lasso_button = _radio_button(
            layer,
            'polygon_lasso',
            Mode.ADD_POLYGON_LASSO,
            "activate_add_polygon_lasso_mode",
        )
        self.polygon_lasso_tablet_button = _radio_button(
            layer,
            'polygon_lasso_tablet',
            Mode.ADD_POLYGON_LASSO_TABLET,
            "activate_add_polygon_lasso_tablet_mode",
        )
        self.vertex_insert_button = _radio_button(
            layer,
            'vertex_insert',
            Mode.VERTEX_INSERT,
            "activate_vertex_insert_mode",
        )
        self.vertex_remove_button = _radio_button(
            layer,
            'vertex_remove',
            Mode.VERTEX_REMOVE,
            "activate_vertex_remove_mode",
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
                "Delete selected shapes ({shortcut})",
                shortcut=Shortcut('Backspace').platform,
            ),
        )

        self._EDIT_BUTTONS = (
            self.select_button,
            self.direct_button,
            self.rectangle_button,
            self.ellipse_button,
            self.line_button,
            self.path_button,
            self.polygon_button,
            self.polygon_lasso_button,
            self.polygon_lasso_tablet_button,
            self.vertex_remove_button,
            self.vertex_insert_button,
            self.delete_button,
            self.move_back_button,
            self.move_front_button,
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
        self.button_group.addButton(self.polygon_lasso_button)
        self.button_group.addButton(self.polygon_lasso_tablet_button)
        self.button_group.addButton(self.vertex_insert_button)
        self.button_group.addButton(self.vertex_remove_button)
        self._on_editable_or_visible_change()

        button_grid = QGridLayout()
        button_grid.addWidget(self.vertex_remove_button, 0, 1)
        button_grid.addWidget(self.vertex_insert_button, 0, 2)
        button_grid.addWidget(self.delete_button, 0, 3)
        button_grid.addWidget(self.direct_button, 0, 4)
        button_grid.addWidget(self.select_button, 0, 5)
        button_grid.addWidget(self.panzoom_button, 0, 6)
        button_grid.addWidget(self.move_back_button, 0, 7)
        button_grid.addWidget(self.move_front_button, 1, 0)
        button_grid.addWidget(self.ellipse_button, 1, 1)
        button_grid.addWidget(self.rectangle_button, 1, 2)
        button_grid.addWidget(self.polygon_button, 1, 3)
        button_grid.addWidget(self.polygon_lasso_button, 1, 4)
        button_grid.addWidget(self.polygon_lasso_tablet_button, 1, 5)
        button_grid.addWidget(self.line_button, 1, 6)
        button_grid.addWidget(self.path_button, 1, 7)
        button_grid.setContentsMargins(5, 0, 0, 5)
        button_grid.setColumnStretch(0, 1)
        button_grid.setSpacing(4)

        self.faceColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_face_color,
            tooltip=trans._('click to set current face color'),
        )
        self._on_current_face_color_change()
        self.edgeColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_edge_color,
            tooltip=trans._('click to set current edge color'),
        )
        self._on_current_edge_color_change()
        self.faceColorEdit.color_changed.connect(self.changeFaceColor)
        self.edgeColorEdit.color_changed.connect(self.changeEdgeColor)

        text_disp_cb = QCheckBox()
        text_disp_cb.setToolTip(trans._('toggle text visibility'))
        text_disp_cb.setChecked(self.layer.text.visible)
        text_disp_cb.stateChanged.connect(self.change_text_visibility)
        self.textDispCheckBox = text_disp_cb

        self.layout().addRow(button_grid)
        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(trans._('edge width:'), self.widthSlider)
        self.layout().addRow(trans._('blending:'), self.blendComboBox)
        self.layout().addRow(trans._('face color:'), self.faceColorEdit)
        self.layout().addRow(trans._('edge color:'), self.edgeColorEdit)
        self.layout().addRow(trans._('display text:'), self.textDispCheckBox)

    def _on_mode_change(self, event):
        """Update ticks in checkbox widgets when shapes layer mode changed.

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
            Mode.ADD_POLYGON_LASSO: self.polygon_lasso_button,
            Mode.ADD_POLYGON_LASSO_TABLET: self.polygon_lasso_tablet_button,
            Mode.VERTEX_INSERT: self.vertex_insert_button,
            Mode.VERTEX_REMOVE: self.vertex_remove_button,
        }

        if event.mode in mode_buttons:
            mode_buttons[event.mode].setChecked(True)
        elif event.mode != Mode.TRANSFORM:
            raise ValueError(
                trans._("Mode '{mode}'not recognized", mode=event.mode)
            )

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

    def _on_editable_or_visible_change(self):
        """Receive layer model editable/visible change event & enable/disable buttons."""
        set_widgets_enabled_with_opacity(
            self,
            self._EDIT_BUTTONS,
            self.layer.editable and self.layer.visible,
        )

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()
